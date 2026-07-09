#!/usr/bin/env python3
"""
moe_trace_hf.py -- extract per-token per-layer routed-expert traces from a real
MoE checkpoint, for the h/A measurement (docs/MOE_LOCALITY_RESEARCH.md open
items 1-3).

WHAT IT MEASURES (and what it can't)
  The 100 tok/s question needs two model-fit numbers nobody has published:
    h : the BANDWIDTH-saving cache hit rate -- the fraction of per-token routed
        expert BYTES that do NOT cross the flash interface because the expert
        is already DRAM-resident.  (Prefetch hits hide latency but still move
        bytes; only residency hits save bandwidth.)
    U : the K-token verify-pass expert-union factor -- an MTP chain of K
        drafted tokens streams union(experts of the K positions), not K*topk;
        U(K) = |union| / topk is the amortization penalty curve.
  This harness measures both ON A PROXY: the largest instrumentable open MoE
  that fits this machine (default allenai/OLMoE-1B-7B-0924-Instruct: 64 experts
  x top-8 x 16 layers, native transformers support with output_router_logits).
  GLM-5.2's own routing stats remain unpublished; the harness reruns unchanged
  on any HF MoE that returns router logits.  Proxy caveat goes in the doc.

OUTPUT
  build/moe_trace/<model_tag>/trace_<workload>.npz
    ids   : int16 [n_tokens, n_layers, topk]  selected expert ids per position
    meta  : json  (model, n_experts, topk, n_layers, workload, prompts)

Run:  tools/moevenv/bin/python tools/moe_trace_hf.py [--model M] [--max-new N]
"""
import argparse, json, os, sys
import numpy as np

WORKLOADS = {
    # deliberately different token statistics: chat / code / math-ish
    "chat": [
        "Tell me about the history of the transistor and why it mattered.",
        "My sourdough starter smells like acetone. What is happening and how do I fix it?",
        "Summarize the plot of Hamlet in a few paragraphs, then discuss its themes.",
        "What should I consider when choosing between renting and buying a home?",
    ],
    "code": [
        "Write a Python function that parses an ELF file header and prints the section table.",
        "Implement an LRU cache in C with O(1) get and put, then explain the data structures.",
        "Given this SQL schema for orders and customers, write a query for the top 10 customers by revenue per quarter.",
        "Write a Verilog module for a parameterizable gray-code FIFO pointer synchronizer.",
    ],
    "math": [
        "Prove that the square root of 2 is irrational, step by step.",
        "A train leaves at 3pm at 60 km/h; another at 4pm at 90 km/h on the same track. When does the second catch the first? Show the algebra.",
        "Compute the integral of x^2 * e^(-x) from 0 to infinity, showing each integration by parts.",
        "Explain and derive the closed form of the Fibonacci sequence.",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924-Instruct")
    ap.add_argument("--max-new", type=int, default=256, help="decode tokens per prompt")
    ap.add_argument("--out", default="build/moe_trace")
    ap.add_argument("--workloads", default="chat,code,math")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()

    cfg = model.config
    n_exp = getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts")
    topk = getattr(cfg, "num_experts_per_tok", None) or getattr(cfg, "num_experts_per_token")
    tag = args.model.split("/")[-1]
    print(f"model={args.model} experts={n_exp} topk={topk} layers={cfg.num_hidden_layers}")

    os.makedirs(os.path.join(args.out, tag), exist_ok=True)

    for wl in args.workloads.split(","):
        prompts = WORKLOADS[wl]
        all_ids = []          # per token: [n_layers, topk]
        bounds = []           # request boundaries (token index where each prompt starts)
        for p in prompts:
            msgs = [{"role": "user", "content": p}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            past, cur = None, ids
            bounds.append(len(all_ids))
            with torch.no_grad():
                for step in range(args.max_new):
                    out = model(
                        cur, past_key_values=past, use_cache=True,
                        output_router_logits=True, return_dict=True,
                    )
                    past = out.past_key_values
                    nxt = out.logits[0, -1].argmax()
                    # router_logits: tuple(n_layers) of [seq, n_experts]; decode
                    # steps have seq==1 -> the LAST position is the new token.
                    step_ids = np.stack([
                        torch.topk(rl[-1].float(), topk).indices.numpy().astype(np.int16)
                        for rl in out.router_logits
                    ])  # [n_layers, topk]
                    all_ids.append(step_ids)
                    if nxt.item() == tok.eos_token_id:
                        break
                    cur = nxt.view(1, 1)
            print(f"  [{wl}] prompt done, total tokens so far: {len(all_ids)}")
        arr = np.stack(all_ids)  # [n_tokens, n_layers, topk]
        meta = dict(model=args.model, n_experts=int(n_exp), topk=int(topk),
                    n_layers=int(cfg.num_hidden_layers), workload=wl,
                    request_bounds=bounds, prompts=prompts)
        out_path = os.path.join(args.out, tag, f"trace_{wl}.npz")
        np.savez_compressed(out_path, ids=arr, meta=json.dumps(meta))
        print(f"  [{wl}] saved {arr.shape} -> {out_path}")


if __name__ == "__main__":
    main()
