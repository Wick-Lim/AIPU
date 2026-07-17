`timescale 1ns/1ps
`include "glm_fp.vh"
`include "glm_fp_pipe_lat.vh"
/* verilator lint_off DECLFILENAME */
//============================================================================
// glm_q4k_system.v  --  PRODUCTION GLM-5.2 Q4_K SINGLE-MODULE SYSTEM
//                       (glm_q4k_soc EVOLVED to route memory through the real
//                        DDR5 fabric + the weight-side DMA loader)
//                       (docs/Q4K_SYSTEM_PLAN.md §1.3 -- the largest top)
//----------------------------------------------------------------------------
// Q4_K RETARGET (vs. the prior glm_fp8_system on branch 'fp8'):  ONE contract
//   change -- the compute die
//   glm_model_fp8 -> glm_model_q4k -- drives everything.  The three weight-bus
//   families that cross the die boundary swap FP8 (8-bit codes + bf16 [128,128]
//   block scales) -> GGUF Q4_K (4-bit codes + per-super-block d/dmin/scales,
//   ceil(K/256) super-blocks): aw_col/aw_scale -> aw_q/aw_d/aw_dmin/aw_scales,
//   rw_col/rw_scale -> rw_q/rw_d/rw_dmin/rw_scales, fw_col*/fw_scale* ->
//   fw_q*/fw_d_*/fw_dmin_*/fw_scales_*.  The block-count derived params triple
//   in field and halve in count: A_NB/R_NB/FF_NB_D -> A_NSB/R_NSB/FF_NSB_D.
//   The weight_loader instance -> weight_loader_q4k (the one place the fabric
//   touches weight BYTES): its descriptor + mm_* drive move to the Q4_K super-
//   block form (mm_w_scale -> mm_w_d/mm_w_dmin/mm_w_scales; weight-row lanes go
//   8-bit -> 4-bit).  ddr5_xbar (u_xbar), expert_cache_pf, kv_cache_pager and
//   the CDC/FIFO/arbiter logic are all BYTE-AGNOSTIC -- carried through with
//   ZERO logic change (they move addresses/slots/IDs, never weight bytes; the
//   Q4_K image is just ~44% fewer bytes/weight -> smaller byte-offsets).  The
//   C8 LOOPBACK==1 path feeds the die aw_q (4-bit) not aw_col (the staged lane
//   register goes 8-bit -> 4-bit); LOOPBACK==0 (default) stays byte-identical.
//   weight_decomp is format-agnostic (unchanged).  No arithmetic is touched.
//----------------------------------------------------------------------------
// WHAT THIS IS  (vs. glm_q4k_soc)
//   glm_q4k_soc wired the VERIFIED Q4_K compute die (glm_model_q4k) to the
//   memory CONTROLLERS (expert_cache_pf + kv_cache_pager) over one Flash
//   channel, with all the die's weight/KV bytes served the SAME cycle by the
//   TB GDDR6/Flash STUBS (the verified combinational PULL contract -- the die
//   cannot be stalled mid-layer).  glm_q4k_system KEEPS that exact, verified
//   compute path and ADDS the two remaining production-fabric blocks INTO the
//   datapath so the memory now FLOWS THROUGH the real multichannel fabric:
//
//     * ddr5_xbar  -- the N_CH-channel BANKED DDR5 READ fabric.  Every DDR5
//       FAST-TIER access (the compute die's HOT-weight pulls, the expert
//       cache's resident-SLOT reads, and the weight_loader's tile fetches)
//       presents its block address to ddr5_xbar, which BANKS it across N_CH
//       independent channels (~N_CH x aggregate read BW) and returns the data.
//       The per-channel DDR5 PHY/memory is modeled by the TB stub.  This makes
//       the channel-parallel bandwidth path REAL in the elaborated datapath:
//       ddr5_xbar's response counters advance as the token is generated.
//
//     * weight_loader -- the WEIGHT-side DMA / pull master for glm_matmul_q4k.
//       It is driven on the HOT / REPRESENTATIVE weight tile: at each compute
//       launch it loads one tile descriptor (Q4_K super-block scale headers +
//       4-bit weight rows) from a fast staging tier and DRIVES the matmul pull stream
//       (mm_start / mm_w_scale / mm_w_row / mm_in_valid).  Its tile-fetch
//       ADDRESSES are MIRRORED into ddr5_xbar, realizing the multichannel
//       bandwidth for the weight stream.
//
//============================================================================
// BLOCK DIAGRAM  (one decode step)
//
//   HOST ─ start/prompt/pos/s_len ─▶┌───────────── glm_q4k_system ───────────┐
//                                   │  ┌──────────────┐                       │
//                                   │  │ glm_model_q4k│  hot-weight PULLS ─────┼─▶ GDDR6 stub (compute bytes)
//                                   │  │ Q4_K COMPUTE │  (em/gn/aw/rw/fw/fn/lw)│        │  (addr mirror)
//                                   │  │  (verified)  │  kc_* KV read ────────┐│        ▼
//                                   │  └──────┬───────┘                       ││  ┌───────────┐
//                                   │  mdl_start │ db_layer/fw_eidx           ││  │ DDR5 XBAR │─▶ N_CH DDR5 stub
//                                   │         ▼  ▼ (router pick)              ││  │  (banked, │   (per-channel
//                                   │  ┌────────────┐   ┌──────────────┐      ││  │   N_CH BW)│    TB memory)
//                                   │  │WEIGHT LOADER│  │ EXPERT-ISSUE │      ││  └─────▲─────┘
//                                   │  │  (DMA pull) │  │  FIFO + FSM  │      ││  slot/hot/load
//   staging-tier RAM ◀──wl_mem──────┼──┤ mm_* stream │  └──────┬───────┘      ││  addresses
//   (TB, latency-1)                 │  └─────┬───────┘         ▼              ││        │
//                                   │   load-addr mirror  ┌──────────────┐    ││  ec_resp_slot
//                                   │        └────────────│expert_cache_pf│◀──┘│        │
//                                   │                     │ (GDDR6 cache) │────┼────────┘ (resident-slot read)
//                                   │                     └──────┬───────┘    │
//                                   │   gather/append    ┌───────┴────────┐   │
//                                   │   (kc_*/per-token) │ kv_cache_pager │   │
//                                   │                    └───────┬────────┘   │
//                                   │              ┌─────────────┴────────┐   │
//                                   │              │  SINGLE FLASH ARBITER │───┼──▶ Flash stub
//                                   │              │   (demand-priority)   │   │
//                                   │              └───────────────────────┘   │
//                                   └ busy/done/next_tok/tok_valid ─▶ HOST ────┘
//
//============================================================================
// INTEGRATION MAP  (what flows through the real fabric vs. the stub)
//   THROUGH ddr5_xbar (multichannel DDR5 fast tier; TB models per-channel mem):
//     - HOT weight pulls  : every cycle the die pulls a hot weight (em/gn/aw/
//                           rw/fw/fn/lw) a banked DDR5 read is issued (TAG_HOT).
//     - EXPERT-SLOT reads : on each expert_cache_pf demand response the resident
//                           GDDR6/DDR5 SLOT is read through the fabric (TAG_SLOT).
//     - LOADER fetches    : the weight_loader's tile-word fetches are mirrored
//                           as banked reads (TAG_LOAD).
//     All three are coalesced by a tiny priority issuer (LOAD > SLOT > HOT) onto
//     ddr5_xbar's single requester port; bank_rot stripes consecutive accepted
//     reads round-robin across channels so the N_CH bandwidth is exercised.
//   THROUGH weight_loader (the matmul weight-pull master, hot/representative tile):
//     - One descriptor per compute launch (mdl_start): Q4_K super-block scale
//       headers + 4-bit rows for a representative attention-projection tile.  It
//       drives the full mm_start/mm_w_scale/mm_w_row/mm_in_valid pull stream a
//       glm_matmul_q4k consumes -- observable, X-clean -- and its fetch addresses
//       feed the xbar.
//   STILL via the STUB (the compute MATH, unperturbed -- exactly as glm_q4k_soc):
//     - The die's actual weight CODES/SCALES + kc_* KV bytes are served the same
//       cycle by the TB GDDR6/Flash stub ports (the verified combinational pull
//       contract).  ddr5_xbar + weight_loader sit IN the datapath as the
//       bandwidth/address/pull engines and are fully exercised & counted, while
//       the verified compute is byte-for-byte unchanged.  Wiring the xbar's
//       returned bytes physically into the die is the remaining step a real PHY
//       closes; every address it needs is already presented here.
//
// STYLE: synchronous ACTIVE-HIGH reset; NO latch; NO combinational loop; header;
//   every reg reset (X-aware).  No arithmetic is reimplemented -- this is WIRING.
//============================================================================
module glm_q4k_system #(
    // ---- compute-die (glm_model_q4k) slice config -- passed straight through --
    parameter integer MODEL_DIM  = 128,
    parameter integer L          = 6,
    parameter integer N_DENSE    = 3,
    parameter integer VOCAB      = 256,
    parameter integer H_HEADS    = 4,
    parameter integer NOPE       = 16,
    parameter integer ROPE       = 16,
    parameter integer V_DIM      = 32,
    parameter integer Q_LORA     = 64,
    parameter integer KV_LORA    = 32,
    parameter integer S_MAX      = 8,
    parameter integer TOPK_ATTN  = 8,
    // attention union-slot scratch depth (mla SWIN), threaded straight to u_model
    //   (glm_model_q4k -> glm_decoder_block_q4k -> mla_attn_q4k, all of which already
    //   thread it).  Default = the model/decoder/mla default min(S_MAX,TOPK_ATTN) --
    //   BYTE-IDENTICAL at PE_M=1 (mirrors glm_model_q4k.v:75).  For PE_M>1 the caller
    //   MUST raise SWIN to the union bound min(PE_M*TOPK_ATTN, S_MAX) that mla_attn_q4k
    //   asserts (mla_attn_q4k.v:476-499); threaded here so a scaled-up batch CAN size
    //   its union scratch instead of being pinned to the PE_M=1 default.
    parameter integer SWIN       = (S_MAX < TOPK_ATTN) ? S_MAX : TOPK_ATTN,
    parameter integer THETA      = 8000000,
    parameter integer PE_N       = 4,
    parameter integer POSW       = 20,
    parameter integer N_EXPERT   = 8,
    parameter integer TOPK       = 2,
    parameter integer INTER_MOE  = 64,
    parameter integer INTER_DENSE= 256,
    parameter [31:0]  RSCALE     = 32'h40200000,
    parameter integer TN         = 4,
    parameter integer BLK        = 128,
    parameter integer LM_TN      = 4,
    parameter integer ACT_HW     = 0,   // activation HW lanes (0 = full) -- result-invariant knob
    // ---- PE_M : query tokens decoded in lockstep (batch B), threaded straight to
    //   u_model (glm_model_q4k -> decoder -> mla_attn_q4k, all already PE_M-capable).
    //   Default 1 == the committed single-token forward = NO speculation, BYTE-IDENTICAL
    //   (mirrors glm_model_q4k.v:92).  PE_M=K+1 is the batched draft-verify shape the
    //   spec composition (SPEC_COMPOSITION_DESIGN.md step 5) uses; exposed here so the
    //   system can no longer silently pin the model at PE_M=1.  NOTE (5a scope): only the
    //   PARAMETER is threaded -- the system's host-facing datapath (prompt_tok/logits/
    //   argmax/h_state) is still single-row, so the batched verify ports are wired in 5b.
    parameter integer PE_M       = 1,
    parameter integer PER_ROW_POS = 0,  // 1 = per-row query positions via pos_vec (P1.3a; model default 0)
    parameter integer PER_ROW_SLEN= 0,  // 1 = per-row causal extents via s_len_vec (P1.3d; model default 0)
    // DSA_REAL_IDX (default 0 == the committed netlist, byte-identical -- `make dsa-thread-equiv`):
    //   0: the DSA indexer is fed ZERO key-index vectors, so every key scores 0 and top-K keeps
    //      keys 0..min(S,TOPK)-1 by lower-index tie-break -- QUERY-INDEPENDENT selection.
    //   1: real query-dependent key-index vectors (proven bit-exact at the leaf by `make mla-sparse`).
    // A NO-OP while DENSE (S <= TOPK: no keys are pulled at all, mla_attn_q4k.v:165-169), which is
    // why the committed S_MAX=8/TOPK_ATTN=8 slice is identical either way. It becomes LOAD-BEARING
    // the moment S_MAX > TOPK_ATTN: at 0, a scaled-up window would attend ONLY to the first TOPK
    // tokens of the sequence, for every query, at every position -- fluent-looking and wrong.
    // Threaded so that raising S_MAX is a decision about attention, not an accident of defaults.
    parameter integer DSA_REAL_IDX = 0,
    // ---- memory-system config ----
    parameter integer CACHE_SLOTS = 4,      // GDDR6 expert-cache slots (slice)
    parameter integer FLASH_LAT   = 8,      // Flash fetch latency (doc; TB models)
    parameter integer KV_CTX      = 1024,   // logical KV context capacity (positions)
    parameter integer KV_RESIDENT = 16,     // KV ring capacity (POWER OF TWO, >= S_MAX)
    parameter integer EFIFO_DEPTH = 16,     // routed-expert request FIFO depth (POW2)
    // SELF_KV=0 (DEFAULT): BYTE-IDENTICAL core to the pre-write-back top -- the
    //   pager APPENDS the TB stub kv_row_in (host H_APPEND/H_DECAP) and the model
    //   READS its KV from the kc_ckv/kc_krope stub INPUTS; the pager's row_out stays
    //   observation-only (kv_row_out).  The die's KV loop is a TB stub.
    // SELF_KV=1 (KV_WRITEBACK_DESIGN.md step 2, L=1): CLOSE THE LOOP.  The model's
    //   committed latent (kv_lat_row, pulsed by kv_lat_valid) is APPENDED to the
    //   pager, and the pager's row_out/row_valid feed the model's kc_ckv/kc_krope/
    //   kc_valid (replacing the stub) -- so each decoded token READS prior tokens'
    //   KV from the pager and APPENDS its own: the die attends to KV it produced.
    //   Host prefill (H_APPEND) is skipped: the persistent pager already holds the
    //   prior tokens' rows (one append per decoded token, on commit).
    parameter integer SELF_KV     = 0,
    // ---- C8 loopback: physically route ddr5_xbar's returned bytes into the die ----
    //   0 = OFF (DEFAULT): BYTE-IDENTICAL to the pre-loopback module.  The die's
    //       attention-weight code lanes (aw_q) come straight from the same-cycle
    //       combinational GDDR6 stub, and ddr5_xbar's returned data stays a counted
    //       OBSERVATION-only output (the C8 gap: returned bytes not fed to the die).
    //   1 = ON: the die's aw_q Q4_K code lanes are SOURCED from ddr5_xbar's returned
    //       read data.  Because the die's weight pull is COMBINATIONAL (same-cycle,
    //       un-stallable) but the xbar answers only after ROW_LAT, the loopback
    //       STALLS the die by clock-gating (die_clk = clk & enable): on each aw beat
    //       it issues a banked DDR5 read (TAG_LBAW) encoding {layer,sel,grp,k},
    //       freezes the die until that beat's response returns, presents the returned
    //       lanes on the die's aw_q, then advances the die exactly one edge.  Every
    //       other weight/KV family still comes same-cycle from the stub (unperturbed).
    //       No glm_model_q4k edit is needed -- the die is stalled EXTERNALLY by
    //       gating its clock, which a synchronous die tolerates bit-exactly (its
    //       per-edge input trajectory is unchanged, so the committed token is
    //       identical to the LOOPBACK=0 run).  See the new local proof TB.
    parameter integer LOOPBACK    = 0,
    // ---- FAITHFUL EXPERT-MISS STALL (make the die pay the Flash memory wait) ----
    //   0 = OFF (DEFAULT): BYTE-IDENTICAL to the pre-stall module.  die_clk===clk;
    //       the compute die runs at full speed and a demand miss is only OBSERVED
    //       by expert_cache_pf on a parallel path (ec_demand_stall_cycles counts
    //       the wait, but the die never actually pays it -- cyc_per_tok is decoupled
    //       from FLASH_LAT).
    //   1 = ON: the die's clock is FROZEN (glitch-free, negedge-latched enable, the
    //       SAME external clock-gate the C8 LOOPBACK proves bit-exact) for exactly
    //       the cycles expert_cache_pf holds ec_busy -- i.e. every cycle a DEMAND
    //       MISS (or a demand miss queued behind a prefetch) is being serviced by
    //       Flash.  Because a synchronous die tolerates clock-gating bit-for-bit
    //       (its per-edge input trajectory is unchanged; the weight/KV stub still
    //       serves every family same-cycle on the edges the die actually takes), the
    //       committed token is IDENTICAL to the OFF run -- but the die now waits
    //       ~FLASH_LAT per miss, so the measured start->tok_valid latency GROWS by
    //       exactly ec_demand_stall_cycles.  No glm_model_q4k edit is needed; the
    //       cache/FIFO/Flash-arbiter all keep running on the ungated clk so the
    //       fetch that clears the stall always completes (no deadlock).
    parameter integer EXPERT_STALL = 0,
    // ---- RESIDENT weight tier (serve expert refills from the DDR-tier fabric) ----
    //   0 = OFF (DEFAULT): BYTE-IDENTICAL to the pre-RESIDENT module.  An expert
    //       cache refill (demand miss or prefetch) goes to the SINGLE FLASH
    //       CHANNEL through the §6 demand-priority arbiter, exactly as before.
    //   1 = ON: the FULL weight image (hot set + every routed expert) is DDR-tier
    //       resident (the LPDDR5X stand-in behind ddr5_xbar), so RUNTIME DECODE
    //       NEVER TOUCHES THE FLASH PATH FOR WEIGHTS: each expert_cache_pf refill
    //       is served by a REAL banked ddr5_xbar read (TAG_EFILL) -- the §10 FSM
    //       edge-detects the cache's held flash_req, presents ONE tagged read to
    //       the §8 issuer, and pulses ec_flash_done on that read's tagged
    //       response, so the refill wait becomes the DDR round-trip (ROW_LAT),
    //       not FLASH_LAT.  The §6 Flash arbiter keeps ONLY the kv_cache_pager
    //       client: the KV NVMe/Flash SPILL path stays available (a cold gather
    //       beyond the KV_RESIDENT ring still fetches from Flash), and the
    //       power-up Flash->DDR copy remains boot_loader's job (standalone,
    //       unchanged).  The expert class can then never own the Flash channel
    //       (flash_req && flash_is_expert never fires -- asserted in §10).
    //       Only WHERE the refill handshake completes changes; the cache's
    //       hit/miss decisions, the die's arithmetic and the token selection are
    //       untouched.
    parameter integer RESIDENT     = 0,
    // ---- DDR5 fast-tier fabric (ddr5_xbar) config ----
    parameter integer DDR_NCH     = 4,      // DDR5 channels (POWER OF TWO)
    parameter integer DDR_ADDR_W  = 32,     // block-address width into the fabric
    parameter integer DDR_DATA_W  = 256,    // DDR5 read-data width (one beat)
    parameter integer DDR_TAG_W   = 8,      // in-flight requester tag width
    parameter integer DDR_ROW_LAT = 10,     // per-channel read latency (TB models)
    parameter integer DDR_RESP_QD = 4,      // per-channel response FIFO depth
    // ---- weight_loader (matmul weight-pull DMA) config ----
    parameter integer WL_KMAX     = 256,    // max K the loader can stream
    parameter integer WL_ADDR_W   = 24,     // loader staging-memory address width
    parameter integer LOADER_KLEN = MODEL_DIM, // representative tile K length (<= WL_KMAX)
    // ---- weight-decompression (die-side Flash-BW lever; §DECOMP) ----
    //   0 = OFF: the wl_mem backing image is the RAW weight-word stream, fed to
    //            weight_loader unchanged (BYTE-IDENTICAL to the pre-DECOMP module).
    //   1 = weight_decomp (order-0 canonical Huffman): the wl_mem backing image is
    //            the LOSSLESSLY-COMPRESSED weight-word byte stream; it is streamed
    //            through weight_decomp, the decoded Q4_K bytes are re-assembled into
    //            WL_DATA_W words in the loader's sequential read order, and the
    //            loader (hence the observable Q4_K code beats) consumes DECOMPRESSED
    //            codes from that reconstruction buffer.
    parameter integer DECOMP      = 0,
    parameter integer WD_MAXLEN   = 15,     // weight_decomp: max canonical code length
    parameter integer WD_SYMW     = 9,      // weight_decomp: symbol width (0..256)
    parameter integer WD_COUNTW   = 10,     // weight_decomp: per-length count width
    parameter integer WD_AW       = 9,      // weight_decomp: table load addr width
    parameter integer WD_BUFW     = 32,     // weight_decomp: bit-buffer width
    parameter integer WD_EOB_SYM  = 256,    // weight_decomp: end-of-block symbol
    parameter integer RECON_DEPTH = 2048,   // decompressed-word reconstruction RAM depth
    // ====================================================================
    // derived (do NOT override) -- mirror glm_model_q4k's port-width derivations
    // ====================================================================
    parameter integer QK_DIM     = NOPE + ROPE,
    parameter integer IDXW       = (S_MAX <= 1) ? 1 : $clog2(S_MAX),
    parameter integer HQK        = H_HEADS * QK_DIM,
    parameter integer HNOPE      = H_HEADS * NOPE,
    parameter integer HV         = H_HEADS * V_DIM,
    parameter integer EIDXW      = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT),
    parameter integer A_KMAX     = (MODEL_DIM > Q_LORA) ?
                               ((MODEL_DIM > KV_LORA) ?
                                ((MODEL_DIM > HV) ? MODEL_DIM : HV)
                              : ((KV_LORA > HV) ? KV_LORA : HV))
                             : ((Q_LORA > KV_LORA) ?
                                ((Q_LORA > HV) ? Q_LORA : HV)
                              : ((KV_LORA > HV) ? KV_LORA : HV)),
    parameter integer A_OMAX     = (HQK > MODEL_DIM) ?
                               ((HQK > HNOPE) ?
                                 ((HQK > HV) ? HQK : HV)
                               : ((HNOPE > HV) ? HNOPE : HV))
                             : ((MODEL_DIM > HNOPE) ?
                                 ((MODEL_DIM > HV) ? MODEL_DIM : HV)
                               : ((HNOPE > HV) ? HNOPE : HV)),
    parameter integer A_NGMAX    = (A_OMAX + PE_N - 1) / PE_N,
    parameter integer A_GRPW     = (A_NGMAX <= 1) ? 1 : $clog2(A_NGMAX),
    parameter integer A_KCW      = (A_KMAX  <= 1) ? 1 : $clog2(A_KMAX),
    parameter integer FF_GWD     = $clog2(((INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM)/TN + 1),
    parameter integer FF_KMAX_D  = (INTER_DENSE > MODEL_DIM) ? INTER_DENSE : MODEL_DIM,
    parameter integer FF_KWD     = $clog2(FF_KMAX_D + 1),
    parameter integer FF_KMAX_M  = (INTER_MOE  > MODEL_DIM) ? INTER_MOE  : MODEL_DIM,
    parameter integer R_KW       = $clog2(FF_KMAX_M + 1),
    // ---- Q4_K super-block counts (256-elem super-blocks; #super-blocks per K) ----
    //   ceil(K/256), mirroring glm_model_q4k -- these size the d/dmin/scales buses
    //   (was A_NB/FF_NB_D/R_NB = ceil(K/128) prior-FP8 block counts).
    parameter integer A_NSB      = (A_KMAX    + 255) / 256,   // attention super-blocks
    parameter integer FF_NSB_D   = (FF_KMAX_D + 255) / 256,   // dense FFN super-blocks
    parameter integer R_NSB      = (FF_KMAX_M + 255) / 256,   // router super-blocks
    parameter integer LAYW       = (L     <= 1) ? 1 : $clog2(L),
    parameter integer TOKW       = (VOCAB <= 1) ? 1 : $clog2(VOCAB),
    parameter integer DIMW       = (MODEL_DIM <= 1) ? 1 : $clog2(MODEL_DIM),
    parameter integer NVTILE     = VOCAB / LM_TN,
    parameter integer VTW        = (NVTILE <= 1) ? 1 : $clog2(NVTILE),
    // ---- memory-system derived ----
    parameter integer ROW_BITS   = (KV_LORA + ROPE) * 16,             // one latent row
    parameter integer KVPOSW     = (KV_CTX <= 1) ? 1 : $clog2(KV_CTX),// pager logical pos
    // ---- KV_WRITEBACK_DESIGN.md step 3: per-(layer,position) keying -------------
    //   The pager keys the flat (layer,pos) index via NSEQ INDEPENDENT ring windows,
    //   one per LAYER (seq == db_layer).  SELF_KV=1 -> KV_NSEQ=L windows (store =
    //   L*KV_RESIDENT rows); SELF_KV=0 -> KV_NSEQ=1 (single window, byte-identical
    //   to the pre-step-3 pager, seq selects forced to 0 internally).
    parameter integer KV_NSEQ    = (SELF_KV != 0) ? L : 1,
    parameter integer KV_SEQW    = (KV_NSEQ <= 1) ? 1 : $clog2(KV_NSEQ),
    parameter integer CSLOTW     = (CACHE_SLOTS <= 1) ? 1 : $clog2(CACHE_SLOTS),
    parameter integer EFW        = (EFIFO_DEPTH <= 1) ? 1 : $clog2(EFIFO_DEPTH),
    // ---- fabric / loader derived ----
    parameter integer CH_IDX_W   = (DDR_NCH <= 1) ? 1 : $clog2(DDR_NCH),
    parameter integer WL_PE_N    = PE_N,
    // Q4_K staging word: a super-block header packs into [127:0] (d/dmin/scales)
    // and a nibble-code beat into [4*PE_N-1:0]; DATA_W stays 256 (== ddr5 beat).
    parameter integer WL_DATA_W  = 256,
    parameter integer WL_NSB     = (WL_KMAX + 255) / 256,   // #super-blocks per tile
    parameter integer WL_KW      = $clog2(WL_KMAX + 1),
    parameter integer WL_SBW     = $clog2(WL_NSB + 1),
    parameter integer WL_D_W     = 16*WL_PE_N*WL_NSB,       // packed d / dmin bus width
    parameter integer WL_SCL_W   = 96*WL_PE_N*WL_NSB        // packed 6-bit scales bus
)(
    input  wire                          clk,
    input  wire                          rst,        // sync, active-high

    //========================== HOST interface (USB-C bridge) ===============
    input  wire                          start,
    input  wire [TOKW-1:0]               prompt_tok,
    input  wire [POSW-1:0]               start_pos,
    input  wire [IDXW:0]                 s_len,
    output reg                           busy,
    output reg                           done,
    output reg  [TOKW-1:0]               next_tok,
    output reg                           tok_valid,
    output wire [VOCAB*16-1:0]           logits,

    //========================== GDDR6 HOT-weight STUBS ======================
    output wire                          em_req,
    output wire [TOKW-1:0]               em_tok,
    output wire [DIMW-1:0]               em_idx,
    input  wire [15:0]                   em_val,
    output wire [LAYW-1:0]               db_layer,
    output wire                          idx_fresh,
    output wire [LAYW-1:0]               idx_win,
    output wire                          gn_req,
    output wire                          gn_which,
    output wire [DIMW-1:0]               gn_idx,
    input  wire [15:0]                   gn_val,
    // ---- attention weights (Q4_K: 4-bit codes + per-super-block d/dmin/scales) ----
    output wire                          aw_req,
    output wire [3:0]                    aw_sel,
    output wire [A_GRPW-1:0]             aw_grp,
    output wire [A_KCW-1:0]              aw_k,
    input  wire [PE_N*4-1:0]             aw_q,       // PE_N Q4_K 4-bit weight lanes
    input  wire [16*PE_N*A_NSB-1:0]      aw_d,       // fp16 d per (col,super-block)
    input  wire [16*PE_N*A_NSB-1:0]      aw_dmin,    // fp16 dmin
    input  wire [96*PE_N*A_NSB-1:0]      aw_scales,  // 6-bit scales
    // ---- MoE router W_g (Q4_K: 4-bit codes + d/dmin/scales) ----
    output wire                          rw_req,
    output wire [R_KW-1:0]               rw_k,
    input  wire [4*N_EXPERT-1:0]         rw_q,       // N_EXPERT Q4_K 4-bit lanes
    input  wire [16*N_EXPERT*R_NSB-1:0]  rw_d,       // fp16 d
    input  wire [16*N_EXPERT*R_NSB-1:0]  rw_dmin,    // fp16 dmin
    input  wire [96*N_EXPERT*R_NSB-1:0]  rw_scales,  // 6-bit scales
    // ---- FFN expert weights (Q4_K) -- routed pass through the cache ----
    output wire                          fw_req,
    output wire [1:0]                    fw_sel,
    output wire [FF_GWD-1:0]             fw_grp,
    output wire [FF_KWD-1:0]             fw_k,
    output wire                          fw_shared,
    output wire [EIDXW-1:0]              fw_eidx,
    input  wire [4*TN-1:0]               fw_q,       // GATE/DOWN Q4_K 4-bit lanes
    input  wire [4*TN-1:0]               fw_q_up,    // UP companion Q4_K 4-bit lanes
    input  wire [16*TN*FF_NSB_D-1:0]     fw_d_g,     // GATE/DOWN fp16 d
    input  wire [16*TN*FF_NSB_D-1:0]     fw_dmin_g,  // GATE/DOWN fp16 dmin
    input  wire [96*TN*FF_NSB_D-1:0]     fw_scales_g,// GATE/DOWN 6-bit scales
    input  wire [16*TN*FF_NSB_D-1:0]     fw_d_u,     // UP fp16 d
    input  wire [16*TN*FF_NSB_D-1:0]     fw_dmin_u,  // UP fp16 dmin
    input  wire [96*TN*FF_NSB_D-1:0]     fw_scales_u,// UP 6-bit scales
    output wire                          fn_req,
    output wire [DIMW-1:0]               fn_idx,
    input  wire [15:0]                   fn_val,
    output wire                          lw_req,
    output wire [VTW-1:0]                lw_vtile,
    output wire [DIMW-1:0]               lw_k,
    input  wire [LM_TN*16-1:0]           lw_col,
    input  wire [KV_LORA*16-1:0]         kc_ckv,
    input  wire [ROPE*16-1:0]            kc_krope,
    output wire                          kc_req,
    output wire [IDXW-1:0]               kc_idx,

    //========================== KV append (latent ROW source) ===============
    output wire [KVPOSW-1:0]             kv_row_sel,
    input  wire [ROW_BITS-1:0]           kv_row_in,

    //========================== SINGLE FLASH CHANNEL (to PHY/TB) ============
    output wire                          flash_req,
    output wire                          flash_is_expert,
    output wire [EIDXW-1:0]              flash_expert_id,
    output wire [KVPOSW-1:0]             flash_row_idx,
    input  wire                          flash_done,
    input  wire [ROW_BITS-1:0]           flash_row,

    //========================== expert prefetch hint (optional) =============
    input  wire                          pf_valid,
    input  wire [EIDXW-1:0]              pf_expert_id,

    //========================== DDR5 fabric channels (to per-channel TB stub) =
    output wire [DDR_NCH-1:0]            mem_req_valid,
    input  wire [DDR_NCH-1:0]            mem_req_ready,
    output wire [DDR_NCH*DDR_ADDR_W-1:0] mem_req_addr,
    output wire [DDR_NCH*DDR_TAG_W-1:0]  mem_req_tag,
    input  wire [DDR_NCH-1:0]            mem_resp_valid,
    output wire [DDR_NCH-1:0]            mem_resp_ready,
    input  wire [DDR_NCH*DDR_DATA_W-1:0] mem_resp_data,
    input  wire [DDR_NCH*DDR_TAG_W-1:0]  mem_resp_tag,

    //========================== weight_loader staging memory (TB, latency-1) =
    output wire                          wl_mem_en,
    output wire [WL_ADDR_W-1:0]          wl_mem_addr,
    input  wire [WL_DATA_W-1:0]          wl_mem_data,

    //========================== weight-decomp table load (DECOMP>=1 only) ===
    //   Loads the canonical-Huffman tables into the internal weight_decomp.
    //   UNUSED when DECOMP==0 (the decompressor is not elaborated), so the
    //   pre-DECOMP behaviour is byte-identical and these may be left unconnected.
    input  wire                          decomp_tbl_we,
    input  wire                          decomp_tbl_sel,   // 1=symbol_table, 0=count_table
    input  wire [WD_AW-1:0]              decomp_tbl_addr,
    input  wire [WD_COUNTW-1:0]          decomp_tbl_wdata,

    //========================== observability ===============================
    output wire [TOKW-1:0]               argmax_o,
    output wire [MODEL_DIM*16-1:0]       h_state,
    output wire                          mdl_busy,
    // expert-cache stats / slot
    output wire                          ec_resp_valid,
    output wire                          ec_hit,
    output wire [CSLOTW-1:0]             ec_resp_slot,
    output wire                          ec_busy,
    output wire [31:0]                   ec_hit_count,
    output wire [31:0]                   ec_miss_count,
    output wire [31:0]                   ec_demand_stall_cycles,
    output wire [31:0]                   ec_pf_issued,
    output wire [31:0]                   ec_pf_hit,
    // KV-pager stats
    output wire                          kv_row_valid,
    output wire [ROW_BITS-1:0]           kv_row_out,
    output wire                          kv_busy,
    // ---- KV latent WRITE-BACK exposure (from the die; KV_WRITEBACK_DESIGN step 1).
    //   Present regardless of SELF_KV (additive observation of the committed latent);
    //   under SELF_KV=1 these ALSO drive the pager append internally. ----
    output wire [ROW_BITS-1:0]           kv_lat_row,
    output wire                          kv_lat_valid,
    output wire [KVPOSW-1:0]             kv_append_count,
    output wire [KVPOSW-1:0]             kv_resident_lo,
    output wire                          kv_overflowed,
    output wire [31:0]                   ec_dropped,
    // ---- DDR5 fabric + loader stats (NEW) ----
    output reg  [31:0]                   xbar_req_count,   // banked reads accepted
    output reg  [31:0]                   xbar_resp_count,  // banked reads returned
    output wire                          xbar_resp_valid,  // requester resp valid (obs)
    output wire [DDR_DATA_W-1:0]         xbar_resp_data,   // last returned beat (obs)
    output wire                          loader_busy,
    output reg  [31:0]                   loader_done_count,// tiles streamed
    output reg  [31:0]                   loader_beat_count,// weight-row beats driven
    output wire [4*WL_PE_N-1:0]          loader_w_q,       // current Q4_K weight-code row (obs)
    output wire                          loader_in_valid   // weight beat valid (obs)
);
    //========================================================================
    // 1) THE COMPUTE DIE -- glm_model_q4k (verified full Q4_K forward pass).
    //========================================================================
    wire                      mdl_done;
    wire [TOKW-1:0]           mdl_argmax;
    reg                       kc_valid_r;
    reg                       mdl_start;

    // ---- C8 loopback nets (driven in §9 generate; tied off when LOOPBACK==0) ----
    //   die_clk  : the die's (possibly gated) clock -- == clk when LOOPBACK==0.
    //   die_aw_q : the die's attention-weight Q4_K CODE lanes (4-bit) -- == the
    //              stub aw_q input when LOOPBACK==0, or the xbar-returned lanes
    //              (low PE_N*4 bits) when ==1.
    //   lb_pending / lb_req_addr / lb_accept : loopback source into the §8 issuer.
    wire                      die_clk;
    wire [PE_N*4-1:0]         die_aw_q;
    wire                      lb_pending;
    wire [DDR_ADDR_W-1:0]     lb_req_addr;
    wire                      lb_accept;

    // ---- RESIDENT=1 expert-refill nets (driven in the §10 generate; tied off
    //      when RESIDENT==0 -- the same idiom as the C8 loopback nets above) ----
    //   ef_pending / ef_id  : one banked DDR read request into the §8 issuer.
    //   ef_accept           : the issuer accepted it (clears ef_pending in §10).
    //   ec_ddr_done         : the tagged ddr5_xbar response returned -> completes
    //                         the expert cache's flash_done handshake in §6.
    wire                      ef_pending;
    wire [EIDXW-1:0]          ef_id;
    wire                      ef_accept;
    wire                      ec_ddr_done;

    //========================================================================
    // KV WRITE-BACK routing (KV_WRITEBACK_DESIGN.md step 2).
    //   mdl_kv_lat_row / mdl_kv_lat_valid : the die's committed latent (exposed by
    //     glm_model_q4k). Drives the observation ports AND, under SELF_KV=1, the
    //     pager append.
    //   mdl_kc_ckv / mdl_kc_krope / mdl_kc_valid : the model's KV READ source.
    //     SELF_KV=0 -> the stub INPUTS kc_ckv/kc_krope and the registered ack
    //       kc_valid_r (BYTE-IDENTICAL to the pre-write-back top).
    //     SELF_KV=1 -> the pager's gather response: row_out split [c_kv|k_rope]
    //       (c_kv LOW KV_LORA*16, k_rope HIGH ROPE*16 -- the pack in mla_attn_q4k /
    //       the unpack in glm_q4k_soc_ms.v:502-503) and row_valid as the ack. The
    //       pager's registered resident gather (row_valid 1 cycle after the accepted
    //       gather=kc_req) matches the kc_valid_r<=kc_req timing it replaces.
    wire [ROW_BITS-1:0]       mdl_kv_lat_row;
    wire                      mdl_kv_lat_valid;
    assign kv_lat_row   = mdl_kv_lat_row;
    assign kv_lat_valid = mdl_kv_lat_valid;

    wire [KV_LORA*16-1:0]     mdl_kc_ckv   = (SELF_KV != 0) ? kv_row_out[0          +: KV_LORA*16]
                                                            : kc_ckv;
    wire [ROPE*16-1:0]        mdl_kc_krope = (SELF_KV != 0) ? kv_row_out[KV_LORA*16 +: ROPE*16]
                                                            : kc_krope;
    wire                      mdl_kc_valid = (SELF_KV != 0) ? kv_row_valid : kc_valid_r;

    glm_model_q4k #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .SWIN(SWIN), .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT),
        .TOPK(TOPK), .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE),
        .TN(TN), .BLK(BLK), .LM_TN(LM_TN), .PE_M(PE_M), .ACT_HW(ACT_HW),
        .PER_ROW_POS(PER_ROW_POS), .PER_ROW_SLEN(PER_ROW_SLEN),
        .DSA_REAL_IDX(DSA_REAL_IDX)
    ) u_model (
        .clk(die_clk), .rst(rst),
        .start(mdl_start), .busy(mdl_busy), .done(mdl_done),
        .token_id(prompt_tok), .pos(start_pos), .s_len(s_len),
        .logits(logits), .argmax(mdl_argmax),
        .em_req(em_req), .em_tok(em_tok), .em_idx(em_idx), .em_val(em_val),
        .db_layer(db_layer), .idx_fresh(idx_fresh), .idx_win(idx_win),
        .gn_req(gn_req), .gn_which(gn_which), .gn_idx(gn_idx), .gn_val(gn_val),
        .aw_req(aw_req), .aw_sel(aw_sel), .aw_grp(aw_grp), .aw_k(aw_k),
        .aw_q(die_aw_q), .aw_d(aw_d), .aw_dmin(aw_dmin), .aw_scales(aw_scales),
        .kc_req(kc_req), .kc_idx(kc_idx), .kc_ckv(mdl_kc_ckv), .kc_krope(mdl_kc_krope),
        .kc_valid(mdl_kc_valid),
        .rw_req(rw_req), .rw_k(rw_k),
        .rw_q(rw_q), .rw_d(rw_d), .rw_dmin(rw_dmin), .rw_scales(rw_scales),
        .fw_req(fw_req), .fw_sel(fw_sel), .fw_grp(fw_grp), .fw_k(fw_k),
        .fw_shared(fw_shared), .fw_eidx(fw_eidx),
        .fw_q(fw_q), .fw_q_up(fw_q_up),
        .fw_d_g(fw_d_g), .fw_dmin_g(fw_dmin_g), .fw_scales_g(fw_scales_g),
        .fw_d_u(fw_d_u), .fw_dmin_u(fw_dmin_u), .fw_scales_u(fw_scales_u),
        .fn_req(fn_req), .fn_idx(fn_idx), .fn_val(fn_val),
        .lw_req(lw_req), .lw_vtile(lw_vtile), .lw_k(lw_k), .lw_col(lw_col),
        .h_state(h_state),
        .kv_lat_row(mdl_kv_lat_row), .kv_lat_valid(mdl_kv_lat_valid)
    );
    assign argmax_o = mdl_argmax;

    // kc_valid : 1-cycle-registered ack of kc_req (the verified read contract).
    //   Clocked by die_clk so the ack stays in the die's (possibly gated) domain --
    //   at LOOPBACK==0 die_clk===clk so this is byte-identical to the original.
    always @(posedge die_clk) begin
        if (rst) kc_valid_r <= 1'b0;
        else     kc_valid_r <= kc_req;
    end

    //========================================================================
    // 2) HOST FSM : prefill KV window -> run die -> append decode latent -> commit
    //========================================================================
    localparam [2:0] H_IDLE   = 3'd0,
                     H_APPEND = 3'd1,
                     H_RUN_W  = 3'd3,
                     H_DECAP  = 3'd4,
                     H_DONE   = 3'd5;
    reg [2:0]        hstate;
    reg [IDXW:0]     ap_i;

    wire ap_active = (hstate == H_APPEND);
    wire ap_decode = (hstate == H_DECAP);
    // Append SOURCE + TRIGGER.  SELF_KV=0: the host prefill/decap appends the stub
    //   kv_row_in (byte-identical).  SELF_KV=1: the pager appends the die's committed
    //   latent (kv_lat_row) on its commit pulse (kv_lat_valid) -- ONE append per
    //   decoded token; the host prefill/decap append is SUPPRESSED (H_APPEND is
    //   skipped entirely and ap_decode no longer drives a write), because the
    //   persistent pager already holds the prior tokens' rows.  The commit pulse
    //   fires AFTER all of this token's gathers (attention done), so a token never
    //   gathers its own not-yet-written row (causal: attend 0..s_len-1, append s_len).
    wire pg_append_valid = (SELF_KV != 0) ? mdl_kv_lat_valid : (ap_active || ap_decode);
    wire [ROW_BITS-1:0] pg_append_row = (SELF_KV != 0) ? mdl_kv_lat_row : kv_row_in;
    assign kv_row_sel = ap_decode ? {{(KVPOSW-(IDXW+1)){1'b0}}, s_len}
                                   : {{(KVPOSW-(IDXW+1)){1'b0}}, ap_i};

    always @(posedge clk) begin
        if (rst) begin
            hstate    <= H_IDLE;
            busy      <= 1'b0;
            done      <= 1'b0;
            next_tok  <= {TOKW{1'b0}};
            tok_valid <= 1'b0;
            ap_i      <= {(IDXW+1){1'b0}};
            mdl_start <= 1'b0;
        end else begin
            done      <= 1'b0;
            tok_valid <= 1'b0;
            mdl_start <= 1'b0;
            case (hstate)
                H_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        ap_i <= {(IDXW+1){1'b0}};
                        // SELF_KV=1: NO host prefill -- the persistent pager already
                        //   holds positions 0..s_len-1 (appended by prior decoded
                        //   tokens), so go straight to the run.  SELF_KV=0: unchanged
                        //   (prefill s_len stub rows unless s_len==0).
                        if ((SELF_KV != 0) || (s_len == {(IDXW+1){1'b0}})) begin
                            mdl_start <= 1'b1;
                            hstate    <= H_RUN_W;
                        end else
                            hstate <= H_APPEND;
                    end
                end
                H_APPEND: begin
                    if (ap_i == (s_len - 1'b1)) begin
                        mdl_start <= 1'b1;
                        hstate    <= H_RUN_W;
                    end
                    ap_i <= ap_i + 1'b1;
                end
                H_RUN_W: begin
                    if (mdl_done) begin
                        next_tok <= mdl_argmax;
                        hstate   <= H_DECAP;
                    end
                end
                H_DECAP: begin
                    hstate <= H_DONE;
                end
                H_DONE: begin
                    done      <= 1'b1;
                    tok_valid <= 1'b1;
                    busy      <= 1'b0;
                    hstate    <= H_IDLE;
                end
                default: hstate <= H_IDLE;
            endcase
        end
    end

    //========================================================================
    // 3) ROUTED-EXPERT EPISODE DETECT -> FIFO -> expert_cache_pf.
    //========================================================================
    wire moe_layer = (db_layer >= N_DENSE[LAYW-1:0]);
    wire cur_routed = fw_req && !fw_shared && moe_layer;

    reg              ep_active;
    reg [EIDXW-1:0]  ep_eidx;
    reg [LAYW-1:0]   ep_layer;
    wire new_episode = cur_routed &&
                       (!ep_active || (fw_eidx != ep_eidx) || (db_layer != ep_layer));

    always @(posedge clk) begin
        if (rst) begin
            ep_active <= 1'b0;
            ep_eidx   <= {EIDXW{1'b0}};
            ep_layer  <= {LAYW{1'b0}};
        end else begin
            ep_active <= cur_routed;
            if (cur_routed) begin
                ep_eidx  <= fw_eidx;
                ep_layer <= db_layer;
            end
        end
    end

    // ---- expert-id FIFO ----
    reg [EIDXW-1:0]  efifo [0:EFIFO_DEPTH-1];
    reg [EFW:0]      ef_wr, ef_rd;
    wire [EFW:0]     ef_cnt   = ef_wr - ef_rd;
    wire             ef_empty = (ef_wr == ef_rd);
    wire             ef_full  = (ef_cnt == EFIFO_DEPTH[EFW:0]);
    reg  [31:0]      dropped_r;
    assign ec_dropped = dropped_r;

    reg              awaiting;
    wire             ec_req_valid = (!ef_empty) && (!awaiting);
    wire [EIDXW-1:0] ec_req_id    = efifo[ef_rd[EFW-1:0]];

    integer fi;
    always @(posedge clk) begin
        if (rst) begin
            ef_wr     <= {(EFW+1){1'b0}};
            ef_rd     <= {(EFW+1){1'b0}};
            awaiting  <= 1'b0;
            dropped_r <= 32'd0;
            for (fi = 0; fi < EFIFO_DEPTH; fi = fi + 1)
                efifo[fi] <= {EIDXW{1'b0}};
        end else begin
            if (new_episode) begin
                if (!ef_full) begin
                    efifo[ef_wr[EFW-1:0]] <= fw_eidx;
                    ef_wr <= ef_wr + 1'b1;
                end else begin
                    dropped_r <= dropped_r + 32'd1;
                end
            end
            if (ec_req_valid) awaiting <= 1'b1;
            if (awaiting && ec_resp_valid) begin
                awaiting <= 1'b0;
                ef_rd    <= ef_rd + 1'b1;
            end
        end
    end

    //========================================================================
    // 4) EXPERT CACHE -- expert_cache_pf (GDDR6 cache + Flash prefetch).
    //========================================================================
    wire                 ec_flash_req;
    wire [EIDXW-1:0]     ec_flash_expert_id;
    wire                 ec_flash_done;
    wire                 ec_pf_ready;
    /* verilator lint_off UNUSEDSIGNAL */
    wire                 _ec_pf_ready_unused = ec_pf_ready;
    /* verilator lint_on UNUSEDSIGNAL */

    expert_cache_pf #(
        .SLOTS(CACHE_SLOTS), .N_EXPERT(N_EXPERT), .FLASH_LAT(FLASH_LAT),
        .CACHE_HIT_LAT(0)
    ) u_ecache (
        .clk(clk), .rst(rst),
        .req_valid(ec_req_valid), .req_expert_id(ec_req_id),
        .resp_valid(ec_resp_valid), .hit(ec_hit), .resp_slot(ec_resp_slot),
        .busy(ec_busy),
        .pf_valid(pf_valid), .pf_expert_id(pf_expert_id), .pf_ready(ec_pf_ready),
        .flash_req(ec_flash_req), .flash_expert_id(ec_flash_expert_id),
        .flash_done(ec_flash_done),
        .hit_count(ec_hit_count), .miss_count(ec_miss_count),
        .demand_stall_cycles(ec_demand_stall_cycles),
        .pf_issued(ec_pf_issued), .pf_hit(ec_pf_hit)
    );

    //========================================================================
    // 5) KV PAGER -- kv_cache_pager (latent ring + Flash overflow).
    //========================================================================
    wire                 pg_flash_req;
    wire [KVPOSW-1:0]    pg_flash_idx;
    wire                 pg_flash_done;
    wire [KVPOSW-1:0]    pg_gather_idx = {{(KVPOSW-IDXW){1'b0}}, kc_idx};

    // ---- KV_WRITEBACK_DESIGN.md step 3: per-(layer,position) KV keying. ---------
    //   KV is one latent row per (layer, position).  The model iterates layers
    //   0..L-1 within each token; layer m runs its OWN gather (kc_req/kc_idx) AND
    //   its OWN append (kv_lat_valid), both during that layer's attention phase --
    //   throughout which db_layer==m is STABLE (db_layer is registered in
    //   glm_model_q4k and only advances when the model observes db_done, which is
    //   the WHOLE decoder block; the attention gather + the S_DONE commit both fire
    //   inside that block, strictly before db_done).  So db_layer is the correct,
    //   stable layer annotation at BOTH the gather and the append edge.
    //
    //   We realise the flat index  pg_idx = db_layer*KV_CTX + pos  as the pager's
    //   per-WINDOW address  seq*RESIDENT + (pos mod RESIDENT)  with seq = db_layer:
    //   KV_NSEQ=L INDEPENDENT ring windows, each with its OWN append counter and
    //   residency.  Consequences:
    //     * layer m's position p lands in window m (a DISTINCT row) and layer m ONLY
    //       ever gathers window-m rows -> no cross-layer alias.
    //     * per token there are L appends (one per layer); window m's counter
    //       advances EXACTLY once per token, so at token t layer m holds positions
    //       0..t and the NEXT token's layer m gathers all of layer m's priors.
    //   L=1 collapses to KV_NSEQ=1 (single window, layer offset*1 = 0) -> the step-2
    //   loop, unchanged.  SELF_KV=0 -> KV_NSEQ=1 and the seq selects fold to a
    //   constant 0 -> the pager instance + its drivers are byte-identical to the
    //   pre-step-3 top (the layer-index logic lives entirely in the SELF_KV!=0 arm).
    wire [KV_SEQW-1:0] pg_append_seq = (SELF_KV != 0) ? db_layer[KV_SEQW-1:0]
                                                      : {KV_SEQW{1'b0}};
    wire [KV_SEQW-1:0] pg_gather_seq = (SELF_KV != 0) ? db_layer[KV_SEQW-1:0]
                                                      : {KV_SEQW{1'b0}};

    kv_cache_pager #(
        .ROW_BITS(ROW_BITS), .RESIDENT(KV_RESIDENT), .S_MAX(KV_CTX),
        .FLASH_LAT(FLASH_LAT), .NSEQ(KV_NSEQ)
    ) u_kvpager (
        .clk(clk), .rst(rst),
        .append_valid(pg_append_valid), .append_row(pg_append_row),
        .append_seq(pg_append_seq),
        .gather_valid(kc_req), .gather_idx(pg_gather_idx),
        .gather_seq(pg_gather_seq),
        .row_valid(kv_row_valid), .row_out(kv_row_out), .busy(kv_busy),
        .flash_req(pg_flash_req), .flash_idx(pg_flash_idx),
        // flash_seq (NSEQ>1: the cold row's LAYER) is left open here: this step's
        //   verified slice sizes KV_RESIDENT >= S_MAX so no (layer,pos) row ever
        //   spills to the COLD tier -- the only cold gathers are the empty-KV first
        //   token, masked to zero.  When KV_RESIDENT < S_MAX (real context), the
        //   external LPDDR backing store (stubbed here per KV_WRITEBACK_DESIGN.md
        //   step-3 note) MUST key (flash_seq=layer, flash_idx=pos); wire flash_seq
        //   through the arbiter to the backing store then.
        .flash_done(pg_flash_done), .flash_row(flash_row),
        .append_count(kv_append_count), .resident_lo(kv_resident_lo),
        .overflowed(kv_overflowed)
    );

    //========================================================================
    // 6) SINGLE FLASH ARBITER (demand-priority: expert-cache first).
    //========================================================================
    localparam G_EXP = 1'b0, G_PG = 1'b1;
    reg fl_busy;
    reg fl_gnt;

    // RESIDENT==0: the expert client reaches the Flash channel exactly as before
    // (both ternaries below fold to the original wiring).  RESIDENT==1: the
    // expert client is TIED OFF here -- arb_ec_req===1'b0 so ec_flash_req never
    // reaches the arbiter (only kv_cache_pager can own flash_req: the KV
    // NVMe-spill path stays available) and ec_flash_done is completed by the
    // §10 DDR-tier read (ec_ddr_done) instead of the Flash channel.
    wire arb_ec_req = (RESIDENT == 0) ? ec_flash_req : 1'b0;

    assign flash_req        = fl_busy;
    assign flash_is_expert  = (fl_gnt == G_EXP);
    assign flash_expert_id  = ec_flash_expert_id;
    assign flash_row_idx    = pg_flash_idx;
    assign ec_flash_done    = (RESIDENT == 0)
                            ? (flash_done && fl_busy && (fl_gnt == G_EXP))
                            : ec_ddr_done;
    assign pg_flash_done    = flash_done && fl_busy && (fl_gnt == G_PG);

    always @(posedge clk) begin
        if (rst) begin
            fl_busy <= 1'b0;
            fl_gnt  <= G_EXP;
        end else begin
            if (!fl_busy) begin
                if (arb_ec_req) begin
                    fl_busy <= 1'b1; fl_gnt <= G_EXP;
                end else if (pg_flash_req) begin
                    fl_busy <= 1'b1; fl_gnt <= G_PG;
                end
            end else if (flash_done) begin
                fl_busy <= 1'b0;
            end
        end
    end

    //========================================================================
    // 7) WEIGHT LOADER -- the matmul weight-pull DMA on the hot/representative
    //    tile.  Loaded once per compute launch (mdl_start); reads its tile from
    //    the latency-1 staging memory (TB) and drives the glm_matmul_q4k pull
    //    stream.  Its fetch addresses feed ddr5_xbar (§8).
    //========================================================================
    wire                       wl_mm_start;
    wire [WL_KW-1:0]           wl_k_len;
    wire [WL_D_W-1:0]          wl_w_d;      // Q4_K per-super-block fp16 d bus (obs)
    wire [WL_D_W-1:0]          wl_w_dmin;   // Q4_K per-super-block fp16 dmin bus (obs)
    wire [WL_SCL_W-1:0]        wl_w_scales; // Q4_K per-super-block 6-bit scales bus (obs)
    wire                       wl_done;

    // ---- weight_loader <-> backing-store nets (muxed by §DECOMP below) ----
    //   DECOMP==0 : these connect straight through to the wl_mem port (raw image).
    //   DECOMP>=1 : ldr_load is released after decompression fills the recon RAM,
    //               ldr_mem_data is the reconstruction-RAM read (latency-1), and
    //               the wl_mem port instead pulls the COMPRESSED byte stream.
    wire                       ldr_load;
    wire                       ldr_mem_en;
    wire [WL_ADDR_W-1:0]       ldr_mem_addr;
    wire [WL_DATA_W-1:0]       ldr_mem_data;

    weight_loader_q4k #(
        .PE_N(WL_PE_N), .KMAX(WL_KMAX), .ADDR_W(WL_ADDR_W), .DATA_W(WL_DATA_W)
    ) u_loader (
        .clk(clk), .rst(rst),
        .load(ldr_load),
        .desc_base({WL_ADDR_W{1'b0}}),
        .desc_klen(LOADER_KLEN[WL_KW-1:0]),
        .desc_nsblk({{(WL_SBW-1){1'b0}}, 1'b1}),   // one Q4_K [256] super-block
        .mem_en(ldr_mem_en), .mem_addr(ldr_mem_addr), .mem_data(ldr_mem_data),
        .mm_start(wl_mm_start), .mm_k_len(wl_k_len),
        .mm_w_q(loader_w_q),
        .mm_w_d(wl_w_d), .mm_w_dmin(wl_w_dmin), .mm_w_scales(wl_w_scales),
        .mm_in_valid(loader_in_valid),
        .busy(loader_busy), .done(wl_done)
    );

    //========================================================================
    // 7b) WEIGHT REFILL PATH -- raw (DECOMP==0) vs. on-chip decompressed (DECOMP>=1)
    //========================================================================
    generate
    if (DECOMP == 0) begin : g_wpath
        // -------- RAW: pass the loader's pulls straight to the wl_mem port ------
        assign ldr_load    = mdl_start;
        assign wl_mem_en   = ldr_mem_en;
        assign wl_mem_addr = ldr_mem_addr;
        assign ldr_mem_data = wl_mem_data;
    end else begin : g_wpath
        // -------- COMPRESSED backing image -> weight_decomp -> recon RAM --------
        localparam integer RAW   = (RECON_DEPTH <= 1) ? 1 : $clog2(RECON_DEPTH);
        localparam integer NBYTE = WL_DATA_W / 8;   // decoded bytes per staging word

        // weight_decomp handshake nets
        reg  [7:0]            wd_in_byte;
        reg                   wd_in_valid;
        wire                  wd_in_ready;
        wire [7:0]            wd_out_byte;
        wire                  wd_out_valid;
        wire                  wd_eob;

        // compressed-byte fetch from the wl_mem port (registered, latency-1)
        reg  [WL_ADDR_W-1:0]  cmp_addr;
        reg                   rd_pending;
        reg  [7:0]            hold_byte;
        reg                   hold_valid;
        reg                   wd_active;    // a decode block is in progress

        wire want_byte = wd_active & wd_in_ready & ~hold_valid & ~rd_pending & ~wd_eob;

        assign wl_mem_en   = want_byte;
        assign wl_mem_addr = cmp_addr;

        always @* begin
            wd_in_byte  = hold_byte;
            wd_in_valid = hold_valid;
        end

        always @(posedge clk) begin
            if (rst) begin
                cmp_addr   <= {WL_ADDR_W{1'b0}};
                rd_pending <= 1'b0;
                hold_byte  <= 8'h00;
                hold_valid <= 1'b0;
                wd_active  <= 1'b0;
            end else begin
                if (rd_pending) begin                // returned byte lands now
                    hold_byte  <= wl_mem_data[7:0];
                    hold_valid <= 1'b1;
                    rd_pending <= 1'b0;
                end
                if (want_byte) begin                 // issue the next fetch
                    rd_pending <= 1'b1;
                    cmp_addr   <= cmp_addr + 1'b1;
                end
                if (wd_in_valid & wd_in_ready)       // decomp consumed the held byte
                    hold_valid <= 1'b0;
                if (wd_eob) wd_active <= 1'b0;        // block finished
                if (mdl_start) begin                 // (re)start on each compute launch
                    cmp_addr   <= {WL_ADDR_W{1'b0}};
                    rd_pending <= 1'b0;
                    hold_valid <= 1'b0;
                    wd_active  <= 1'b1;
                end
            end
        end

        weight_decomp #(
            .MAXLEN(WD_MAXLEN), .SYMW(WD_SYMW), .COUNTW(WD_COUNTW),
            .AW(WD_AW), .BUFW(WD_BUFW), .EOB_SYM(WD_EOB_SYM)
        ) u_wdecomp (
            .clk(clk), .rst(rst),
            .tbl_we(decomp_tbl_we), .tbl_sel(decomp_tbl_sel),
            .tbl_addr(decomp_tbl_addr), .tbl_wdata(decomp_tbl_wdata),
            .start(mdl_start),
            .in_byte(wd_in_byte), .in_valid(wd_in_valid), .in_ready(wd_in_ready),
            .out_byte(wd_out_byte), .out_valid(wd_out_valid),
            .out_ready(1'b1), .eob(wd_eob)
        );

        // reassemble decoded Q4_K bytes (LE) into WL_DATA_W words; write recon RAM
        reg  [WL_DATA_W-1:0]  recon [0:RECON_DEPTH-1];
        reg  [RAW-1:0]        word_idx;
        reg  [1:0]            byte_lane;
        reg  [WL_DATA_W-1:0]  word_acc;
        reg                   recon_ready;
        reg                   recon_ld_pulse;
        reg  [WL_DATA_W-1:0]  recon_rd;

        always @(posedge clk) begin
            if (rst) begin
                word_idx       <= {RAW{1'b0}};
                byte_lane      <= 2'd0;
                word_acc       <= {WL_DATA_W{1'b0}};
                recon_ready    <= 1'b0;
                recon_ld_pulse <= 1'b0;
            end else begin
                recon_ld_pulse <= 1'b0;
                if (mdl_start) begin
                    word_idx    <= {RAW{1'b0}};
                    byte_lane   <= 2'd0;
                    word_acc    <= {WL_DATA_W{1'b0}};
                    recon_ready <= 1'b0;
                end else begin
                    if (wd_out_valid) begin
                        word_acc[8*byte_lane +: 8] <= wd_out_byte;
                        if (byte_lane == (NBYTE-1)) begin
                            recon[word_idx] <= { wd_out_byte, word_acc[8*(NBYTE-1)-1:0] };
                            word_idx  <= word_idx + 1'b1;
                            byte_lane <= 2'd0;
                        end else begin
                            byte_lane <= byte_lane + 2'd1;
                        end
                    end
                    // release the loader one cycle after EOB (final word committed)
                    if (wd_eob && !recon_ready) begin
                        recon_ready    <= 1'b1;
                        recon_ld_pulse <= 1'b1;
                    end
                end
            end
        end

        // latency-1 reconstruction-RAM read serving the loader's sequential pulls
        always @(posedge clk)
            recon_rd <= recon[ldr_mem_addr[RAW-1:0]];

        assign ldr_load     = recon_ld_pulse;
        assign ldr_mem_data = recon_rd;
    end
    endgenerate

    always @(posedge clk) begin
        if (rst) begin
            loader_done_count <= 32'd0;
            loader_beat_count <= 32'd0;
        end else begin
            if (wl_done)          loader_done_count <= loader_done_count + 32'd1;
            if (loader_in_valid)  loader_beat_count <= loader_beat_count + 32'd1;
        end
    end

    //========================================================================
    // 8) DDR5 FAST-TIER READ FABRIC -- ddr5_xbar.
    //    A tiny priority issuer presents one banked DDR5 read per cycle from
    //    three address sources: the weight_loader's tile fetches (LOAD), the
    //    expert cache's resident-slot reads (SLOT), and the compute die's hot-
    //    weight pulls (HOT).  bank_rot stripes accepted reads round-robin across
    //    channels so all DDR_NCH channels carry traffic (the N_CH BW path).
    //    Sources coalesce while pending (a bandwidth model -- redundant reads are
    //    dropped, never lost permanently: a continuing demand re-asserts).
    //========================================================================
    localparam [DDR_TAG_W-1:0] TAG_HOT  = 8'h01;
    localparam [DDR_TAG_W-1:0] TAG_SLOT = 8'h02;
    localparam [DDR_TAG_W-1:0] TAG_LOAD = 8'h03;
    localparam [DDR_TAG_W-1:0] TAG_LBAW = 8'h04;   // C8 loopback attention-weight read
    localparam [DDR_TAG_W-1:0] TAG_EFILL= 8'h05;   // RESIDENT=1 expert refill read

    wire hot_pull = em_req | gn_req | aw_req | rw_req | fw_req | fn_req | lw_req;

    reg                  p_hot, p_slot, p_load;
    reg [CSLOTW-1:0]     slot_q;
    reg [WL_ADDR_W-1:0]  load_addr_q;
    reg [CH_IDX_W-1:0]   bank_rot;

    // §9 loopback read has TOP priority (the die is frozen waiting on it).  When
    // LOOPBACK==0, lb_pending===0 so every term below reduces to the original.
    // The §10 RESIDENT expert-refill read is next (the expert cache -- and with
    // EXPERT_STALL==1 the die itself -- is stalled on it); when RESIDENT==0,
    // ef_pending===0 so every term folds back to the original as well.
    wire sel_lb     = lb_pending;
    wire sel_ef     = ef_pending & ~lb_pending;
    wire sel_load   = p_load & ~ef_pending & ~lb_pending;
    wire sel_slot   = p_slot & ~p_load & ~ef_pending & ~lb_pending;
    wire sel_hot    = p_hot  & ~p_load & ~p_slot & ~ef_pending & ~lb_pending;
    wire any_pending = lb_pending | ef_pending | p_load | p_slot | p_hot;
    /* verilator lint_off UNUSEDSIGNAL */
    wire _sel_hot_unused = sel_hot;
    /* verilator lint_on UNUSEDSIGNAL */

    // combinational requester address/tag (feed-forward from registered state)
    reg  [DDR_ADDR_W-1:0] xreq_addr;
    reg  [DDR_TAG_W-1:0]  xreq_tag;
    always @* begin
        if (sel_lb) begin
            xreq_tag  = TAG_LBAW;
            xreq_addr = lb_req_addr;
        end else if (sel_ef) begin
            xreq_tag  = TAG_EFILL;
            xreq_addr = { {(DDR_ADDR_W-EIDXW-CH_IDX_W){1'b0}}, ef_id, bank_rot };
        end else if (sel_load) begin
            xreq_tag  = TAG_LOAD;
            xreq_addr = { {(DDR_ADDR_W-WL_ADDR_W-CH_IDX_W){1'b0}}, load_addr_q, bank_rot };
        end else if (sel_slot) begin
            xreq_tag  = TAG_SLOT;
            xreq_addr = { {(DDR_ADDR_W-CSLOTW-CH_IDX_W){1'b0}}, slot_q, bank_rot };
        end else begin
            xreq_tag  = TAG_HOT;
            xreq_addr = { {(DDR_ADDR_W-CH_IDX_W){1'b0}}, bank_rot };
        end
    end

    wire xreq_valid = any_pending;
    wire xreq_ready;
    wire xreq_fire  = xreq_valid & xreq_ready;
    assign lb_accept = xreq_fire & sel_lb;   // §9 loopback read accepted by the xbar
    assign ef_accept = xreq_fire & sel_ef;   // §10 refill read accepted by the xbar

    // issuer state: clears come FIRST, sets AFTER -> a same-cycle new event keeps
    // the source pending (never lost), bank_rot still advances on every accept.
    always @(posedge clk) begin
        if (rst) begin
            p_hot       <= 1'b0;
            p_slot      <= 1'b0;
            p_load      <= 1'b0;
            slot_q      <= {CSLOTW{1'b0}};
            load_addr_q <= {WL_ADDR_W{1'b0}};
            bank_rot    <= {CH_IDX_W{1'b0}};
            xbar_req_count <= 32'd0;
        end else begin
            // ---- consume the granted source ----
            if (xreq_fire) begin
                bank_rot       <= bank_rot + 1'b1;
                xbar_req_count <= xbar_req_count + 32'd1;
                if (sel_lb)        begin /* loopback pending cleared in §9 FSM */ end
                else if (sel_ef)   begin /* refill pending cleared in §10 FSM */ end
                else if (sel_load) p_load <= 1'b0;
                else if (sel_slot) p_slot <= 1'b0;
                else               p_hot  <= 1'b0;
            end
            // ---- register new fast-tier read events (override a same-cycle clear) ----
            if (hot_pull)       p_hot  <= 1'b1;
            if (ec_resp_valid) begin p_slot <= 1'b1; slot_q <= ec_resp_slot; end
            if (wl_mem_en)     begin p_load <= 1'b1; load_addr_q <= wl_mem_addr; end
        end
    end

    wire [DDR_TAG_W-1:0] xbar_resp_tag;

    ddr5_xbar #(
        .N_CH(DDR_NCH), .ADDR_W(DDR_ADDR_W), .DATA_W(DDR_DATA_W),
        .TAG_W(DDR_TAG_W), .ROW_LAT(DDR_ROW_LAT), .RESP_QD(DDR_RESP_QD),
        .BANK_LSB(0)
    ) u_xbar (
        .clk(clk), .rst(rst),
        .req_valid(xreq_valid), .req_ready(xreq_ready),
        .req_addr(xreq_addr), .req_tag(xreq_tag),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_addr(mem_req_addr), .mem_req_tag(mem_req_tag),
        .mem_resp_valid(mem_resp_valid), .mem_resp_ready(mem_resp_ready),
        .mem_resp_data(mem_resp_data), .mem_resp_tag(mem_resp_tag),
        .resp_valid(xbar_resp_valid), .resp_ready(1'b1),
        .resp_data(xbar_resp_data), .resp_tag(xbar_resp_tag)
    );

    /* verilator lint_off UNUSEDSIGNAL */
    wire _wl_obs_unused = &{1'b0, wl_mm_start, wl_k_len, wl_w_d, wl_w_dmin, wl_w_scales};
    /* verilator lint_on UNUSEDSIGNAL */

    //========================================================================
    // 9) C8 LOOPBACK -- feed ddr5_xbar's returned bytes into the die's aw_q.
    //    (LOOPBACK==0 : this generate ties die_clk=clk and die_aw_q=stub, so the
    //     module is BYTE-IDENTICAL to the pre-loopback design.)
    //
    //    The die's attention-weight pull is COMBINATIONAL: it drives {db_layer,
    //    aw_sel,aw_grp,aw_k} and expects the PE_N Q4_K lanes the SAME cycle.  The
    //    xbar answers only ROW_LAT cycles later, so we STALL the die by gating its
    //    clock (die_clk = clk & enable, enable latched on the low phase => glitch-
    //    free).  Per aw beat:  freeze the die, issue a banked DDR5 read (TAG_LBAW)
    //    whose address encodes the exact {layer,sel,grp,k}, capture the tagged
    //    response, present its low PE_N*4 bits on die_aw_q, then release the die
    //    for exactly one edge (which retires the staged lanes and advances aw_k).
    //    Because the die is synchronous, freezing its clock is transparent: at every
    //    die edge ALL its inputs equal what they'd be in the LOOPBACK==0 run (the
    //    stub still serves every other family same-cycle, and the staged aw lanes
    //    == the stub aw lanes for the same key), so the committed token is identical.
    //========================================================================
    generate
    if (LOOPBACK == 0) begin : g_lb
        assign die_aw_q    = aw_q;                // stub Q4_K code lanes, straight through
        assign lb_pending  = 1'b0;
        assign lb_req_addr = {DDR_ADDR_W{1'b0}};
        if (EXPERT_STALL == 0) begin : g_free
            // die runs at full clk -- BYTE-IDENTICAL to the pre-stall module.
            assign die_clk = clk;
        end else begin : g_estall
            // FAITHFUL EXPERT-MISS STALL: freeze the die (glitch-free clock gate,
            // enable latched on the LOW phase of clk) for exactly the cycles the
            // expert cache holds ec_busy -- every cycle a DEMAND MISS is serviced
            // by Flash.  The cache/FIFO/Flash-arbiter keep running on the ungated
            // clk, so the fetch that drops ec_busy always completes; the die then
            // advances again.  Clock-gating a synchronous die is transparent, so
            // the committed token is identical to EXPERT_STALL==0 (see g_lb.LOOPBACK
            // proof), while start->tok_valid grows by ec_demand_stall_cycles.
            wire die_ce = ~ec_busy;
            reg  die_en_lat;
            initial die_en_lat = 1'b1;
            always @(negedge clk) die_en_lat <= die_ce;
            assign die_clk = clk & die_en_lat;
        end
        /* verilator lint_off UNUSEDSIGNAL */
        wire _lb_off_unused = &{1'b0, lb_accept, xbar_resp_tag, xbar_resp_data};
        /* verilator lint_on UNUSEDSIGNAL */
    end else begin : g_lb
        // ---- encode the die's current aw pull key -> a banked-read address ----
        reg [DDR_ADDR_W-1:0] cur_addr;
        always @* begin
            cur_addr               = {DDR_ADDR_W{1'b0}};
            cur_addr[3:0]          = aw_sel;
            cur_addr[4  +: A_KCW]  = aw_k;
            cur_addr[12 +: A_GRPW] = aw_grp;
            cur_addr[20 +: LAYW]   = db_layer;
            cur_addr[24 +: 8]      = 8'hA5;        // TAG_LBAW address marker
        end

        // ---- staging + fetch state ----
        reg  [PE_N*4-1:0]     lb_col_q;      // xbar-returned Q4_K 4-bit code lanes (staged)
        reg                   lb_col_valid;  // staged lanes valid for lb_key_q
        reg  [DDR_ADDR_W-1:0] lb_key_q;      // key the staged lanes belong to
        reg                   lb_busy;       // a loopback read is in flight
        reg                   lb_pend_r;     // request asserted into the §8 issuer
        reg  [DDR_ADDR_W-1:0] lb_addr_r;     // encoded request address

        // staged lanes are for the exact key the die wants right now?
        wire lb_have = lb_col_valid && (lb_key_q == cur_addr);
        // die may advance iff it is not mid-aw-pull, or the beat is ready
        wire die_ce  = ~aw_req | lb_have;
        // need a fresh fetch: mid-aw-pull, nothing staged for this key, none in flight
        wire lb_want = aw_req & ~lb_have & ~lb_busy;

        // glitch-free clock gate: latch the enable on the LOW phase of clk
        reg die_en_lat;
        initial die_en_lat = 1'b1;
        always @(negedge clk) die_en_lat <= die_ce;
        assign die_clk = clk & die_en_lat;

        assign die_aw_q    = lb_col_q;   // the die reads the xbar-returned Q4_K code lanes
        assign lb_pending  = lb_pend_r;
        assign lb_req_addr = lb_addr_r;

        always @(posedge clk) begin
            if (rst) begin
                lb_col_q     <= {PE_N*4{1'b0}};
                lb_col_valid <= 1'b0;
                lb_key_q     <= {DDR_ADDR_W{1'b0}};
                lb_busy      <= 1'b0;
                lb_pend_r    <= 1'b0;
                lb_addr_r    <= {DDR_ADDR_W{1'b0}};
            end else begin
                // (1) the die consumed the staged lanes this edge -> retire them
                if (die_ce & aw_req) lb_col_valid <= 1'b0;
                // (2) launch a new banked read for the current key
                if (lb_want & ~lb_pend_r) begin
                    lb_pend_r <= 1'b1;
                    lb_addr_r <= cur_addr;
                    lb_key_q  <= cur_addr;
                    lb_busy   <= 1'b1;
                end
                // (3) the issuer accepted our request -> drop the request line
                if (lb_accept) lb_pend_r <= 1'b0;
                // (4) the tagged response returned -> stage the returned lanes
                if (xbar_resp_valid & (xbar_resp_tag == TAG_LBAW)) begin
                    lb_col_q     <= xbar_resp_data[PE_N*4-1:0];   // low PE_N*4 = Q4_K codes
                    lb_col_valid <= 1'b1;
                    lb_busy      <= 1'b0;
                end
            end
        end
        /* verilator lint_off UNUSEDSIGNAL */
        wire _lb_on_unused = &{1'b0, xbar_resp_data[DDR_DATA_W-1:PE_N*4]};
        /* verilator lint_on UNUSEDSIGNAL */
    end
    endgenerate

    // drain counter (responses are always accepted, resp_ready=1)
    always @(posedge clk) begin
        if (rst) xbar_resp_count <= 32'd0;
        else if (xbar_resp_valid) xbar_resp_count <= xbar_resp_count + 32'd1;
    end

    //========================================================================
    // 10) RESIDENT=1 EXPERT REFILL -- complete expert_cache_pf's Flash-DMA
    //     handshake from the DDR-tier fabric (ddr5_xbar, the LPDDR5X stand-in)
    //     instead of the single Flash channel.
    //     (RESIDENT==0 : this generate ties ef_*/ec_ddr_done off, the §6/§8
    //      RESIDENT qualifiers fold to constants, and the module is
    //      BYTE-IDENTICAL to the pre-RESIDENT design.)
    //
    //     expert_cache_pf HOLDS flash_req high until flash_done and is strictly
    //     SERIAL (never a second refill before the first completes), so ONE
    //     in-flight tagged read suffices: edge-detect the held request, latch
    //     {pending, expert id}, present one banked DDR5 read (TAG_EFILL) to the
    //     §8 issuer, and pulse ec_flash_done ( = ec_ddr_done, §6) on that read's
    //     tagged response.  The refill wait the cache -- and, with
    //     EXPERT_STALL==1, the frozen die -- pays is therefore the REAL
    //     ddr5_xbar round-trip (issuer accept -> banked channel -> tagged
    //     response), never FLASH_LAT.  The cache/issuer/xbar all run on the
    //     ungated clk, so the response that clears the stall always completes
    //     (no deadlock -- same argument as the EXPERT_STALL Flash path).
    //========================================================================
    generate
    if (RESIDENT == 0) begin : g_res
        assign ef_pending  = 1'b0;
        assign ef_id       = {EIDXW{1'b0}};
        assign ec_ddr_done = 1'b0;
        /* verilator lint_off UNUSEDSIGNAL */
        wire _res_off_unused = &{1'b0, ef_accept};
        /* verilator lint_on UNUSEDSIGNAL */
    end else begin : g_res
        reg              ef_pend_r;   // refill read presented to the §8 issuer
        reg [EIDXW-1:0]  ef_id_r;     // expert id of the in-flight refill
        reg              ec_req_d;    // ec_flash_req delayed (rising-edge detect)

        assign ef_pending  = ef_pend_r;
        assign ef_id       = ef_id_r;
        assign ec_ddr_done = xbar_resp_valid & (xbar_resp_tag == TAG_EFILL);

        always @(posedge clk) begin
            if (rst) begin
                ef_pend_r <= 1'b0;
                ef_id_r   <= {EIDXW{1'b0}};
                ec_req_d  <= 1'b0;
            end else begin
                ec_req_d <= ec_flash_req;
                // a new refill request (held-high handshake) -> latch + present
                if (ec_flash_req & ~ec_req_d) begin
                    ef_pend_r <= 1'b1;
                    ef_id_r   <= ec_flash_expert_id;
                end
                // the issuer accepted our read -> drop the request line
                if (ef_accept) ef_pend_r <= 1'b0;
            end
        end

`ifndef SYNTHESIS
        // RESIDENT invariant (simulation check): the EXPERT class never owns the
        // Flash channel -- any flash_req that fires is the kv_cache_pager
        // NVMe-spill client (fl_gnt==G_PG).  With arb_ec_req tied 0 in §6 this
        // holds by construction; the check locks it in for every future TB run.
        always @(posedge clk) begin
            if (!rst && flash_req && flash_is_expert)
                $fatal(1, "glm_q4k_system(RESIDENT=1): expert-class flash_req fired");
        end
`endif
    end
    endgenerate

endmodule
/* verilator lint_on DECLFILENAME */
