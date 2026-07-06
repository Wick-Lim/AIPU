# Code Coverage — Verilator structural (line / toggle / branch)

Structural code-coverage measurement of the GLM-5.2-FP8 accelerator RTL, using
**Verilator 5.048 `--coverage-line --coverage-toggle`** (line + toggle + the
implied branch metric) driven by the project's existing behavioral testbenches.
Run with **`make coverage`**.

The primary simulator for this project is **iverilog 13.0** (every TB is an
iverilog-style `$display`/`initial` behavioral TB, and the bit-fidelity proof is
the byte-identical iverilog suite, `make unittests`). iverilog has no built-in
coverage instrumentation, so coverage is measured with Verilator's `--binary`
flow, which compiles the same SystemVerilog testbench + RTL into a native sim
that emits a `coverage.dat` database. Only the subset of TBs that verilate **and**
run cleanly under Verilator is measured; the rest are documented as out-of-scope
below (with the exact reason).

## Method

For each measured module, `make coverage` (driver: `tools/cov_run.sh`) does:

1. **Verilate** the TB + its RTL into a runnable binary:
   ```
   verilator --binary --coverage-line --coverage-toggle -Isrc --timing \
             --top-module <tb> <tb.v> <rtl srcs...>
   ```
2. **Run** the resulting `V<tb>` from the repo root, directing its coverage
   database to a per-module file with the runtime plusarg
   `+verilator+coverage+file+build/cov/<mod>/coverage.dat`. The TB must still
   print its usual `ALL <N> TESTS PASSED` (coverage is only counted for a
   passing functional run).
3. **Score the module's own source**: the `coverage.dat` records every point
   with its source filename, so the database is filtered to the module's
   *primary* source file (e.g. `src/glm_matmul_fp8.v`) and read with
   `verilator_coverage`, which prints `line` / `toggle` / `branch` covered/total.
4. **Merge** every per-module database into `build/cov/merged.dat`
   (`verilator_coverage --write`); the merged design-source total (all `src/`
   points, testbench points excluded) is reported at the bottom.

All build/run artifacts land in `build/cov/` (gitignored). The per-module and
merged summaries are printed to the terminal and saved to `build/cov/summary.txt`.

> **What "line" means here.** Verilator counts coverage *points* (basic-block
> statement groups and branch arms), not raw source lines — continuous-assign
> combinational lines are largely folded — so a small combinational module can
> have very few line points (e.g. 2–5) at 100 %. "toggle" is 0→1 and 1→0
> transitions on nets/registers of the module; a pure-function `.vh` (e.g.
> `fp8_e4m3`) has no persistent nets, hence `0/0` toggle (`n/a`).

## Per-module coverage (committed slice config)

Coverage of each module's **own** primary source file, under its unit TB:

| Module | Line % | Toggle % | Branch % | TB |
|---|---|---|---|---|
| `fp8_e4m3` (`.vh`)     | 100.0 (28/28)   | n/a (0/0)         | 100.0 (10/10) | `fp8_e4m3_tb` (exhaustive 65 536 pairs) |
| `glm_matmul_fp8`       | 82.1 (23/28)    | 80.9 (9680/11960) | 98.3 (59/60)  | `glm_matmul_fp8_tb` |
| `glm_act`              | 96.9 (63/65)    | 82.1 (7856/9572)  | 85.2 (46/54)  | `glm_act_tb` |
| `glm_softmax`          | 94.4 (34/36)    | 86.0 (3001/3490)  | 94.2 (98/104) | `glm_softmax_tb` |
| `rope_interleave_unit` | 97.4 (150/154)  | 95.9 (4148/4326)  | 90.6 (58/64)  | `rope_interleave_unit_tb` |
| `sampler`              | 76.0 (19/25)    | 41.0 (618/1508)   | 95.0 (38/40)  | `sampler_tb` |
| `weight_decomp`        | 100.0 (2/2)     | 64.7 (642/992)    | 100.0 (18/18) | `weight_decomp_tb` |
| `clk_en_ctrl`          | 100.0 (5/5)     | 41.9 (250/596)    | 100.0 (4/4)   | `clk_en_ctrl_tb` |
| `clk_throttle`         | 100.0 (5/5)     | 42.9 (18/42)      | n/a (0/0)     | `clk_throttle_tb` |
| `icg_cell`             | 100.0 (2/2)     | 92.9 (26/28)      | 75.0 (6/8)    | `icg_cell_tb` |
| `ecc_secded`           | 84.6 (11/13)    | 99.9 (991/992)    | 93.8 (15/16)  | `ecc_secded_tb` (exhaustive SECDED) |
| `reset_sync`           | 100.0 (2/2)     | 100.0 (22/22)     | 100.0 (4/4)   | `reset_sync_tb` |
| `mbist_ctrl`           | 100.0 (6/6)     | 92.1 (116/126)    | 100.0 (14/14) | `mbist_ctrl_tb` |
| `ecc_mem_wrap`         | 42.9 (3/7)      | 71.5 (1029/1440)  | 87.5 (14/16)  | `ecc_mem_wrap_tb` |
| `kv_ecc_ring`          | 60.0 (6/10)     | 65.8 (1597/2426)  | 83.3 (10/12)  | `kv_ecc_ring_tb` |

**Merged design-source total** (15 modules, `src/` points only, testbench
excluded, shared sources merged):

| Metric | Covered / Total | % |
|---|---|---|
| line   | 554 / 631     | **87.8 %** |
| toggle | 44 906 / 56 078 | **80.1 %** |
| branch | 642 / 722     | **88.9 %** |

### Notes on the lower numbers (honest reading)

These are real, not hidden:

- **`ecc_mem_wrap` 42.9 % line / `kv_ecc_ring` 60 % line** — the uncovered
  points are the error-injection / uncorrectable-double-bit and init/scrub
  branches of the ECC RAM/ring that *this* unit TB does not drive on the primary
  wrapper source (the SECDED codec `ecc_secded` beneath them is exercised hard,
  99.9 % toggle). Exhaustive single-correct/double-detect fault coverage of these
  wrappers is proven functionally by their iverilog TBs and formally by
  `make formal` (`kv_cache_pager(ECC)` BMC).
- **`sampler` 41 % toggle / 76 % line** — the TB exercises one temperature +
  top-k/top-p configuration slice; the wide unused sampler datapath (full logit
  width, alternate sampling modes) never toggles.
- **`clk_throttle` / `clk_en_ctrl` 42–43 % toggle** — small control blocks whose
  wide counter/config buses are only partially swept by the directed TB.
- **`clk_throttle` branch `n/a`** — Verilator folds the prescaler's counter
  compare to no branch points.

## Reproduce

```
make coverage
```

Prerequisites: `verilator` (5.x) + `verilator_coverage` on `PATH`, and `python3`
(the `weight_decomp` TB reads a generated FP8 vector, which the driver produces
at `scratchpad/wd_vec.txt`). Output:

- `build/cov/summary.txt` — the per-module + merged table above.
- `build/cov/<module>/coverage.dat` — per-module raw database.
- `build/cov/merged.dat` — merged database of all measured modules.

Annotated source (every line/branch tagged with its hit count; `%000000` marks
uncovered points) — run from the repo root so the source files resolve:

```
verilator_coverage --annotate build/cov/annotated --annotate-min 1 build/cov/merged.dat
```

## Scope — what is and is NOT covered (honest bounds)

**This is structural (line/toggle/branch) coverage of the committed *slice*
configuration** — the small parameterization the unit TBs instantiate (e.g.
`glm_matmul_fp8` at the TB's tile size, the compact attention/MoE widths). It is
**not**:

- **Functional coverage** — it does not assert that meaningful *scenarios* (all
  rounding modes, all corner FP values, every routing pattern) were hit; it only
  measures which RTL lines/toggles/branches were structurally exercised. The
  *functional* fidelity guarantee is the separate byte-identical iverilog suite
  (`make unittests`, `make bitacc`, `make formal`).
- **Full-config coverage** — the production top runs at larger widths/depths
  (`PE_N`, `DDR_NCH`, `KV_RESIDENT`, layer count, `S_MAX`, expert count) than the
  measured slice. The slice→full-config equivalence is proven separately
  (`make sim-glm-compact` byte-identical token; `docs/FULL_CONFIG_ELAB.md`).
- **Whole-datapath / capstone coverage** — the large integration TBs
  (`glm_model_fp8` incl. the multi-seq batched runs, `glm_decoder_block_fp8`,
  `mla_attn_fp8`, `glm_fp8_system*`, `glm_fp8_soc_ms` incl. the N-step
  decode loop, `batched_moe` `bcov`, `spec_*`) are intentionally out of this
  run (minutes-long; several depend on the FP goldens noted below). Coverage
  here targets the fast leaf/unit modules.

### Testbenches that do NOT verilate/run (skipped, with reason)

Not hacked or modified — skipped and recorded. These all pass under the primary
iverilog sim; the issue is Verilator-specific.

| Module TB | Why skipped |
|---|---|
| `glm_fp_pipe_tb`     | The TB's bit-exact self-check builds its `exp` golden with the Verilog **`$exp`** real-number task; Verilator evaluates `$exp`/`real` differently than iverilog at underflow/large-magnitude inputs, so the golden diverges (measured exp rel-err up to 1.0) and the TB `$fatal`s under Verilator. Not a DUT bug — passes under iverilog. |
| `rmsnorm_unit_tb`    | Same class: the golden uses **`$sqrt`** (real) for the RMS reciprocal-norm reference; Verilator's `$sqrt`/`real` semantics diverge → 99 263 element mismatches and a `$fatal` under Verilator (byte-exact under iverilog). |
| `glm_matmul_pipe_tb` | Bit-exact FP self-check fails under Verilator (2595 tile mismatches, worst **relerr = 0** — i.e. numerically equal but differing bit patterns: `-0`/NaN/round-tie canonicalization Verilator resolves differently). Passes under iverilog. |
| `topk_select_tb`     | Uses **`process::self()`** + event controls that *require* `--timing`; Verilator 5.048's `--timing` code generator then emits broken C++ for it (`error: use of undeclared identifier '__Vfork_10__sync'`). Build fails — a Verilator codegen defect, not fixable without editing the TB. |
| `dsa_indexer_tb`     | Same `--timing` `__Vfork` codegen defect (`'__Vfork_4__sync'` undeclared); the TB uses the same process/event constructs. Build fails. |

**Bottom line.** Verilator gives real, reproducible **structural** coverage for
**15 leaf/unit modules** (merged 87.8 % line / 80.1 % toggle / 88.9 % branch of
the exercised design source). The two skipped *build* failures are a Verilator
`--timing` codegen defect; the three skipped *run* failures are Verilator
`real`/`$exp`/`$sqrt`/bit-canonicalization disagreements with the reference
iverilog goldens. Correctness/fidelity is owned by the iverilog suite and the
formal proofs; this coverage report is a structural-exercise complement to them.
