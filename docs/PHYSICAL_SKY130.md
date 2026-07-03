# Physical characterization — REAL sky130 standard cells

Moves the FP8 compute die's area/timing from **[EST]** (market/physics models) to **real
synthesized numbers on a real open-source PDK**. Flow: yosys 0.66 `synth` → `dfflibmap` →
`abc -liberty` mapping to the **SkyWater sky130 high-density** standard-cell library
(`sky130_fd_sc_hd`, typical corner `tt_025C_1v80`, via `volare` PDK
`c6d73a35…`). `stat -liberty` gives real cell area (µm²); `abc … stime` gives the mapped
register-to-register critical-path delay (→ fmax). This is **pre-P&R** (gate-level synth,
no routing parasitics), so post-route fmax/area would be somewhat worse — but these are
*real cells on a real PDK*, not estimates.

## Results (sky130_fd_sc_hd, tt corner)

| Module | Real area (µm²) | of which sequential | Critical path | fmax (pre-route) |
|---|---|---|---|---|
| `glm_matmul_fp8` (block-scaled FP8 GEMM, PE 4×4, K=256) | **262,689** | 39.1 % (102,719) | **35.4 ns** | ~28 MHz |
| `fp32_mac_pipe` (the pipelined fp32 MAC) | **48,121** | — | **7.6 ns** | **~131 MHz** |

## What this establishes

1. **The −87.6% accumulator claim now has a real-cell anchor.** `glm_matmul_fp8`'s
   BFP fixed-point accumulator maps to real sky130 cells; the 262,689 µm² figure (39%
   sequential) is a concrete area for the FP8 GEMM tile, replacing the cell-count `[EST]`
   of `docs/PPA_FP8.md` with a PDK-mapped number.

2. **The fmax-limiting path is REAL and confirms PPA_FP8's thesis.** The pipelined MAC
   closes at ~131 MHz (7.6 ns), but `glm_matmul_fp8`'s own register-to-register path is
   35.4 ns (~28 MHz) — i.e. the **block-dequant / accumulate-fold logic AROUND the
   pipelined MAC, not the MAC itself, is the fmax limiter**. This is exactly the
   "fmax-limiting paths" `docs/PPA_FP8.md` flagged, now measured on real cells: the
   actionable fix is to pipeline the dequant/fold stage (the accumulator drain), which
   would lift the GEMM toward the MAC's ~131 MHz.

## Honest scope (what would take it to full physical sign-off)

- **Pre-route only.** These are post-synthesis (gate-level) numbers. Real fmax/area needs
  **place-and-route** (OpenLane/OpenROAD: floorplan → placement → CTS → routing → parasitic
  extraction → post-route STA). The OpenLane Docker flow was attempted; standing it up in
  this environment hit image-pull friction, so the PDK-mapped **synthesis** numbers above
  are the delivered real characterization. Post-route is the remaining step.
- **No power yet.** Dynamic/leakage power needs the post-route netlist + activity (VCD) →
  a power analysis pass. The clock-gating (`clk_en_ctrl`/`icg_cell`) is in place to exploit
  the ~75%-idle die, but the J/token figure stays a model estimate until a routed power run.
- **Per-module, not full-chip.** The whole `glm_fp8_system_cdc` is elaboration- and
  structurally-clean (`make synth-glm`), but a full-chip sky130 map/P&R (with the memory
  macros as real SRAM) is a larger effort; the FP8 GEMM + MAC above are the compute core.

## Reproduce

```sh
python3 -m volare enable --pdk sky130 c6d73a35f524070e85faff4a6a9eef49553ebc2b
LIB=~/.volare/volare/sky130/versions/c6d73a35*/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
yosys -p "read_verilog -sv -I src src/glm_matmul_fp8.v src/glm_fp_pipe.v; \
          synth -top glm_matmul_fp8 -flatten; dfflibmap -liberty $LIB; \
          abc -liberty $LIB; stat -liberty $LIB"          # -> Chip area
# timing: append an abc -script ending in `topo; stime`  # -> Delay = <ps>
```
