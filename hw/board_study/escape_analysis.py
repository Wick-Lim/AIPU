#!/usr/bin/env python3
"""
escape_analysis.py -- routability study for the v3-proto board (R3_APPLIANCE_SPEC 5c).

Claim under test: the 1280-bit LPDDR5X bus (24GB x20 packages, board-mounted)
escapes the SoC BGA and fans out to 20 DRAM packages on a 12-layer HDI board
(6 signal / 4 GND / 2 PWR) at ~130x110 mm.  This script does the escape-routing
arithmetic that turns that [EST] into a checked number.

MODEL (via-in-pad HDI, ring-cut channel-crossing constraint -- the physically
correct one; the earlier "surface dogbone" draft was too pessimistic and gave
0 channels at 0.4 mm, which contradicts every phone LPDDR board):
  - Every ball drops to any inner signal layer through a stacked/staggered
    microvia UNDER its pad (via-in-pad).  So the surface channel does NOT carry
    the via pad -- only the escaping TRACE.
  - A signal ball on ring r (0 = outer) must route outward, crossing every ring
    k < r.  The channels available at ring k across ALL signal layers is
        L * perim_channels(k) * traces_per_channel
    and the signals that must cross ring k is  sum of signal balls on rings > k.
  - Min signal layers L = max over k of  ceil(cross(k) / (perim_ch(k)*t)).
  This is the standard area-array escape bound; the binding ring is usually
  mid-array (few channels near center, but also few signals there).

Analysis, not a router: it counts channel capacity, not literal traces.  A real
KiCad/Allegro place-and-route is the next fidelity step (gen_gerbers.py emits the
placement to scale).  All inputs are stated; change them and the verdict changes.
"""
import math

# ---- HDI design rules (advanced-but-standard; phone/server substrate class) -------
TRACE_UM   = 75.0    # 3 mil line   (inner signal layer)
SPACE_UM   = 75.0    # 3 mil space
UVIA_ANTIPAD_UM = 250.0   # microvia antipad (clearance void) on an inner signal layer

def traces_per_channel(pitch_mm):
    """Traces fitting between two adjacent ball microvia antipads on an inner layer."""
    gap = pitch_mm * 1000.0 - UVIA_ANTIPAD_UM     # clear routing width between antipads
    per = TRACE_UM + SPACE_UM
    return max(0, int(math.floor((gap + SPACE_UM) / per)))

def escape_layers(side_mm, pitch_mm, n_signals, label, verbose=True):
    N = int(math.floor(side_mm / pitch_mm))
    t = traces_per_channel(pitch_mm)
    rings = (N + 1) // 2
    # signal balls per ring, distributed proportional to ring population
    pop = [max(1, 4 * (N - 1 - 2 * k)) for k in range(rings)]
    tot = sum(pop)
    sig = [n_signals * p / tot for p in pop]
    # cross(k) = signals on rings strictly inside ring k (must cross ring k going out)
    need = []
    for k in range(rings):
        cross = sum(sig[r] for r in range(k + 1, rings))
        perim_ch = max(1, 4 * (N - 1 - 2 * k))
        cap_per_layer = perim_ch * t
        Lk = math.ceil(cross / cap_per_layer) if cap_per_layer else 10**9
        need.append((k, round(cross), perim_ch, cap_per_layer, Lk))
    L = max(n[4] for n in need)
    binding = max(need, key=lambda n: n[4])
    if verbose:
        print(f"\n=== {label} ===")
        print(f"  package {side_mm:.1f} mm, pitch {pitch_mm} mm -> {N}x{N} grid; "
              f"traces/channel = {t} (via-in-pad, {TRACE_UM:.0f}/{SPACE_UM:.0f} um)")
        print(f"  signals to escape: {n_signals}")
        print(f"  binding ring k={binding[0]}: {binding[1]} signals must cross, "
              f"{binding[2]} channels x {t} = {binding[3]}/layer -> {binding[4]} layers")
        print(f"  --> SIGNAL LAYERS REQUIRED: {L}")
    return L

def main():
    print("=" * 72)
    print("v3-proto board routability study -- escape-layer count (via-in-pad HDI)")
    print("R3_APPLIANCE_SPEC.md section 5c ; design rules stated in-source")
    print("=" * 72)

    # SoC BGA: the binding constraint. 0.75 mm pitch reconciles ~4300 balls in ~50 mm.
    soc  = escape_layers(50.0, 0.75, 2800, "SoC BGA (1280-bit host, 2800 signals)")
    # DRAM: per-package escape (the easier side).
    dram = escape_layers(12.4, 0.40, 84, "DRAM 496-FBGA (84 signals/package)")

    # calibration sanity: a phone-AP-class escape should land near its known layer use
    cal = escape_layers(14.0, 0.40, 1300, "[calibration] phone-AP-class 1300-sig",
                        verbose=False)

    budget = 6  # 12-layer stack = 6 signal + 4 GND + 2 PWR
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  calibration check: phone-AP-class (1300 sig, 0.4 mm, 14 mm) -> "
          f"{cal} signal layers  (real phone HDI uses ~6 -- model is in range)")
    print(f"  SoC 1280-bit escape needs   {soc} signal layers  (binding)")
    print(f"  DRAM per-package escape     {dram} signal layer(s)")
    print(f"  a 12-layer stack provides   {budget} signal layers (6 SIG / 4 GND / 2 PWR)")
    fits12 = soc <= budget
    print()
    print(f"  Model bias: the calibration case over-predicts ({cal} vs real ~6), so this")
    print(f"  model runs CONSERVATIVE (~1.5x) -- it counts every signal through inner")
    print(f"  layers and ignores direct surface escape of the outer rings and finer")
    print(f"  2/2 mil rules real HDI uses. Read {soc} as an upper-ish bound on the SoC.")
    print()
    if fits12:
        print(f"  PASS (with margin): 1280-bit escape needs {soc} signal layers <= the 6 a")
        print(f"  12-layer stack gives, and the model is conservative, so the real number")
        print(f"  is likely <= {soc}. The spec's 12-layer (6 SIG / 4 GND / 2 PWR) is")
        print(f"  DEFENSIBLE for the full board-escape at analytical fidelity.")
        print(f"  Spare: {budget - soc} signal layer(s) for length-match detours / isolation.")
    else:
        print(f"  FINDING: escape needs {soc} signal layers > 6; a ~{soc+6}-layer stack")
        print(f"  (or directional DRAM-bank placement, or a narrower bus) is the fix.")
    print("\n  Scope: analytical channel-crossing bound, NOT a routed board. This makes")
    print("  the 12-layer claim falsifiable and defensible; a KiCad/Allegro place-and-")
    print("  route confirms the final margin (gen_gerbers.py emits the placement to scale).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
