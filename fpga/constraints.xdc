# ============================================================================
# constraints.xdc -- timing + CDC constraints for glm_q4k_system_cdc on XCKU3P.
#
#   TIMING/CDC ONLY. Physical PIN LOCATIONS (set_property PACKAGE_PIN / IOSTANDARD)
#   are BOARD-SPECIFIC -- add them from your dev board's master XDC (they depend on
#   which XCKU3P board you have; e.g. a KCU/KU3P eval board's clock + GPIO pins).
#
#   Two asynchronous clock domains:
#     core_clk -- the compute die (glm_q4k_system, the whole datapath).
#     host_clk -- the USB-C/host device interface.
#   host<->core is a VERIFIED CDC (cdc_async_fifo gray-pointer + reset_sync), so the
#   two domains are declared asynchronous (the crossings are false-paths, handled by
#   the synchronizers). This mirrors constraints the design was written against.
# ============================================================================

# --- clocks (adjust the period to your board's clock source + your Fmax target) ---
#   core_clk period is the number to sweep for the Fmax study: tighten it and read
#   the worst negative slack (WNS) from report_timing_summary; achieved Fmax =
#   1 / (period - WNS).
create_clock -name core_clk -period 5.000  [get_ports core_clk]   ;# 200 MHz target
create_clock -name host_clk -period 10.000 [get_ports host_clk]   ;# 100 MHz (host/USB-C)

# --- the two domains are asynchronous (crossings go through the CDC FIFO/reset_sync) ---
set_clock_groups -asynchronous \
    -group [get_clocks core_clk] \
    -group [get_clocks host_clk]

# --- resets are synchronous, active-high, in their own domains (no cross-clock timing) ---
set_false_path -from [get_ports host_rst]
set_false_path -from [get_ports core_rst]
