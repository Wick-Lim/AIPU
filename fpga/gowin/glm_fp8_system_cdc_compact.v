`timescale 1ns/1ps
//============================================================================
// glm_fp8_system_cdc_compact.v  --  COMPACT-CONFIG passthrough wrapper
//----------------------------------------------------------------------------
// WHY THIS FILE EXISTS
//   The Gowin `gw_sh` Tcl flow (unlike yosys `-chparam`) has NO clean way to
//   override a top-level module's parameters from the build script.  So to
//   synthesize the *compact* FPGA-miniaturization config (docs/MINIATURIZATION.md
//   "L0 compact config"), we wrap the REAL, UNMODIFIED product top
//   `glm_fp8_system_cdc` in a thin module whose ONLY difference is that the five
//   RESULT-INVARIANT resource parameters take their compact values:
//
//       PE_N        : 4 -> 2   (matmul PE-array columns halved)
//       DDR_NCH     : 4 -> 2   (DDR5 crossbar channels halved)
//       KV_RESIDENT : 16 -> 8  (KV ring resident rows, >= S_MAX)
//       EFIFO_DEPTH : 16 -> 8  (expert-FIFO depth halved)
//       CACHE_SLOTS : 4 -> 2   (expert-cache slots halved)
//
//   These change CAPACITY / PARALLELISM / BANDWIDTH only -- NOT the math.  The
//   decoded token stream is BYTE-IDENTICAL to the default config (proven by
//   `make sim-glm-compact` in the repo Makefile: committed vs compact tokens
//   diff-clean).  The area reduction is therefore "for free."
//
//   This wrapper reproduces the top's FULL parameter + port declaration VERBATIM
//   (so every derived width -- A_GRPW, ROW_BITS, DDR_NCH*DDR_DATA_W, etc.--
//   recomputes exactly as in the original) and connects the instance with `.*`
//   (implicit name-matched connect).  It changes ONLY the module name and the
//   five compact defaults above; NOTHING in src/ is touched.
//
//   To synthesize the compact config, set the top module to
//   `glm_fp8_system_cdc_compact` (the build script does this when COMPACT=1) and
//   add this file to the file list ALONGSIDE the unchanged src/ files.
//
//   NOTE: keep this header in sync if src/glm_fp8_system_cdc.v's port list ever
//   changes.  The 5 compact values mirror the Makefile `synth-glm-compact` /
//   `sim-glm-compact` targets and docs/MINIATURIZATION.md.
//============================================================================
/* verilator lint_off DECLFILENAME */
module glm_fp8_system_cdc_compact #(
    // ---- compute-die (glm_model_fp8) slice config -- passed straight through --
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
    parameter integer THETA      = 8000000,
    parameter integer PE_N       = 2,       // COMPACT: was 4
    parameter integer POSW       = 20,
    parameter integer N_EXPERT   = 8,
    parameter integer TOPK       = 2,
    parameter integer INTER_MOE  = 64,
    parameter integer INTER_DENSE= 256,
    parameter [31:0]  RSCALE     = 32'h40200000,
    parameter integer TN         = 4,
    parameter integer BLK        = 128,
    parameter integer LM_TN      = 4,
    // ---- memory-system config ----
    parameter integer CACHE_SLOTS = 2,      // COMPACT: was 4
    parameter integer FLASH_LAT   = 8,
    parameter integer KV_CTX      = 1024,
    parameter integer KV_RESIDENT = 8,      // COMPACT: was 16 (must stay >= S_MAX)
    parameter integer EFIFO_DEPTH = 8,      // COMPACT: was 16
    // ---- DDR5 fast-tier fabric (ddr5_xbar) config ----
    parameter integer DDR_NCH     = 2,      // COMPACT: was 4
    parameter integer DDR_ADDR_W  = 32,
    parameter integer DDR_DATA_W  = 256,
    parameter integer DDR_TAG_W   = 8,
    parameter integer DDR_ROW_LAT = 10,
    parameter integer DDR_RESP_QD = 4,
    // ---- weight_loader (matmul weight-pull DMA) config ----
    parameter integer WL_KMAX     = 256,
    parameter integer WL_ADDR_W   = 24,
    parameter integer LOADER_KLEN = MODEL_DIM,
    // ---- CDC FIFO depths (this wrapper) ----
    parameter integer REQ_AW      = 2,      // request FIFO addr width (depth 2**AW)
    parameter integer TOK_AW      = 3,      // token   FIFO addr width (depth 2**AW)
    // ====================================================================
    // derived (do NOT override) -- mirror glm_fp8_system's port-width derivations
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
    parameter integer A_NB       = (A_KMAX    + BLK - 1) / BLK,
    parameter integer FF_NB_D    = (FF_KMAX_D + BLK - 1) / BLK,
    parameter integer R_NB       = (FF_KMAX_M + BLK - 1) / BLK,
    parameter integer LAYW       = (L     <= 1) ? 1 : $clog2(L),
    parameter integer TOKW       = (VOCAB <= 1) ? 1 : $clog2(VOCAB),
    parameter integer DIMW       = (MODEL_DIM <= 1) ? 1 : $clog2(MODEL_DIM),
    parameter integer NVTILE     = VOCAB / LM_TN,
    parameter integer VTW        = (NVTILE <= 1) ? 1 : $clog2(NVTILE),
    parameter integer ROW_BITS   = (KV_LORA + ROPE) * 16,
    parameter integer KVPOSW     = (KV_CTX <= 1) ? 1 : $clog2(KV_CTX),
    parameter integer CSLOTW     = (CACHE_SLOTS <= 1) ? 1 : $clog2(CACHE_SLOTS),
    parameter integer WL_PE_N    = PE_N,
    parameter integer WL_DATA_W  = (8*PE_N >= 16) ? 8*PE_N : 16,
    // ---- this wrapper's packed-request width ----
    parameter integer REQ_W      = TOKW + POSW + (IDXW+1)
)(
    //========================== TWO ASYNCHRONOUS CLOCK DOMAINS ===============
    input  wire                          host_clk,   // USB-C device domain
    input  wire                          host_rst,   // sync, active-high (host)
    input  wire                          core_clk,   // compute-die domain
    input  wire                          core_rst,   // sync, active-high (core)

    //========================== HOST interface (sampled on host_clk) ========
    input  wire                          start,
    input  wire [TOKW-1:0]               prompt_tok,
    input  wire [POSW-1:0]               start_pos,
    input  wire [IDXW:0]                 s_len,
    // NOTE: these four are `output reg` in the original module (driven
    // procedurally there); in THIS passthrough wrapper they are driven by the
    // u_dut instance output, so they MUST be `wire` (a submodule output cannot
    // drive a parent-level variable/reg).  This is the only intentional
    // deviation from the verbatim header copy.
    output wire                          busy,
    output wire                          done,
    output wire [TOKW-1:0]               next_tok,
    output wire                          tok_valid,

    //====== everything below is the core_clk domain, passed straight through ===
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
    output wire                          aw_req,
    output wire [3:0]                    aw_sel,
    output wire [A_GRPW-1:0]             aw_grp,
    output wire [A_KCW-1:0]              aw_k,
    input  wire [PE_N*8-1:0]             aw_col,
    input  wire [16*PE_N*A_NB-1:0]       aw_scale,
    output wire                          rw_req,
    output wire [R_KW-1:0]               rw_k,
    input  wire [8*N_EXPERT-1:0]         rw_col,
    input  wire [16*N_EXPERT*R_NB-1:0]   rw_scale,
    output wire                          fw_req,
    output wire [1:0]                    fw_sel,
    output wire [FF_GWD-1:0]             fw_grp,
    output wire [FF_KWD-1:0]             fw_k,
    output wire                          fw_shared,
    output wire [EIDXW-1:0]              fw_eidx,
    input  wire [8*TN-1:0]               fw_col,
    input  wire [8*TN-1:0]               fw_col_up,
    input  wire [16*TN*FF_NB_D-1:0]      fw_scale_g,
    input  wire [16*TN*FF_NB_D-1:0]      fw_scale_u,
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

    //========================== DDR5 fabric channels ========================
    output wire [DDR_NCH-1:0]            mem_req_valid,
    input  wire [DDR_NCH-1:0]            mem_req_ready,
    output wire [DDR_NCH*DDR_ADDR_W-1:0] mem_req_addr,
    output wire [DDR_NCH*DDR_TAG_W-1:0]  mem_req_tag,
    input  wire [DDR_NCH-1:0]            mem_resp_valid,
    output wire [DDR_NCH-1:0]            mem_resp_ready,
    input  wire [DDR_NCH*DDR_DATA_W-1:0] mem_resp_data,
    input  wire [DDR_NCH*DDR_TAG_W-1:0]  mem_resp_tag,

    //========================== weight_loader staging memory ================
    output wire                          wl_mem_en,
    output wire [WL_ADDR_W-1:0]          wl_mem_addr,
    input  wire [WL_DATA_W-1:0]          wl_mem_data,

    //========================== observability ===============================
    output wire [TOKW-1:0]               argmax_o,
    output wire [MODEL_DIM*16-1:0]       h_state,
    output wire                          mdl_busy,
    output wire                          ec_resp_valid,
    output wire                          ec_hit,
    output wire [CSLOTW-1:0]             ec_resp_slot,
    output wire                          ec_busy,
    output wire [31:0]                   ec_hit_count,
    output wire [31:0]                   ec_miss_count,
    output wire [31:0]                   ec_demand_stall_cycles,
    output wire [31:0]                   ec_pf_issued,
    output wire [31:0]                   ec_pf_hit,
    output wire                          kv_row_valid,
    output wire [ROW_BITS-1:0]           kv_row_out,
    output wire                          kv_busy,
    output wire [KVPOSW-1:0]             kv_append_count,
    output wire [KVPOSW-1:0]             kv_resident_lo,
    output wire                          kv_overflowed,
    output wire [31:0]                   ec_dropped,
    output wire [31:0]                   xbar_req_count,
    output wire [31:0]                   xbar_resp_count,
    output wire                          xbar_resp_valid,
    output wire [DDR_DATA_W-1:0]         xbar_resp_data,
    output wire                          loader_busy,
    output wire [31:0]                   loader_done_count,
    output wire [31:0]                   loader_beat_count,
    output wire [8*WL_PE_N-1:0]          loader_w_row,
    output wire                          loader_in_valid
);
    // ------------------------------------------------------------------
    // Instantiate the REAL, UNMODIFIED product top with the compact base
    // parameters (all forwarded from THIS module's params, so the derived
    // widths recompute identically on both sides).  `.*` connects every
    // same-named port -- guaranteed to match because the wrapper reproduces
    // the exact port declarations above.
    // ------------------------------------------------------------------
    glm_fp8_system_cdc #(
        .MODEL_DIM   (MODEL_DIM),
        .L           (L),
        .N_DENSE     (N_DENSE),
        .VOCAB       (VOCAB),
        .H_HEADS     (H_HEADS),
        .NOPE        (NOPE),
        .ROPE        (ROPE),
        .V_DIM       (V_DIM),
        .Q_LORA      (Q_LORA),
        .KV_LORA     (KV_LORA),
        .S_MAX       (S_MAX),
        .TOPK_ATTN   (TOPK_ATTN),
        .THETA       (THETA),
        .PE_N        (PE_N),         // compact = 2
        .POSW        (POSW),
        .N_EXPERT    (N_EXPERT),
        .TOPK        (TOPK),
        .INTER_MOE   (INTER_MOE),
        .INTER_DENSE (INTER_DENSE),
        .RSCALE      (RSCALE),
        .TN          (TN),
        .BLK         (BLK),
        .LM_TN       (LM_TN),
        .CACHE_SLOTS (CACHE_SLOTS), // compact = 2
        .FLASH_LAT   (FLASH_LAT),
        .KV_CTX      (KV_CTX),
        .KV_RESIDENT (KV_RESIDENT), // compact = 8
        .EFIFO_DEPTH (EFIFO_DEPTH), // compact = 8
        .DDR_NCH     (DDR_NCH),     // compact = 2
        .DDR_ADDR_W  (DDR_ADDR_W),
        .DDR_DATA_W  (DDR_DATA_W),
        .DDR_TAG_W   (DDR_TAG_W),
        .DDR_ROW_LAT (DDR_ROW_LAT),
        .DDR_RESP_QD (DDR_RESP_QD),
        .WL_KMAX     (WL_KMAX),
        .WL_ADDR_W   (WL_ADDR_W),
        .LOADER_KLEN (LOADER_KLEN),
        .REQ_AW      (REQ_AW),
        .TOK_AW      (TOK_AW)
    ) u_dut ( .* );

endmodule
