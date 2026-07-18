# 다음 단계 계획 — GLM-5.2 **Q4_K** 가속기 (로컬 디바이스 · GGUF-native)

> **마이그레이션 노트 (FP8 → GGML Q4_K):** 이 저장소는 FP8 트랙에서 **GGML Q4_K 트랙으로 이전**됐다.
> 활성 데이터패스는 이제 `glm_model_q4k` / `glm_matmul_q4k` / `mla_attn_q4k` / `moe_router_q4k` /
> `swiglu_expert_q4k` / `mtp_head_q4k`, 시스템 top은 `glm_q4k_soc` → `glm_q4k_system` →
> `glm_q4k_system_cdc`다. **이전 FP8 트랙은 브랜치 `fp8` + 태그 `fp8-verified-baseline`에 보존**되며
> 여기서는 *과거 베이스라인*으로만 참조한다(현재 main이 개발하는 코드가 아님). `docs/Q4K_RETARGET.md` 참조.
> **아래 file:line 참조는 FP8 감사 시점 기록**이라 Q4_K 이전 후 줄 번호가 달라졌을 수 있다 — 모듈명을
> 기준으로 읽고, 검증한 줄만 갱신했다(`make all`=Makefile:26, `synth-glm`=Makefile:357).

> **범위 결정:** 실제 753B 모델을 GPU로 돌려 대조하는 **실체크포인트 검증(P1.1)은 이 no-GPU 계획의
> 범위 밖**이지만 *제외되어 사라진 것이 아니라* 여전히 **열린(OPEN) #1 정합성 게이트**다(아래 "여전히
> 열린 갭" §참조, 아래 D1). 이 계획 내부의 검증 신뢰 기준은 기존 **모듈 단위
> 유닛테스트(각 TB의 독립 fp64/Q4_K golden, iverilog/CPU, GPU 0)**이며 아래 모든 작업은 **GPU 불필요**.

> **제품 정체성 (단일 렌즈):** 이 가속기는 **이더넷 뽑은 채로 도는 오프라인·에어갭 로컬 싱글유저 개인
> 박스**다 — 한 대·한 사용자가 전체 GLM-5.2 753B를 **인터넷·클라우드 없이** 완전히 로컬에서 돌린다.
> 가중치 저장소는 공개 **`unsloth/GLM-5.2-GGUF : UD-Q4_K_XL`(467 GB, 753 GB FP8 대비 ~38% 작음)**이고,
> ggml **Q4_K** 블록 포맷(4-bit 코드 + 6-bit 서브블록 스케일 + fp16 슈퍼블록 스케일/min,
> `get_scale_min_k4`)을 하드웨어로 구현해 GGUF Q4_K 가중치를 직접 소비한다. 데이터가 나갈 **경로 자체가
> 없어** 아무것도 새어나가지 않는다("이더넷 케이블 뽑고도 되나?" 테스트; VPC-내/제로리텐션/TEE '시큐어
> 클라우드'도 연결이 필요해 이 테스트를 통과 못 함). **해자는 조합**이다: 오프라인 + 풀 프런티어(753B) +
> 어플라이언스/좌석 가격(`docs/USBC_PRODUCT_PLAN.md`).
>
> **해자의 정직한 범위 (반드시 좁게 읽을 것 — 2026-07 갱신):** Q4_K **GEMM 코어**는 **독립 ggml-Q4_K
> 레퍼런스(`tools/q4k_ref.py`, 팀 자체 `dequantize_row_q4_K` Python 재구현)에 대해 bit-exact**하고,
> 그 재구현은 이제 **실제 공개 GGUF 바이트의 dequant 계층에 대해 비트 단위로 증명**됐다
> (376,586,096 가중치 — Q4_K/Q6_K/**Q8_0**, 실공개 GGUF 2개 — llama.cpp 자체 dequant와 전부 비트 동일 — `docs/GGUF_CROSSCHECK.md`, 커밋
> 05639bf). *동적* UD-Q4_K_XL 믹스의 **Q6_K/Q8_0/F16 텐서도 RTL 소비자가 랜딩**됐다(`make
> mixedtype`, 커밋 a730b37), 조립 forward의 **수치 golden**도 랜딩됐다(`make model-q4k` 1155/1155,
> 커밋 b058f6f). 정직하게 남은 것: **llama.cpp 런타임 전체**(어텐션/누산 순서)와의 수치 동일성은
> 계약 밖(out-of-contract), 467 GB GLM 파일
> 자체의 end-to-end 구동은 미실시(D1). tok/s·비용·LOI 수치는 여전히 **[EST]/[PENDING]**
> (FPGA fit만 MEASURED — `fpga/results/`).
>
> **제품 속도는 하드웨어 사다리로 단계화**된다(`docs/HARDWARE_LADDER.md`; 성능 = 메모리 대역폭 = 칩
> IO/PHY = 자금): **① 저가 FPGA(KU3P급)+DDR4 ~5–8 tok/s [EST](지금, 동작 증명) · ② 시드 후
> 커스텀보드(DDR5/HBM) ~15–40 [EST] · ③ 볼륨 ASIC ~40+ [EST]** (모두 B=1 싱글유저, 동일 bit-exact RTL).
> *(갱신 2026-07: rung-③ 1차 설계점은 이제 **512GB LPDDR5X 완전 상주** — 설계점 **≈80 tok/s
> [실측-입력 EST]** (U(K)·수락률 r 실측, GLM-5.2 MTP 공표 수준이면 ~95); `docs/R3_APPLIANCE_SPEC.md` 참조.)*
> 이전의 평평한 ~25–40은 ②(자금-조달-후) 수치다. **main은 정확히 rung-①(FPGA 동작증명) 하나만 개발한다** —
> rung-②③은 문서화된 로드맵이지 지금 main의 코드가 아니다. 예전 5-스테이지 스칼라 'TPU v2.0' 코어는
> 저장소에서 **제거**됐다(git 히스토리 참조). 아래 완료 목록의 **배치 멀티시퀀스 트랙**
> (`glm_q4k_soc_ms`, N_STEPS 연속배치 디코드, `glm_decoder_block_q4k`에 인라인으로 접힌 expert-union-skip MoE 배칭)은 *같은 실리콘*을 배치했을 때의
> **비대상(non-target) 데이터센터 배치 분석**이지 제품 타깃이 아니다 — 개인 박스는 B=1로 돈다.

> **동기화 노트 (2026-07, Q4_K 코드 대조 재감사):** 아래 계획의 **Track B 전 항목과 Track C 대부분이
> 이후 커밋에서 완료됨**(모두 Q4_K 모듈 기준). 완료 마일스톤(코드로 확인): A2 멀티시퀀스 배치 어텐션
> (`PER_ROW_SEQ`, 풀모델 B=2·B=4 per-row **자기일관(self-consistency)** — 아래 갭 주의) · 배치 멀티시퀀스
> SoC top `glm_q4k_soc_ms` (+ 실 per-layer KV 저장 `kv_mem`) · N_STEPS 연속배치 디코드 루프 ·
> `DSA_REAL_IDX=1` 멀티시퀀스 · `kv_cache_pager` NSEQ 독립 링 · expert-union-skip MoE 배칭(`glm_decoder_block_q4k` 인라인, PE_M>1 union scan; 독립 `batched_moe.v` 모듈은 main에서 제거) ·
> `spec_chain_top` 실 드래프트 체이닝 (`make spec-slow`, `test/spec_chain_top_tb.v`) · DVFS `clk_throttle` ·
> verilator 커버리지 (`make coverage`) · 풀config(753B) **elaborate 클린** (`test/full_config_elab_wrap.v` /
> `configs/full_glm52.vh` — **시뮬레이션 아님**) · 교차전문가 압축 상한 **~1.34× [EST]**(그 이상 주장 금지).
>
> **완료(코드 확인):** B1·B2·B3·B4·B6·B7·B8, C1·C2·C3·C4·C5·C9 (+ C8: LOOPBACK default-off & `make cdc`).
> **부분/진행:** B5(구조·elaboration 계약 확립 — 중간크기 **기능** sim 미확인) · C6(`kv_ecc_ring`
> lane-SECDED 유닛 + `kv_cache_pager_ecc_fv` formal 완료; DDR5/NVMe payload ECC·BMC 재파라미터 잔여) ·
> C7(**정정 2026-07**: ICG는 이미 탑에 realized — `die_clk = clk & die_en_lat`
> (`glm_q4k_system.v:1307-1311`)가 `icg_cell` 글리치-프리 패턴 인라인으로 다이 전체를 게이트; 미세 ICG는
> 합성 추론. `mbist_ctrl`은 검증된 단일포트 March **레퍼런스** — 실제 저장소 `ring`/`vstore_mem`이 2-port
> async라 손배선은 theatre. 계약 [`docs/P2_MEMORY_MAP.md`](docs/P2_MEMORY_MAP.md) §4에 문서화; 잔여는
> **2-port BIST collar**(메모리 컴파일러 생성) + top scan stitch) · C10(P2 클로저 잔여).

## ⚠️ 여전히 열린 갭 — **진짜 남은 것** (증거 원장 기준, 2026-07 재감사)

> **갱신 (2026-07):** 이 목록의 구 1·2·3·5번(B9 조립 수치 golden · B10 혼합타입 · D2 FPGA fit)은
> **이후 커밋에서 닫혔다** — 폐쇄 증거는 아래 각 트랙 표(B9·B10·D2 행) 참조. 정직하게 **아직 열린**
> 것은 다음이다.

1. **실체크포인트 end-to-end 검증(D1)이 여전히 OPEN #1 정합성 게이트다.** 조립-Q4_K 토큰 정합성을
   llama.cpp/실 GGUF **런타임**에 대조하는 도구는 저장소에 없다(GPU/대용량 호스트 의존). 전제였던
   B9(조립 수치 golden — `make model-q4k` 1155/1155)와 B10(혼합타입 — `make mixedtype`)은 닫혔고,
   dequant 계층은 실 GGUF 바이트로 봉인됐다(`docs/GGUF_CROSSCHECK.md` — Q4_K/Q6_K/Q8_0 전부,
   2026-07-11 Q8_0 확장 포함) — 남은 것은 **whole-runtime** 대조 자체다. llama.cpp 전체
   런타임(어텐션/누산 순서) 수치 동일성은 계약 밖으로 유지.
2. **P2 클로저 잔여.** (정정 2026-07) ICG는 이미 탑에 인라인(`die_clk`, `glm_q4k_system.v:1307-1311`);
   `mbist_ctrl`은 검증된 단일포트 March 레퍼런스이고 실제 저장소가 2-port async라 탑 손배선은 theatre —
   진짜 잔여는 **2-port BIST collar**(메모리 컴파일러 생성, 계약 `docs/P2_MEMORY_MAP.md` §4) + top scan
   stitch; DDR5/NVMe payload ECC + BMC 재파라미터; PHY-클로저 loopback(바이트를 실제 die로 되먹임). → **C6·C7·C10**.
3. **경제성/BOM/TCO·LOI는 미검증 계획 문서**다(`docs/BOM.md`, `docs/USBC_PRODUCT_PLAN.md`,
   `docs/ICP*.md`). **LOI는 존재하지 않는다** — ICP 킷의 "서명된 비구속 LOI 1건"은 목표이지 증거가
   아니다.
4. **GLM-5.2 플래그십 자체의 r/U 확인 실측.** U(K)·수락률 r은 GLM-4.5-Air로 실측 완료
   (`docs/H_MEASUREMENT.md` 2차·3차) — 플래그십 자체 확인만 남음. tok/s는 [실측-입력 EST] 유지.
5. **풀config 기능 sim 불가침(구조적).** 753B 실형상은 elaborate-clean까지만 — LM-head GEMV
   ~2.4e8 K-beat/token이라 기능 sim은 비현실적(변화 없음).

## FP8 감사 시점 발견 갭 (역사적 기록 — 대부분 Q4_K 커밋에서 폐쇄)

> 아래 번호 갭들은 FP8 감사 시점 기록이다. **대부분 이후 커밋에서 폐쇄됨**(B1·B2·B3·C1·C2·C3·C9 등).
> 모듈명은 Q4_K 등가물로 갱신했고 FP8-era 줄 번호는 신뢰하지 말 것.

1. **전체칩 synth 게이트가 없었다** → **C1로 폐쇄**: `make synth-glm`(Makefile:357)이 `glm_q4k_system_cdc`
   전체를 `hierarchy -top glm_q4k_system_cdc -check; proc; opt; check -assert; stat`로 게이트하고 `make all`
   (Makefile:26)에 편입. **이는 구조 elaboration + 어서션 사인오프이지 sim이 아니다.**
2. **sparse-DSA 마스킹 버그** → **B1+B6로 폐쇄**: `mla_attn_q4k`가 선택 슬롯이 아니라 실제 키 인덱스로
   마스크; sparse per-row **union** 데이터패스 완결(dense TB 전부 byte-identical, line-81 caveat 제거).
3. **P2 신뢰성 유닛 + weight_decomp 미인스턴스** → **부분 폐쇄**: `reset_sync` CDC top 배선(C3),
   `weight_decomp` NVMe→loader refill 경로 배선(C9, `glm_q4k_system` `DECOMP=1` 빌드옵션). **잔여(정정
   2026-07):** ICG는 이미 탑 인라인(`die_clk`); MBIST는 2-port BIST collar(메모리 컴파일러 생성)가
   실질 잔여 — 단일포트 `mbist_ctrl` 손배선은 theatre(C7, 계약 `docs/P2_MEMORY_MAP.md` §4).
4. **`weight_decomp`이 tok/s를 움직이는 유일한 die-side 레버인데 미배선** → **C9로 배선**(order-0, quant
   바이트를 불투명 심볼로 처리; 실 NVMe 바이트 1.34×→~1.42× 절감 **[EST]**, 토큰 출력 불변).
5. **`spec_chain_top` 미완성** → **B8로 완전 승격**(pull 포트 승격, seed 규약 헤더 문서화,
   `test/spec_chain_top_tb.v`, `make spec-slow` 편입; committed==greedy).
6. **CI 전무** → **폐쇄**: `.github/` 존재.
7. **P1.2 "파라미터만 올리면 됨" 과소평가** → **B4/B7로 스코핑**: `mla_attn_q4k` scratch를 SWIN 디커플;
   풀config는 **elaborate만** 가능(LM head ~2.4e8 K-beat/token — 기능 sim 비현실적).

---

## Track B — RTL 정확성 & 스케일 (no GPU)

> **줄 번호는 FP8-era 기록** — Q4_K 모듈명 기준으로 읽을 것.

| # | 작업 | 수락 기준 | 노력 |
|---|------|-----------|------|
| **B1** (완료) | `mla_attn_q4k` — 선택 슬롯이 아니라 실제 키 인덱스 `sel_list[sf_feed_i] < slen_r[r]`로 마스크 | dense에서 no-op 증명(`sel_list[s]=s`); `mla_attn_q4k` per-row TB byte-identical | S |
| **B2** (완료·FP8-era; Q4_K 트리에 TB 파일 부재 — 재확인 필요) | sparse per-row 오라클 TB(S_MAX=8, TOPK=4, row별 상이 `x`, PER_ROW_POS/SLEN) — 각 행 `===` `mla_attn_q4k` PE_M=1 모듈; fetch 수 == distinct keys. **주의: 해당 sparse-perrow TB는 현재 `test/`에 없음**(FP8 트랙에 있던 파일) → Q4_K용으로 재작성/재확인 필요 | S |
| **B3** (완료·B8 흡수) | `spec_chain_top` 값싼 수정 — accepted-prefix 커서 전진 포팅, DRAIN 상태 추가, seed 불일치 헤더 문서화 | multi-pass가 커밋 토큰 안에서 재시작 안 함; `done`이 drain beat와 레이스 안 함 | S |
| **B4** (완료·elaborate만) | P1.2 elaboration — `configs/full_glm52.vh`(MODEL_DIM 6144,L 78,N_EXPERT 256,TOPK 8,VOCAB 154880,KV_LORA 512 **[PENDING safetensors]**,NOPE192/ROPE64,TOPK_ATTN 2048,POSW20,THETA 8e6,BLK128); `test/full_config_elab_wrap.v`로 `yosys hierarchy -top glm_model_q4k; check` + `verilator -Wall` | 미해결 param/zero-width/포트 불일치 0; lint clean. **elaboration 스터디이지 sim 아님** | M |
| **B5** (부분: 구조계약 확립·기능 sim 미확인) | P1.2 중간크기 기능검증 — FFN/MoE/vocab param만 올리고(어텐션 slice) `glm_model_q4k` 1토큰 vs in-TB fp64 golden; 비-/128 out-dim 블록스케일 부기 검증 | 1토큰 FFN/vocab sim argmax 일치·X-clean; 블록스케일 TB가 `q4k_ref.py` 레이아웃과 일치; 풀config 기능 sim은 비시도로 명시 | L |
| **B6** (완료) | `mla_attn_q4k` sparse per-row **union** 데이터패스 — row별 DSA 선택, distinct 키당 1회 `kc`+`W_uk`/`W_uv` fetch, row별 score/softmax/context 재인덱싱 | B2가 모든 행 bit-exact(3-row distinct-extent 포함); fetch == distinct union keys; dense TB byte-identical; line-81 caveat 제거 | XL |
| **B7** (완료) | SWIN 디커플 — `mla_attn_q4k`의 `scores`/`probs`/`vstore`·`glm_softmax #(.LEN())`를 `SWIN=TOPK_ATTN`으로 재범위, `IDXW`/`kc_idx`는 full S_MAX(1M) 유지 | 기존 TB byte-identical @S_MAX=SWIN; S_MAX=64/SWIN=8 sparse TB가 golden 일치. 경고: SWIN=2048 `vstore`≈4.3 Gbit — scratch BRAM/pager 이동은 별도 단계 | XL |
| **B8** (완료) | `spec_chain_top` 완전 승격 — `mn_*/tn_*/vn_*` pull 포트 승격(verify는 `spec_batched_top` 재사용, MTP는 `mtp_head_q4k` pull set), `em_*` embed pull, seed 규약을 numpy/fp64 MTP-chain 레퍼런스로 확정, `test/spec_chain_top_tb.v`(committed==greedy, K∈{2,3}) | `make spec-slow`가 `spec_chain_top` green; committed stream == 독립 greedy 레퍼런스; seed 헤더 기록. **DUT-vs-DUT 자기일관(수치 golden 아님)** | L |
| **B9** (✅ **완료** — 커밋 b058f6f) | **조립-Q4_K 수치 golden** — `glm_model_q4k`(+`mla_attn_q4k` `1/sqrt(d_head)` 스케일) full forward를 **독립 조립 numpy 레퍼런스**(`tools/glm_model_q4k_ref.py`, 같은 `q4k_ref.py` dequant 임포트)에 대조 | **닫힘**: `make model-q4k` **1155/1155** bit-exact (logits+argmax+h_state 바이트 동일; `model-q4k-acthw` 1155/1155로 ACT_HW 결과 불변까지) — GAP #1 폐쇄 | XL |
| **B10** (✅ **완료** — 커밋 a730b37) | **혼합타입 Q6_K/Q8_0/F16 소비자** — `q4k_mixed.vh` dequant 프리미티브 + `glm_matmul_q4k` per-column `w_type` 라우팅 + `weight_loader_q4k` `desc_wtype`으로 *동적* UD-Q4_K_XL 믹스 소비 | **닫힘**: `make mixedtype` — `q6k_prim`+`q8_0_prim`+`glm_matmul_mixed` **32/32** + `weight_loader_q4k_mixed` **192/192** (4타입 전부 ggml-reimpl 골든에 bit-exact, 24-tile 혼합 시퀀스 포함) — GAP #3 폐쇄 | XL |

## Track C — 제품화 / DFT / formal (no GPU)

| # | 작업 | 수락 기준 | 노력 |
|---|------|-----------|------|
| **C1** (완료) | `make synth-glm` — `glm_q4k_system_cdc` set을 `hierarchy -top glm_q4k_system_cdc -check; proc; opt; check -assert; stat`; `make all`(Makefile:26) 편입 | **최초 전체칩 구조 게이트**; exit 0, `check -assert` clean, leaf 전부 resolved. **구조 elaboration이지 sim 아님** | S |
| **C2** (완료) | `docs/P2_MEMORY_MAP.md` — 모든 비-TB `reg [] arr[]`(kv_cache_pager ring, ddr5/flash_xbar 응답 FIFO, cdc_async_fifo mem, boot/weight 버퍼 vs `expert_cache_pf` directory)를 SECDED / parity-MBIST / off-die로 분류 | grep된 reg array 100% 커버 + 근거 | S |
| **C3** (완료) | `reset_sync`를 `glm_q4k_system_cdc` host_clk/core_clk 양 경계 배선(glm_q4k_system_cdc.v:337/342) | `glm_q4k_system_cdc` TB 통과 유지; 도메인별 STAGES-edge 동기 deassert directed case | S |
| **C4** (완료) | `ecc_mem_wrap` scrub-write-back + sticky `serr`/`derr` + ack | `ecc_mem_wrap_tb`: `bd_we` 주입 → read(serr=1, 정정) → 재read ⇒ serr=0(scrub) | M |
| **C5** (완료) | `ddr5_xbar` 응답-FIFO no-overflow/underflow를 **unbounded k-induction**으로 승격 — `cnt[]` connect-bind, `test/formal/ddr5_xbar_ind_fv.v` | `make formal-ind` ddr5 통과(base+step, 비-vacuity 재보증); `docs/FORMAL.md` BOUNDED→UNBOUNDED | M |
| **C6** (부분: `kv_ecc_ring`+`kv_cache_pager_ecc_fv` 완료·DDR5/NVMe payload ECC 잔여) | DDR5/NVMe payload(가중치 바이트) + `kv_cache_pager` ring에 ECC; 위젠 워드에 대해 committed BMC 증명 재파라미터/재검증 | fault-injection TB: single-bit 정정 / double-bit `derr`; 기존 유닛+formal green. ROW_BITS=768(/64 아님) lane 분할 주의 | L |
| **C7** (정정 2026-07: ICG 탑 인라인 완료·MBIST collar 잔여) | ICG는 이미 탑에 realized (`die_clk`, `glm_q4k_system.v:1307-1311`, 다이 전체 게이트); `mbist_ctrl`은 검증된 단일포트 March **레퍼런스**로 실제 저장소 `ring`/`vstore_mem`이 2-port async라 손배선 불가 → 잔여는 **2-port BIST collar**(메모리 컴파일러 생성) + top `scan_enable`. 계약 `docs/P2_MEMORY_MAP.md` §4 | 주입 stuck-at에 `bist_fail=1`; `bist_mode=0`서 bit-identical; gated-clock는 `die_clk` off서 bit-identical·runt 없음(`make cdc`) | L/XL |
| **C8** (완료: LOOPBACK default-off + `make cdc`) | CDC 사인오프 — async crossing에 SDC + `make cdc` 구조 체커; "returned bytes not fed into die" loopback 폐쇄(default-off, 검증된 combinational 경로 불변) | `make cdc` unguarded crossing 0; loopback 모드가 combinational-stub와 동일 next token; `synth-glm check -assert` clean | M/XL |
| **C9** (완료) | `weight_decomp`(order-0)를 `glm_q4k_system` NVMe→DDR5 refill 경로 배선(`DECOMP=1` 빌드옵션, quant 바이트를 불투명 심볼로) + raw-vs-decompressed byte-identical 증명 | **tok/s를 움직이는 유일한 die-side 레버**(실 NVMe 1.34×→~1.42× [EST]); 토큰 출력 불변, `make unittests` green | L |
| **C10** (부분: `synth-glm`은 `make all` 편입·MBIST system TB 잔여) | P2 클로저 — `make all`에 ECC/MBIST/gated-clock system TB; PRODUCT_ROADMAP P2 항목을 증명 TB에 링크; unit-proven vs system-proven 문서화 | `make all`이 P2 system TB green; 각 `ALL N TESTS PASSED` | S |

## Track D — GPU/상용툴 필요 (no-GPU 범위 밖 · OPEN 게이트)

| # | 작업 | 수락 기준 | 비고 |
|---|------|-----------|------|
| **D1** (🔴 OPEN·#1 정합성 게이트) | **실체크포인트 검증** — 실 753B GLM(llama.cpp/실 GGUF)의 next-token argmax를 우리 데이터패스에 대조. 전제 B9·B10은 **완료**; dequant 계층은 실 GGUF 바이트로 봉인(`docs/GGUF_CROSSCHECK.md` — Q4_K/Q6_K/Q8_0 총 376,586,096 가중치 비트 동일). 남은 것: whole-runtime 대조 자체 | 코퍼스에서 argmax 일치. **저장소에 whole-runtime 대조 도구 없음 — 신규 필요** | GPU/대용량 호스트 |
| **D2** (✅ **완료 — MEASURED**, 커밋 bc8176d→c1c622d→69a32f7) | **FPGA fit / Vivado 사인오프** — XCKU3P에서 실 Vivado synth + full PnR: **142.3K LUT (87.5%)**, ~100K FF, 421 DSP, hold met, routed Fmax **10.2→17.2→46.5 MHz** (bit-exact 재파이프라인 3라운드, 캠페인 4.6×로 종료 — 잔여 worst path는 라우팅 지배) | **닫힘**: `bash fpga/run_fit.sh` · 리포트 `fpga/results/` · `fpga/README.md` (비트스트림/보드 브링업은 rung-① 데모 잔여 — 보드+핀 XDC 필요) | Vivado/벤더IP |

## Quick wins — no GPU

- [x] **전체칩 게이트:** `make synth-glm`(Makefile:357, `make all` 편입) → *Makefile*, *src/glm_q4k_system_cdc.v* (**C1**)
- [x] **sparse 갭 고정:** 마스크 수정 + union 데이터패스 → *src/mla_attn_q4k.v* (sparse per-row 오라클 TB는 Q4_K 트리에 부재 — 재확인 필요) (**B1+B6**)
- [x] **spec_chain 완전 승격:** 커서 전진 + DRAIN + seed 헤더 + pull 포트 + TB → *src/spec_chain_top.v*, *test/spec_chain_top_tb.v* (`make spec-slow`) (**B3→B8**)
- [x] **reset 하드닝:** `reset_sync`를 CDC top 배선(glm_q4k_system_cdc.v:337/342) → *src/glm_q4k_system_cdc.v* (**C3**)
- [x] **ECC/MBIST 언블록:** reg-array 분류 → *docs/P2_MEMORY_MAP.md* (**C2**)
- [x] **unbounded ddr5 증명:** connect-bind lift(`make formal-ind`) → *test/formal/ddr5_xbar_ind_fv.v*, *docs/FORMAL.md* (**C5**)
- [x] **CI 부트스트랩:** `.github/workflows/` 존재
- [x] **조립-Q4_K 수치 golden(B9):** GAP #1 폐쇄 — `make model-q4k` 1155/1155 (`tools/glm_model_q4k_ref.py`, 커밋 b058f6f)
- [x] **혼합타입 소비자(B10):** GAP #3 폐쇄 — `make mixedtype` 32/32 + 192/192 (`q4k_mixed.vh` + `w_type`, 커밋 a730b37)
- [x] **FPGA fit(D2):** XCKU3P 실측 — 142.3K LUT/87.5%, routed Fmax 46.5 MHz (`fpga/results/`, 커밋 69a32f7)
- [x] **GGUF 교차검증:** dequant 계층을 실 GGUF 바이트로 봉인 — Q4_K/Q6_K/Q8_0 총 376,586,096 가중치 비트 동일, 실파일 2개 (`docs/GGUF_CROSSCHECK.md`)
- 문서 정합화: `make all` = **`unittests synth-glm formal`**(Makefile:26) — GLM 동작증명 게이트. `q_lora/kv_lora` = **2048/512**(q_lora safetensors-CONFIRMED, kv_lora **[PENDING]**).

## 재조준 타임라인 (no-GPU 트랙)

```
WEEK 0 — Enabler
  make synth-glm + CI + 문서 정합화                         [C1, quick wins]  ✅

WEEKS 1-2 — 값싼 자체완결 수정 (완료)
  Track B:  B1 마스크 + B2 오라클 ; B3 spec_chain 커서/drain
  Track C:  C2 메모리 맵 ; C3 reset_sync ; C4 ecc scrub ; C5 ddr5 formal lift

WEEKS 3-5 — 중량급 (완료)
  Track B:  B4 풀config elaborate → B5 중간크기(부분)
  Track C:  C9 weight_decomp 통합 ; C8 CDC 사인오프 + loopback

WEEKS 4-8 — XL 구조 (완료)
  Track B:  B8 spec_chain → B6 sparse union(B2 게이트) → B7 SWIN 디커플
  Track C:  C6 payload/KV ECC(부분) ; C7 ICG 탑 인라인 완료·MBIST collar+scan 잔여 ; C10 P2 클로저(부분)

DONE (이후 커밋에서 완료)
  Track B:  B9 조립-Q4_K 수치 golden(GAP#1, make model-q4k 1155/1155) ;
            B10 혼합타입 Q6_K/Q8_0/F16(GAP#3, make mixedtype)
  Track D:  D2 FPGA/Vivado fit — XCKU3P MEASURED (87.5% LUT, routed 46.5 MHz)
  검증:      GGUF 교차검증 — dequant 계층 실바이트 봉인 (docs/GGUF_CROSSCHECK.md)

NEXT (🔴 OPEN — 진짜 다음 단계)
  Track D:  D1 실체크포인트 whole-runtime 검증(GPU) — 전제 B9·B10은 닫힘
  Track C:  C6/C7/C10 P2 클로저 잔여 (ICG 탑 인라인 완료; MBIST 2-port collar = 메모리 컴파일러 생성)
```

**게이트 관계:** B9(조립 golden)가 D1(실체크포인트)의 전제 · B10(혼합타입)도 D1 전제 · B2가 B6 게이트 ·
B6는 B7보다 먼저 · C1+C2가 C6/C7 게이트.

## 리스크 & 미지수 (no-GPU 범위)

- **(해소됨 — B9 완료)** ~~조립-Q4_K 수치 golden 부재~~ → `make model-q4k` 1155/1155 (조립 forward가
  독립 numpy 골든과 bit-exact). 남은 오독 주의: 이 골든도 **우리 reimpl 기준**이며 llama.cpp
  whole-runtime 대조(D1)는 여전히 OPEN.
- **(해소됨 — B10 완료)** ~~Q4_K 전용 한계~~ → 혼합타입 Q6_K/Q8_0/F16 RTL 소비자 랜딩(`make
  mixedtype`). 실체크포인트 end-to-end 구동(D1)만 남음.
- **B6 sparse union 순서 민감성 (진짜 XL).** serial fp32 softmax/context 체인은 순서 의존적. 잘못된
  per-row gather 순서는 저비트 mismatch만 내서 노이즈로 오독하기 쉬움 — DSA emit 순서가 정확한 계약.
- **B7 SWIN 디커플이 메모리 재구조화를 과소평가.** `vstore`가 SWIN=2048서 ~4.3 Gbit — flop으로 비현실적.
  "elaborate clean" ≠ "스케일서 realizable". 풀config **기능** sim은 불가(LM head ~2.4e8 cyc/token).
- **C6 formal 결합.** ECC check bit로 워드 확장 시 committed BMC datapath가 바뀜 → 재파라미터/재검증.
- **B8 spec_chain seed 규약은 설계 결정.** single-MTP-layer 자기회귀 체이닝은 1-layer 체크포인트 밖
  외삽 — "정답"을 문서화된 수치 레퍼런스로 고정해야(spec==greedy 안전성은 무관).
- **툴링 상한(정직한 경계).** CI yosys/iverilog는 로컬 **0.66** 베이스라인과 일치해야(connect-bind formal
  트릭 의존). **NO formal proof는 Q4_K 수치 데이터패스를 건드리지 않는다** — memory-system/control-plane
  안전성만(BMC는 BOUNDED, k-induction 서브셋만 UNBOUNDED). 실 scan stitching·JTAG·ATPG·static CDC·풀config
  STA/power는 상용툴 필요 — **P2는 hooks+harness+문서화된 hand-off이지 측정된 coverage가 아니다.** D1(실
  체크포인트)·D2(FPGA fit)·P3(PHY/STA/power)·P4(PCB/driver/tokenizer/qual)는 이 OSS 플로 밖. **ASIC은
  out-of-scope가 아니라 하드웨어 사다리 ③단계(볼륨 엔드게임)**(`docs/HARDWARE_LADDER.md`).

## 브리핑 정정 (RTL/시스템 — Q4_K 코드로 검증됨)

1. **활성 데이터패스는 Q4_K다.** FP8 모듈(`glm_model_fp8` 등)은 삭제됨 — 브랜치 `fp8`+태그
   `fp8-verified-baseline`에 보존. 현재 top은 `glm_q4k_system_cdc` → `glm_q4k_system` → `glm_q4k_soc`.
2. **`make synth`(레거시 `-top TPU`)는 제거됨.** 현재 게이트는 `make synth-glm`(`glm_q4k_system_cdc`
   전체칩 elaborate + `check -assert`, Makefile:357)이고 `make all`(Makefile:26)에 편입.
3. **bit-exact 범위는 좁다.** `glm_matmul_q4k`(160)·`q4k_prim`(18)이 ggml-Q4_K 레퍼런스에
   **bit-exact**(+ 혼합타입 `make mixedtype`, 조립 forward `make model-q4k` 1155);
   `swiglu_expert_q4k`(240)은 **functional**; `moe_router_q4k`(40)은 **structural 불변식**. 골든은
   **`tools/q4k_ref.py`(팀 자체 ggml 재구현)** 기준 — 단, 그 재구현의 dequant 계층은 이제 **실
   GGUF 바이트로 비트 단위 증명**됨(`docs/GGUF_CROSSCHECK.md`; llama.cpp whole-runtime은 계약 밖).
4. **`h_mtp`는 Q4_K MTP 전용.** `src/mtp_head_q4k.v`만 포트 있음 — `src/mtp_head.v`(bf16)엔 없음. bf16 체인
   레퍼런스는 추가 필요.
5. **`spec_chain_top`은 B8로 완전 승격됨**(pull 포트 전부, seed 헤더, `test/spec_chain_top_tb.v`,
   `make spec-slow`). committed==greedy는 **DUT-vs-DUT 자기일관**(수치 golden 아님).
6. **`reset_sync`는 CDC top 배선됨(C3), `weight_decomp`는 refill 경로 배선됨(C9, `DECOMP=1`).** ICG는
   이미 탑 인라인(`die_clk`, `glm_q4k_system.v:1307-1311`); `mbist_ctrl`은 단일포트 March 레퍼런스로
   2-port 저장소엔 손배선 불가 → 잔여는 2-port BIST collar(메모리 컴파일러 생성, C7; 계약 `docs/P2_MEMORY_MAP.md` §4).
7. **`ecc_mem_wrap`은 scrub-write-back + sticky serr/derr + ack**(C4로 read-후-재read serr=0).
8. **풀config는 elaborate만.** `test/full_config_elab_wrap.v`(MODEL_DIM 6144/L 78/N_EXPERT 256/VOCAB
   154880/Q_LORA 2048/KV_LORA 512 [PENDING])는 **elaboration 스터디이지 시뮬레이션이 아니다**(no stimulus,
   no golden). *(마이너: 해당 파일과 `configs/full_glm52.vh`에 `mla_attn_fp8`/`glm_model_fp8` stale 주석 잔존
   — 코드는 `glm_model_q4k`를 인스턴스화.)*
9. **문서 tok/s 사다리는 전부 [EST]** — 저가 FPGA ~5–8 / 커스텀보드 ~15–40 / 볼륨 ASIC ~40+
   *(갱신 2026-07: rung-③ 1차는 완전 상주, 설계점 ≈80 [실측-입력 EST] — U(K)·수락률 r 모두
   GLM-계열 실측(`docs/H_MEASUREMENT.md` 잡 B), GLM-5.2 MTP가 공표 수준이면 ~95 —
   `docs/R3_APPLIANCE_SPEC.md`)* (`docs/HARDWARE_LADDER.md`). GLM-5.2 플래그십 자체 트레이스 +
   실측 대역폭 전까지 측정값 아님.
10. **"모든 dim은 param bump" 과소평가** — `mla_attn_q4k` scratch를 S_MAX(1M)로 사이징 → SWIN 디커플
    필요(B7). RTL default(`S_MAX=8` 등)는 slice 값.
