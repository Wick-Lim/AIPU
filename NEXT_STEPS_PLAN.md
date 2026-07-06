# 다음 단계 계획 — GLM-5.2-FP8 가속기 (P1.1 실체크포인트 검증 제외)

> **범위 결정:** 실제 753B 모델을 GPU로 돌려 대조하는 P1.1 트랙은 **완전 제외**.
> 검증은 기존 **모듈 단위 유닛테스트(각 TB의 독립 fp64/fp8 golden, iverilog/CPU, GPU 0)** 방식을 신뢰 기준으로 삼는다.
> 아래 모든 작업은 **GPU 불필요**. file:line 참조는 이번 감사에서 실제 코드로 검증됨.

> **제품 정체성 (단일 렌즈):** 이 가속기는 **이더넷 뽑은 채로 도는 오프라인·에어갭 로컬 싱글유저 개인
> 박스**다 — 한 대·한 사용자가 전체 GLM-5.2-FP8 753B를 **인터넷·클라우드 없이** 완전히 로컬에서
> 돌린다(지금껏 클라우드로 막혀 있던 현장·업무에서 프런티어 모델을 처음으로 돌리고, 박스를 통째로
> 소유). 데이터가 나갈 **경로 자체가 없어** 아무것도 새어나가지 않는다 — 비-egress·프라이버시는
> 헤드라인이 아니라 **결과이자 증명**("이더넷 케이블 뽑고도 되나?" 테스트; VPC-내/제로리텐션/TEE
> '시큐어 클라우드'도 연결이 필요해 이 테스트를 통과 못 한다 → 시큐어 클라우드 논쟁 종결). **해자는
> 오프라인 하나가 아니라 조합**이다: 오프라인 + 풀 프런티어(753B) + 어플라이언스/좌석 가격(753 GB
> 가중치는 **한 번** 오프라인 프로비저닝 후 완전 오프라인; 모델 갱신은 물리 재프로비저닝. 토큰당 API
> 요금 0; 개인 어플라이언스; `docs/USBC_PRODUCT_PLAN.md`). 제품이 보는 **유일한** 속도 지표는 **싱글유저 대화형 처리량
> ~3–12 → ~25–40 tok/s [EST]**(B=1, 채팅 가능). 아래 완료 목록의 **배치 멀티시퀀스 트랙**
> (`glm_fp8_soc_ms`, N_STEPS 연속배치 디코드, `make bcov` B∈{1,2,3,5,8})은 *같은 실리콘*을
> 배치했을 때 무엇이 되는지에 대한 **비대상(non-target) 데이터센터 배치 분석**이지 제품 타깃이
> 아니다 — 개인 박스는 B=1로 돈다(집계 상한을 배치수로 나눈 per-user 수치를 제품 속도로 제시하지
> 않는다).

> **동기화 노트 (2026-07, 코드 대조 재감사):** 아래 계획의 **Track B 전 항목과 Track C
> 대부분이 이후 커밋에서 완료됨**. 완료된 마일스톤(코드로 확인): A2 멀티시퀀스 배치 어텐션
> (`PER_ROW_SEQ`, 풀모델 B=2·B=4 per-row bit-exact) · 배치 멀티시퀀스 SoC top
> `glm_fp8_soc_ms` (+ 실 per-layer KV 저장 `kv_mem`) · N_STEPS 연속배치 디코드 루프
> (`make` 라벨 `glm_fp8_soc_ms(decode-loop)`) · `DSA_REAL_IDX=1` 멀티시퀀스 · `kv_cache_pager`
> NSEQ 독립 링 · `make bcov` (B∈{1,2,3,5,8}) · `spec_chain_top` 실 드래프트 체이닝
> (`make spec-slow`, `test/spec_chain_top_tb.v` 있음) · DVFS `clk_throttle` · verilator 커버리지
> (`make coverage`) · 풀config(753B) elaborate 클린 (`docs/FULL_CONFIG_ELAB.md` / `configs/full_glm52.vh`) ·
> 교차전문가 압축 상한 **~1.34×** (그 이상 주장 금지).
>
> **완료:** B1·B2·B3·B4·B6·B7·B8, C1·C2·C3·C4·C5·C9 (+ C8: LOOPBACK default-off & `make cdc`).
> **부분/진행:** B5(구조·elaboration 계약은 `docs/P12_SCALEUP.md`로 확립 — 중간크기 **기능**
> sim은 미확인) · C6(`kv_ecc_ring` lane-SECDED 유닛+`kv_cache_pager_ecc_fv` formal 완료; DDR5/Flash
> payload ECC·BMC 재파라미터는 잔여) · C7(`clk_gate_cluster` ICG+clk_en 유닛검증 완료 — MBIST 래퍼 +
> system-top scan stitch 잔여, `mbist_ctrl`/`icg_cell`은 아직 `glm_fp8_system*`에 미인스턴스) ·
> C10(P2 클로저 잔여). **여전히 미완(설계상):** 리던던트 dense 드래프트(가중치 필요),
> P1.1 실체크포인트(GPU 필요), 풀config **기능** sim(비현실적 — elaborate만).

## 이번 감사에서 검증된 "지금 고쳐야 할 실제 갭" (리드)

> **주의(사후):** 아래 번호 갭들은 위 동기화 노트대로 **대부분 이후 커밋에서 폐쇄됨**
> (B1·B2·B3·C1·C2·C3·C9 등). 원문은 감사 시점 기록으로 보존한다.

1. **전체칩 synth 게이트가 없다.** `make synth`는 `hierarchy -top TPU`(레거시 스칼라 코어)만 검사한다(Makefile:505). GLM top `glm_fp8_system_cdc`는 **whole-chip 구조 게이트가 전무**. → **C1**
2. **sparse-DSA 마스킹 버그.** `mla_attn_fp8.v:1068-1073`이 실제 키 인덱스 `sel_list[sf_feed_i]`가 아니라 **선택 슬롯 `sf_feed_i`로 마스크**한다. dense fallback에선 `sel_list[s]=s`라 no-op(그래서 테스트 통과)이지만, sparse+per-row extent에선 틀림. line 81이 sparse PE_M>1을 out-of-scope로 선언 중. → **B1 + B2**
3. **P2 신뢰성 유닛 + weight_decomp가 어떤 product top에도 인스턴스화 안 됨.** `reset_sync`, `ecc_mem_wrap`, `mbist_ctrl`, `icg_cell`, `clk_en_ctrl`, `weight_decomp/2` 전부 유닛검증만 되고 배선된 곳 없음. → **C3, C9, C7**
4. **`weight_decomp`은 tok/s를 실제로 움직이는 유일한 die-side 레버**(Flash-BW 바운드, 1.34×→~1.42× 실제 Flash 바이트 절감)인데 datapath에 안 붙어 있음. → **C9**
5. **`spec_chain_top`은 "syntax-checked 스켈레톤"보다 더 미완성.** TB 없음, pull 포트 전부 hard-zero(spec_chain_top.v:217-345), `mtp_emb` placeholder-zero, FSM C_IDLE→C_DONE에 DRAIN 없음 + multi-pass 커서 깨짐. → **B3 + B8**
6. **CI 전무** — `.github/` 없음. → quick win
7. **P1.2 "파라미터만 올리면 됨"은 한 가지 구조 변경을 과소평가.** `mla_attn_fp8`이 `scores`/`probs`/`vstore`와 `glm_softmax` LEN을 `S_MAX`(=1M 캐시주소 범위)로 잡아서, 어텐션 스케일업은 SWIN-vs-S_MAX 디커플(B7)이 필요. 풀config 기능 sim은 비현실적(LM head ~238M cyc/token). → **B4/B5/B7 + 정직한 스코핑**

---

## Track B — RTL 정확성 & 스케일 (no GPU)

| # | 작업 | 수락 기준 | 노력 |
|---|------|-----------|------|
| **B1** (완료) | `mla_attn_fp8.v:1068-1073` — 선택 슬롯이 아니라 실제 키 인덱스 `sel_list[sf_feed_i] < slen_r[r]`로 마스크 | dense에서 no-op 증명(`sel_list[s]=s`); `mla_attn_fp8_pslen_tb` byte-identical 통과 | S |
| **B2** (완료) | `test/mla_attn_fp8_sparse_perrow_tb.v` 작성 (S_MAX=8, TOPK=4, row별 상이한 `x`, PER_ROW_POS/SLEN 변형) — 각 배치 행 `===` 그 행의 `(x_r,pos_r,s_len_r)`로 돌린 PE_M=1 모듈; `kc_req`/`W_uk`/`W_uv` fetch 수 == distinct keys | **현 RTL에서 rows>0 실패**(sparse 갭 고정) / all-equal-x·PE_M=1 fold는 통과. B6의 pass-gate 오라클이 됨 | S |
| **B3** (완료·B8로 흡수) | `spec_chain_top.v` 값싼 수정 — `spec_batched_top.v:294-324`에서 accepted-prefix 커서 전진 포팅, `spec_batched_top` B_DRAIN(422-425) 미러하는 DRAIN 상태 추가, seed 불일치(step-0 post-final-norm `h_state` vs step≥1 pre-norm `db_y`)를 헤더에 문서화 | multi-pass가 커밋된 토큰 안에서 재시작 안 함; `done`이 drain beat와 레이스 안 함 | S |
| **B4** (완료) | P1.2 elaboration — `configs/full_glm52.vh`(MODEL_DIM=6144,L=78,N_EXPERT=256,TOPK=8,VOCAB=154880,KV_LORA=512,NOPE=192/ROPE=64,TOPK_ATTN=2048,POSW=20,THETA=8e6,BLK=128); `yosys hierarchy -top glm_model_fp8; check`(elaborate만) + `verilator -Wall`; `glm_matmul_fp8` leaf-synth @KMAX=16384 | 미해결 param/zero-width/포트 불일치 0; lint clean (256-expert/VOCAB 버스가 OOM이면 서브모듈 개별 elaborate) | M |
| **B5** (부분: 구조계약 확립·기능 sim 미확인) | P1.2 중간크기 기능검증 — FFN/MoE/vocab param만 올리고(어텐션은 slice) `glm_model_fp8` 1토큰 vs in-TB fp64 golden; 비-/128 out-dim(`W_kr` out=64, NOPE=192)에서 `[128,128]` 블록스케일 부기 검증; 사이클수 노트로 **구조+중간크기 P1.2 계약** 확립 | 1토큰 FFN/vocab sim argmax 일치·X-clean; 블록스케일 TB가 `glm_fp8_contract` 레이아웃과 일치; 풀config 기능 sim은 비시도로 명시 | L |
| **B6** (완료) | `mla_attn_fp8` sparse per-row **union** 데이터패스 — row별 DSA 선택(각 행 `qrot[r]`/`slen_r`), distinct 키당 1회 `kc`+`W_uk`/`W_uv` fetch, row별 score/softmax/context를 각 행의 descending-score 순서로 재인덱싱; 먼저 param-gated serialize 스톱갭 옵션 | B2가 모든 행에서 bit-exact 통과(3-row distinct-extent 포함); fetch 수 == distinct union keys; dense TB 전부 byte-identical; line 81 caveat 제거 | XL |
| **B7** (완료) | SWIN 디커플 — `scores`/`probs`/`vstore`·`glm_softmax #(.LEN())`를 `SWIN=TOPK_ATTN=2048`로 재범위, `IDXW`/`kc_idx`는 full S_MAX(1M) 유지; default S_MAX=SWIN. **B6 이후 순서**(둘 다 `scores`/`vstore`/`sel_list` 재범위) | 기존 TB byte-identical @S_MAX=SWIN; S_MAX=64/SWIN=8 sparse TB가 fp64 golden 일치. 경고: SWIN=2048 `vstore`≈4.3 Gbit — scratch를 BRAM/pager로 옮기는 **1단계**일 뿐 | XL |
| **B8** (완료) | `spec_chain_top` 완전 승격 — `mn_*/tn_*/vn_*` pull 포트 승격(verify는 `spec_batched_top.v:165-207` 재사용, MTP는 `mtp_head_fp8` pull set 추가), `em_*` embed pull로 `mtp_emb=embed(prev_tok)`, seed 규약을 numpy/fp64 MTP-chain 레퍼런스로 확정, `test/spec_chain_top_tb.v`(committed==greedy, X-free, K∈{2,3}), K_eff 테스트, `make unittests` 편입 | `make unittests`가 `spec_chain_top` green; committed stream == 독립 greedy 레퍼런스(K∈{2,3}); seed 결정 헤더 기록 | L |

## Track C — 제품화 / DFT / formal (no GPU)

| # | 작업 | 수락 기준 | 노력 |
|---|------|-----------|------|
| **C1** (완료) | `make synth-glm` 추가 — `glm_fp8_system_cdc` set을 `hierarchy -top glm_fp8_system_cdc -check; proc; opt; check -assert; stat`; `make all`에 편입 | **최초 전체칩 구조 게이트**(현재 synth는 `-top TPU`만); exit 0, `check -assert` clean, `stat`에 leaf cell 전부 resolved | S |
| **C2** (완료) | `docs/P2_MEMORY_MAP.md` — 모든 비-TB `reg [] arr[]`(kv_cache_pager 768b ring, ddr5/flash_xbar 응답 FIFO, cdc_async_fifo mem, boot/weight 버퍼 vs `expert_cache_pf` directory)를 SECDED / parity-MBIST / off-die로 분류 | grep된 reg array 100% 커버 + 근거 | S |
| **C3** (완료) | `reset_sync`를 `glm_fp8_system_cdc`의 host_clk/core_clk 양 경계에 배선(현재 `host_rst`/`core_rst`는 pre-synchronized 가정; reset_sync는 검증됐지만 어디에도 미인스턴스) | `glm_fp8_system_cdc_tb` 통과 유지; 도메인별 STAGES-edge 동기 deassert directed case | S |
| **C4** (완료) | `ecc_mem_wrap` scrub-write-back + sticky `serr`/`derr` + ack(현재 read시 정정만 — 썩은 비트가 남아 double error로 누적 가능; P2.1은 retry/recovery 요구) | 새 `ecc_mem_wrap_tb`: `bd_we` 주입 → read(serr=1, 정정) → 재read ⇒ serr=0(scrub) | M |
| **C5** (완료) | `ddr5_xbar` 응답-FIFO no-overflow/underflow를 **unbounded k-induction**으로 승격 — `cnt[0:N_CH-1]`(ddr5_xbar.v:159) connect-bind, `test/formal/flash_xbar_ind_fv.v` 템플릿 미러 | `make formal-ind`에 통과하는 ddr5 run(base+step, 비-vacuity 재보증); `docs/FORMAL.md` 행 BOUNDED→UNBOUNDED | M |
| **C6** (부분: `kv_ecc_ring`+formal 완료·DDR5/Flash payload ECC 잔여) | DDR5/Flash payload 경로(weight 바이트 운반) + `kv_cache_pager` ring에 ECC; **위젠 워드에 대해 6개 committed BMC 증명 재파라미터/재검증** | fault-injection TB: single-bit 정정 / double-bit `derr`; 기존 유닛+formal 전부 green (ROW_BITS=768은 /64 아님 — lane 분할이 pager read latency 이동 가능) | L |
| **C7** (부분: `clk_gate_cluster` 완료·MBIST+scan 잔여) | MBIST 래퍼(SRAM별 functional/BIST mux + daisy-chain + `bist_mode/done/fail`, `mbist_ctrl`용 registered-read 어댑터) + `clk_en_ctrl`/`icg_cell`을 실제 compute cluster에 + top `scan_enable`→모든 `icg_cell.test_en` | 주입 stuck-at에 MBIST `bist_fail=1`(macro id 정확), `bist_mode=0`서 bit-identical; gated-clock TB가 free-running과 bit-identical·runt 없음; `scan_enable`시 전 도메인 `gated_clk==clk` | L/XL |
| **C8** (완료: LOOPBACK default-off + `make cdc`) | CDC 사인오프 — 모든 async crossing에 SDC `set_false_path`/`set_max_delay` + `make cdc` 구조 체커; **"returned bytes not fed into die" loopback 폐쇄**(glm_fp8_system.v:82-89) — `xbar_resp_data`를 die의 weight/KV 소비로 valid/stall 핸드셰이크 뒤 되먹임, default-off(검증된 combinational 경로 불변) | `make cdc` unguarded crossing 0; loopback 모드가 combinational-stub와 동일 next token, `synth-glm check -assert` clean | M/XL |
| **C9** (완료) | `weight_decomp`(order-0)를 `glm_fp8_system.v` Flash→DDR5 refill 경로에 배선(`weight_decomp2` order-1은 빌드 옵션) + raw-vs-decompressed FP8 코드 byte-identical 증명 system TB | **tok/s를 움직이는 유일한 die-side 레버**(실제 Flash 바이트 1.34×→~1.42× 절감); 토큰 출력 불변, `make unittests` green | L |
| **C10** (부분: `synth-glm`은 `make all` 편입·MBIST system TB 잔여) | P2 클로저 — `make all`에 `synth-glm` + ECC/MBIST/gated-clock system TB; 각 PRODUCT_ROADMAP P2 항목을 증명 TB에 링크; unit-proven vs system-proven 문서화 | `make all`이 P2 system TB green; 각 `ALL N TESTS PASSED` | S |

## Quick wins — 이번 주 시작 (no GPU)

- [x] **전체칩 게이트:** `make synth-glm` 추가(Makefile:652, `make all` 편입) → *Makefile*, *src/glm_fp8_system_cdc.v* (**C1**)
- [x] **sparse 갭 고정:** 마스크 수정 + 회귀 오라클(B6 union 데이터패스로 완결) → *src/mla_attn_fp8.v*, *test/mla_attn_fp8_sparse_perrow_tb.v* (**B1+B2**)
- [x] **spec_chain 값싼 수정 → 완전 승격:** 커서 전진 + DRAIN + seed 헤더 + pull 포트 승격 + TB → *src/spec_chain_top.v*, *test/spec_chain_top_tb.v* (`make spec-slow`) (**B3→B8**)
- [x] **reset 하드닝:** `reset_sync`를 CDC top에 배선(glm_fp8_system_cdc.v:305/310) → *src/glm_fp8_system_cdc.v* (**C3**)
- [x] **ECC/MBIST 작업 언블록:** 모든 reg-array 분류 → *docs/P2_MEMORY_MAP.md* (**C2**)
- [x] **unbounded ddr5 증명:** connect-bind lift(`make formal-ind`에 편입) → *test/formal/ddr5_xbar_ind_fv.v*, *docs/FORMAL.md* (**C5**)
- [x] **CI 부트스트랩:** `.github/workflows/ci.yml` 존재
- [x] **문서 정합화:** single-user tok/s 사다리 통일 — **~3 baseline → ~16–27 built today → ~25–40 [EST] ceiling**(전체 레버); README/SSP를 이 사다리에 정렬(README ~30+→~25–40, SSP ~3–6은 보수 subset으로 명시하고 full-stack ~25–40 지시). `q_lora/kv_lora` = 실 checkpoint **2048/512**(q_lora CONFIRMED, kv_lora standard-assumed). `make all` = `test hazard unittests lint synth synth-glm formal`(Makefile:58).

## 재조준 타임라인 (P1.1 제거)

```
WEEK 0 — Enabler (모든 no-GPU 검증의 게이트)
  make synth-glm + CI + 문서 정합화                         [C1, quick wins]

WEEKS 1-2 — 값싼 자체완결 수정 (병렬)
  Track B:  B1 mla 마스크 + B2 sparse 오라클 TB ; B3 spec_chain 커서/drain
  Track C:  C2 메모리 맵 ; C3 reset_sync ; C4 ecc scrub ; C5 ddr5 formal lift

WEEKS 3-5 — 중량급 (CI/오라클/synth-glm 존재 후)
  Track B:  B4 풀config elaborate → B5 중간크기 기능검증 + 사이클 계약
  Track C:  C9 weight_decomp 통합 (tok/s 레버) ; C8 CDC 사인오프 + loopback 폐쇄

WEEKS 4-8 — XL 구조 작업 (오라클 게이트)
  Track B:  B8 spec_chain 완전승격 → B6 sparse union 데이터패스(B2 게이트) → B7 SWIN 디커플(B6 후)
  Track C:  C6 payload/KV ECC + BMC 재검증 ; C7 MBIST+ICG+scan ; C10 P2 클로저
```

**게이트 관계:** Week-0 CI/`synth-glm`이 모든 B/C 검증 게이트 · **B2가 B6 게이트** · **B6는 B7보다 먼저**(shared `scores`/`vstore`/`sel_list`) · **C1+C2가 C6/C7 게이트**.

## 리스크 & 미지수 (no-GPU 범위)

- **B6 sparse union 순서 민감성 (진짜 XL).** serial fp32 softmax/context 체인은 순서 의존적. 잘못된 per-row gather 순서는 저비트 mismatch만 내서 fp8 노이즈로 오독하기 쉬움 — DSA emit 순서가 정확한 계약.
- **B7 SWIN 디커플이 메모리 재구조화를 과소평가.** `vstore`가 SWIN=2048서 ~4.3 Gbit — flop으로 비현실적. "elaborate clean" ≠ "스케일서 realizable". 풀config **기능** sim은 불가(LM head ~238M cyc/token) — P1.2를 "param 올리고 TB 돌리기"로 잡으면 조용히 안 끝남.
- **C6 formal 결합.** ECC check bit로 워드 확장 시 6개 committed BMC가 도는 datapath가 바뀜 → 재파라미터/재검증. ROW_BITS=768(/64 아님)이 pager read latency 이동 가능.
- **B8 spec_chain seed 규약은 배선이 아니라 설계 결정.** single-MTP-layer 자기회귀 체이닝은 1-layer 체크포인트 밖 외삽 — "정답"을 문서화된 수치 레퍼런스로 고정해야. K_eff에 영향(spec==greedy 안전성은 무관).
- **툴링 상한(정직한 경계).** CI yosys/iverilog는 로컬 **0.66** 베이스라인과 일치해야(connect-bind formal 트릭 의존). 실제 scan stitching·JTAG TAP·ATPG·static CDC 사인오프·풀config STA/power는 이 OSS 플로에 없는 상용툴 필요 — **P2는 hooks+harness+문서화된 hand-off를 제공하지, 측정된 coverage가 아님.** P3(PHY/FPGA-vs-ASIC/STA/power)·P4(PCB/driver/tokenizer/qual)는 설계상 out-of-scope.

## 브리핑 정정 (RTL/시스템 관련 — 코드로 검증됨)

3. **`make synth`는 product 계층을 게이트하지 않음.** `synth:`은 `hierarchy -top TPU`(레거시 스칼라). GLM top은 감사 시점엔 전체칩 구조 게이트 전무였으나 **C1로 폐쇄됨** — `make synth-glm`(Makefile:652, `glm_fp8_system_cdc` 전체칩 elaborate + `check -assert`)이 추가되고 `make all`에 편입됨.
6. **`h_mtp`는 FP8 전용.** `src/mtp_head_fp8.v`만 포트 있음 — `src/mtp_head.v`(bf16)엔 없음. bf16 체인 레퍼런스는 추가 필요.
8. **~~`spec_chain_top`은 스켈레톤보다 더 미완성~~ — B8로 폐쇄됨.** 감사 시점엔 TB 없음·pull 포트 hard-zero·`mtp_emb` zero·DRAIN 없음이었으나, 이후 pull 포트(m_/t_/v_ + em_) 전부 승격 + seed 규약 헤더 문서화 + `test/spec_chain_top_tb.v`(committed==greedy) 추가, `make spec-slow`에 편입.
9. **~~`reset_sync`·P2 프리미티브·`weight_decomp/2`가 어떤 product top에도 미인스턴스~~ — 부분 폐쇄.** `reset_sync`는 CDC top에 배선(C3), `weight_decomp`는 Flash→loader refill 경로에 배선(C9, `glm_fp8_system` DECOMP=1). **남은 미인스턴스:** `mbist_ctrl`/`icg_cell`(system top에는 아직 미배선 — `clk_gate_cluster`는 유닛 레벨만; C7 잔여).
10. **"모든 dim은 param bump"는 한 구조 변경 과소평가** — `mla_attn_fp8`이 scratch를 S_MAX(1M)로 사이징 → SWIN 디커플 필요(B7). RTL default(`Q_LORA=64`,`KV_LORA=32`,`POSW=20`,`S_MAX=8`)는 slice 값.
11. **~~`ecc_mem_wrap`은 read시 정정만~~ — C4로 폐쇄됨.** scrub-write-back + sticky `serr`/`derr` + `err_ack` + back-door raw-codeword 주입이 구현되어 P2.1 "retry/recovery" 요구를 충족(read 후 재read에서 `serr=0`).
12. **문서 불일치 — 해소됨:** single-user tok/s를 한 사다리로 통일(**~3 → ~16–27 built → ~25–40 [EST] ceiling**); README ~30+→~25–40, SSP ~3–6은 보수 subset으로 라벨. `make all` = **`test hazard unittests lint synth synth-glm formal`**(Makefile:58 — C1로 `synth-glm` 추가됨)이고 `bitacc`/`cache-study`/`bcov`/`formal-ind`/`coverage`/`spec-slow`/`cdc`는 별도.
