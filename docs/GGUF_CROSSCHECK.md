# GGUF 교차 검증 — q4k_ref ≡ 진짜 ggml/llama.cpp (2026-07-10, Q8_0 확장 2026-07-11)

**닫힌 갭.** 지금까지 모든 "bit-exact" 게이트의 골든은 우리가 만든 ggml 재구현
(`tools/q4k_ref.py`)이었다 — 자기참조. 이 검증은 **실제 공개 GGUF 파일의 원시
블록 바이트**를 (A) q4k_ref와 (B) llama.cpp가 실제로 실행하는
`dequantize_row_q4_K`/`_q6_K`/`_q8_0`(빌드된 libggml 직링크)에 넣고 fp32 출력을
**비트 단위(uint32 view)로** 대조한다. 허용오차 없음.

**검증 대상 파일 (둘 다 공식/공개 배포본, sha256은 HF x-linked-etag와 일치 확인):**

| 파일 | 출처 (HuggingFace) | 크기 | sha256 (short) |
|---|---|---|---|
| qwen2.5-0.5b-instruct-q4_k_m.gguf | Qwen/Qwen2.5-0.5B-Instruct-GGUF | 491,400,032 B | `74a4da8c` |
| SmolLM2-135M-Instruct-Q8_0.gguf | bartowski/SmolLM2-135M-Instruct-GGUF | 144,811,360 B | `5a139571` |

**결과 — Qwen2.5-0.5B q4_k_m (Q4_K/Q6_K 회귀 + 같은 파일 내 Q8_0):**

| 타입 | 텐서 | 가중치 | 판정 |
|---|---|---|---|
| Q4_K | 12 | 52,297,728 | **전부 비트 동일** |
| Q6_K | 12 | 52,297,728 | **전부 비트 동일** |
| Q8_0 | 13 | 137,510,912 | **전부 비트 동일** |

**결과 — SmolLM2-135M Q8_0 (Q8_0 전용 파일):**

| 타입 | 텐서 | 가중치 | 판정 |
|---|---|---|---|
| Q8_0 | 211 | 134,479,872 | **전부 비트 동일** |

**신뢰 사슬:** RTL ≡ q4k_ref (기존 게이트) ∧ q4k_ref ≡ ggml (본 검증)
⇒ **RTL의 dequant ≡ 실제 GGUF 파일의 ggml dequant.**

*(독립 재검증 2026-07-11: 두 파일 전체를 별도 세션에서 재실행, 상기 수치
그대로 BIT-EXACT PASSED. 이 과정에서 q4k_ref.py의 numpy≥2 비호환을 발견·수정 —
`np.int8(원시바이트)`가 numpy 1.x에선 조용히 랩되지만 2.x에선 OverflowError.
명시적 랩 `_s8()`로 교체 후 numpy 1.26/2.5 양쪽에서 셀프테스트 1600/1600 및
본 교차검증 전체 통과. 값은 불변 — 랩 결과 == 기존 1.x 동작.)*

**정직한 스코프:**
- 증명된 것: Q4_K/Q6_K/Q8_0 **dequant 계층**의 재구현이 진짜 ggml과 동일
  (실파일 2개, 총 376,586,240 가중치 기준). Q8_0은 서로 다른 두 배포본
  (Qwen 공식 믹스 내 13텐서 + bartowski 전용 파일 211텐서)으로 교차 확인됨.
- 남은 것: F16/Q5_0 등 이 파일들에 있는 나머지 타입은 본 스크립트 스코프 밖
  (F16은 dequant가 아닌 단순 변환, Q5_0은 q4k_ref에 골든이 없는 타입). llama.cpp
  **런타임 전체**(어텐션/누산 순서)와의 수치 동일성은 별개 문제 — 우리 계약은
  "ggml dequant 비트 동일 + 문서화된 자체 fp32 순서"이며 그 dequant 절반이
  실파일로 봉인됨.

**재현 (커밋된 하네스로 전부 재구축 가능):**
```bash
# 1. ggml 사이드 하네스 빌드: llama.cpp clone + libggml 빌드 + dequant_dump 링크
#    (tools/dequant_dump.c: 원시 블록 -> ggml dequantize_row_* -> fp32 덤프)
tools/build_dequant_dump.sh <llamacpp_dir>     # 없으면 clone, 있으면 재사용

# 2. 실파일 교차 검증 (Q4_K/Q6_K/Q8_0 텐서 전수, 비트 단위)
python3 tools/gguf_crosscheck.py <model.gguf> <llamacpp_dir>
```
검증 당시 llama.cpp 커밋: `8f114a9b573b69035299f9b924047f53c1e22c7e`
(빌드: Release, BUILD_SHARED_LIBS=ON, GGML_METAL=OFF — CPU 레퍼런스 경로).
