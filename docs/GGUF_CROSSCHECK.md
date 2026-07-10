# GGUF 교차 검증 — q4k_ref ≡ 진짜 ggml/llama.cpp (2026-07-10)

**닫힌 갭.** 지금까지 모든 "bit-exact" 게이트의 골든은 우리가 만든 ggml 재구현
(`tools/q4k_ref.py`)이었다 — 자기참조. 이 검증은 **실제 공개 GGUF 파일의 원시
블록 바이트**를 (A) q4k_ref와 (B) llama.cpp가 실제로 실행하는
`dequantize_row_q4_K`/`_q6_K`(빌드된 libggml 직링크)에 넣고 fp32 출력을
**비트 단위(uint32 view)로** 대조한다.

**결과 (Qwen2.5-0.5B-Instruct-GGUF q4_k_m, 공식 배포본):**

| 타입 | 텐서 | 가중치 | 판정 |
|---|---|---|---|
| Q4_K | 12 | 52,297,728 | **전부 비트 동일** |
| Q6_K | 12 | 52,297,728 | **전부 비트 동일** |

**신뢰 사슬:** RTL ≡ q4k_ref (기존 게이트) ∧ q4k_ref ≡ ggml (본 검증)
⇒ **RTL의 dequant ≡ 실제 GGUF 파일의 ggml dequant.**

**정직한 스코프:**
- 증명된 것: Q4_K/Q6_K **dequant 계층**의 재구현이 진짜 ggml과 동일 (실파일 기준).
- 남은 것: Q8_0은 이 파일에 없어 미포함 (자체 골든 게이트는 있음; q8_0 GGUF로
  동일 방법 재실행 가능). llama.cpp **런타임 전체**(어텐션/누산 순서)와의 수치
  동일성은 별개 문제 — 우리 계약은 "ggml dequant 비트 동일 + 문서화된 자체
  fp32 순서"이며 그 dequant 절반이 이제 실파일로 봉인됨.

**재현:**
```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=ON && cmake --build build --target ggml -j
cc -O2 -o dequant_dump <repo>/tools/... # tools/gguf_crosscheck.py 헤더 참조
python3 tools/gguf_crosscheck.py <model.gguf> <llama.cpp dir>
```
