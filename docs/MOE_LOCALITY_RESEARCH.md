# MoE expert locality / caching / flash-streaming — 문헌 조사 (2026-07)

**왜 이 문서인가.** 단일 사용자 bit-exact 스트리밍 박스가 **100 tok/s**에 도달하려면
`14GB/token routed-expert 스트림`이 (MTP 수락률 A) × (캐시 재사용률 h)로 ~14× 상각되어야
한다. 그 A와 h가 문헌에서 어디까지 측정되어 있는지 7-agent 병렬 웹 스윕(172 검색)으로
조사했다. 결론 요약: **연구는 밀집해 있으나, "대역폭을 실제로 절약하는 재사용-h 0.9"를
GLM급 fine-grained MoE에서 보여준 논문은 없다** — 그 지점이 이 프로젝트가 직접 측정해야
할 공백이다. (아래 본문은 워크플로 종합 에이전트의 출력 그대로.)

---

# MoE LLM 추론 웹 리서치 종합 (6개 앵글 통합)

## 1. 연구가 존재하는가 — 분야 지도

이 분야는 2023년 말 이후 4갈래로 밀집 발전했다. (a) **expert offloading/캐싱 시스템**: Eliseev-Mazur의 Mixtral-offloading(2023)이 원조이고, MoE-Infinity(2024), ProMoE(2024), ExpertFlow(2024), HOBBIT(2024), AdapMoE(ICCAD'24), Fiddler(ICLR'25), FineMoE/fMoE(EuroSys'26), FlashMoE(2026), SpecMD(2026)로 이어지며 일관된 결론은 "LRU는 MoE에 틀린 정책"이다. (b) **라우팅 통계 측정**: Mixtral 기술보고서(2024)의 토큰 간 expert 반복률, ExFlow(IPDPS'24)의 층간 affinity, Patterns behind Chaos(2025)의 DeepSeek-V3/Kimi-K2급 24,000-요청 트레이스, ReMoE(2026)의 EOR 정량화. (c) **flash 스트리밍 + 하드웨어**: LLM-in-a-flash(Apple, 2023), PowerInfer-2(2024), EdgeMoE(2023), Cambricon-LLM(MICRO'24, NAND 칩렛), Duplex(MICRO'24, PIM), Expert Streaming 칩렛(2026), Sieve(2026). (d) **speculative decoding × expert prefetch 결합**: DeepSeek-V3 MTP(2024), EAGLE-2/3, SpecMoEOff(2025), MoE-SpeQ(2025), SP-MoE(2025), SpecExec(NeurIPS'24). 단, **공개된 spec-decode+expert-offload 시스템은 전부 CPU DRAM(PCIe 23–32GB/s) 기반이고, flash 직결 스트리밍과 speculation을 결합한 시스템·bit-exact 제약을 다룬 연구·fab된 LLM-MoE flash ASIC은 없다** — 본 프로젝트가 노리는 조합 자체가 문헌상 공백이다.

## 2. 핵심 측정치 표

| 논문/연도 | 측정 대상 | 숫자 |
|---|---|---|
| Eliseev-Mazur 2023 | Mixtral-8x7B LRU hit / end tok/s | k=2 캐시 ~40–60%, k=4 ~60–75%, k=6 ~85%; 1-layer-ahead 예측 recall ~70%; 2–4 tok/s (PCIe, 2-bit expert) |
| Mixtral report 2024 | 연속 토큰 expert 반복률 | layer 0: 13.6–14.9% (랜덤 12.5%≈무의미), layer 15: first-or-second 61.6–67.0% — 층별 편차 큼 |
| MoE-Infinity 2024 | 요청 내 working set | ~100-expert 모델은 <5%, Mixtral ~25%만 활성; 단 제3자 측정 hit rate은 16.73–17% (10% 캐시) |
| ExpertFlow 2024 | 예측 캐시 vs LRU hit | 25% 캐시: 71.89% vs 36.22%; 50% 캐시: 91.90% vs 76.61%; 경로 예측기 80–95% |
| ProMoE 2024 | LRU 한계 | 50% 캐시여도 expert 로딩이 critical path의 60.4% 차단; 학습 예측기 84.7% |
| HOBBIT 2024 | 층간 gate 예측 | 다음 층 top-1 96%, 2–3층 앞 ~90% |
| Fate 2025 | 층간 lookahead 예측 | 전체 97.15%, decode 시 78.79%; 초기층(0–3) 고정 시 hit 99.08% |
| Pre-Attention 2025 | 현 층 사전 예측 | exact-match 93–98% (DeepSeek-V2-Lite/Qwen3-30B), ~1.5x over-fetch 시 ~99% |
| PreScope 2025 | 학습 예측기 top-4 hit | DeepSeek 94%, Mixtral 99%, Qwen3 97%; 최저 80% |
| MoE-Beyond 2025 | 학습 예측 vs 휴리스틱 (10% 캐시) | hit 17% → 72% (DeepSeek-V2-Lite); 예측 정확도 97.5% |
| SP-MoE 2025 | spec-draft 기반 prefetch | draft의 expert 예측 88.94% top-1; hit 40.06% vs LRU계 16.7–21.9%; TPOT 1.07–3.5x |
| MoE-SpeQ 2025 | 양자화 self-draft + prefetch | 토큰 수락 >90%, expert 라우팅 예측 90.9%; hit 96.25–99.85% (32–16GB 캐시); DeepSeek-V2-Lite 13.02 tok/s |
| SpecMD 2026 | 미세 캐시 + least-stale | 5% 캐시로 OLMoE hit >88%; 그러나 end 속도 ~1.4–2.3 tok/s |
| FlashMoE 2026 | 실 NVMe(7.4GB/s) 스트리밍 | expert 로딩 = decode 시간의 >70%; ML 정책 +21% hit vs LRU, +22% tok/s |
| ReMoE 2026 | DeepSeek계 토큰 간 overlap (EOR) | 27.3% (6-of-64, 랜덤 ~9.4%의 2배지만 절대값 낮음); router 미세조정 시 34.5% |
| Patterns behind Chaos 2025 | prefill→decode 예측력 | prefill top-20이 decode top-5의 ~90% 커버 (DeepSeek-V3/Llama4/Qwen3/Kimi-K2) |
| DeepSeek-V3 2024 | MTP-1 수락률 | 85–90%, ~1.8x TPS |
| GLM-5 보고서 2026 / SGLang | GLM MTP accept length | GLM-5: 2.76 (vs DeepSeek-V3.2 2.55); GLM-5.2 저지연 시 4+; GLM-4.5 권장 draft 3–4토큰; 프로덕션 수락률 0.57–0.85 |
| EAGLE-3 2025 / SpecExec 2024 | 배치-1 draft 상한 | τ=4.05–7.5, 3–6.5x; SpecExec 대형 트리로 패스당 최대 20토큰 수락 |
| PowerInfer-2 2024 / Cambricon-LLM MICRO'24 | flash 스트리밍 실기 tok/s | 폰에서 Mixtral-47B 11.68 tok/s (I/O를 critical path의 13.7%로); NAND 칩렛 70B 3.44 tok/s |
| M3 Max SSD 스트리밍 2026 / R1 mmap 2025 | 실사용 flash 스트리밍 | Qwen3.5-397B 4.36–5.5 tok/s (이론 상한 ~18.6 @17.5GB/s); R1-671B NVMe mmap 1.3–3.5 tok/s, SSD 실효 2–5GB/s |
| Strix Halo / DGX Spark 2025–26 | DRAM 상주 상한 (비교점) | gpt-oss-120b ~55 tok/s @256GB/s, ~50 tok/s @273GB/s — decode ≈ bandwidth/active-bytes |
| SSD 에너지 연구 2025 | flash vs HBM 에너지 | 110.8 vs 4.2 pJ/b (~26x); 토큰당 3.8–12.5x; 단 prefetch 시 지연 페널티 1.25–1.32x |

## 3. h≈0.9 필요조건에 대한 문헌의 시사

**산수부터**: 14GB/token × (1−h) ≤ 100GB/s ÷ 100tok/s = 1GB/token이려면 MTP 없이는 h ≥ 0.93. MTP accept length A로 나누면 필요 h는 1 − A/14: A=2.76(GLM-5 측정치)이면 h ≥ ~0.80, A=4+(GLM-5.2 저지연)이면 h ≥ ~0.71.

**낙관 근거**: (1) 예측 기반 캐싱의 도달치가 목표대에 걸쳐 있다 — 10% 캐시에서 72%(MoE-Beyond), 5% 캐시에서 >88%(SpecMD), FineMoE 실워크로드 75–85%, MoE-SpeQ는 speculative prefetch로 96–99.9%. (2) 예측 정확도 자체는 충분히 높다 — 층간 gate 예측 90–97%(HOBBIT/Fate/Pre-Attention), over-fetch로 ~99%, 그리고 **spec-decode draft가 공짜 expert 오라클**(88.9–90.9%, SP-MoE/MoE-SpeQ)이라는 점은 본 설계의 MTP 체인과 정확히 합치한다. (3) prefill 라우팅이 decode hot set을 ~90% 예고(Patterns behind Chaos)하므로 세션 시작 시 캐시 워밍 가능. (4) 요청 내 working set이 작고(<5%, MoE-Infinity) 태스크 단위로 재발(eMoE) — 단일 사용자 시나리오에 유리. (5) MTP 이득은 concurrency 1에서 최대(AMD 측정 2.11x@1 → 1.25x@64) — 단일 사용자 전제와 정합. (6) 예측·prefetch·eviction은 출력을 바꾸지 않으므로 bit-exact와 충돌 없음.

**비관 근거**: (1) **문헌의 "hit rate"는 대부분 prefetch 포함 수치다.** prefetch hit은 지연은 숨기지만 flash에서 바이트를 여전히 읽는다 — 100 tok/s의 병목은 지연이 아니라 **대역폭**이므로, 바이트를 절약하는 것은 순수 재사용(residency) hit뿐이다. 순수 재사용 지표는 훨씬 낮다: DeepSeek계 연속 토큰 top-K overlap 27.3%(ReMoE), LRU 캐시 hit 16.7–40%(SP-MoE 계열 측정), Mixtral도 layer 0은 랜덤 수준. GLM은 DeepSeek류 fine-grained MoE이므로 낙관 시나리오가 아니라 이쪽이 기본값일 가능성이 높다. (2) MoE-SpeQ의 96–99.9%는 16–32GB 캐시 + 소형 모델(Phi/Qwen1.5/V2-Lite) 조건 — 14GB/token급 라우팅 스트림 규모에서 검증된 바 없다. (3) 검증 패스의 expert 합집합은 draft 깊이에 따라 커져 A의 실효 상각률이 A보다 작다(SpecMoEOff가 draft 길이에 따른 수확체감을 명시). (4) **end-to-end 실증 최고치와의 격차**: flash 스트리밍 실측은 1.3–13 tok/s(R1 mmap, M3 Max, PowerInfer-2, MoE-SpeQ)이고 DRAM 전체 상주조차 256–273GB/s에서 50–55 tok/s다. 100 tok/s는 문헌 최고치의 ~8배. (5) bit-exact 제약이 문헌의 주력 레버 다수를 봉쇄한다: 저비트 miss fetch(HOBBIT), expert 스킵(AdapMoE −25% fetch), 2-bit expert(Eliseev-Mazur, M3 Max — 2-bit는 tool calling 파손), router 미세조정(ReMoE +27%), pre-gate 재학습(Pre-gated MoE). (6) 에너지: flash 읽기 110.8 pJ/b vs HBM 4.2 pJ/b — 3.8–12.5x/token. 결론적으로 문헌은 "h 0.8–0.9는 예측+spec-prefetch로 도달 가능한 범위"임을 시사하되, **대역폭 절약형(재사용) h로서의 0.9는 어떤 논문도 GLM급 fine-grained MoE에서 보여준 적이 없다.**

## 4. 문헌이 답하지 않는 것 — 직접 측정 항목

1. **GLM 라우팅 통계 자체**: 측정치는 Mixtral/DeepSeek/Qwen/OLMoE/Switch에 몰려 있고 GLM-4.x/5의 토큰 간 EOR, 층별 반복률, 요청 내 working set 곡선은 미공개. GLM-4.5 MTP 수락률의 공식 발표도 없음(GLM-5의 2.76만 존재). → stock(미세조정 불가, bit-exact) GLM 라우터의 배치-1 트레이스를 직접 뽑아야 함.
2. **residency-hit vs prefetch-hit 분해**: 문헌은 둘을 합산 보고. 대역폭 한정 설계에서는 "flash에서 안 읽은 바이트 비율"이 진짜 h — 캐시 용량(우리 DRAM GB) 대비 이 값의 곡선을 자체 측정해야 함.
3. **MTP 체인 깊이별 expert 합집합 크기**: 검증 패스당 실제 스트리밍 바이트 = |union(experts of k draft tokens)| × expert 크기. A_eff = k×14GB ÷ 합집합 바이트 곡선은 어느 논문에도 없음 — h와 A가 독립이 아니라는 점이 핵심 미지수.
4. **목표 워크로드에서의 GLM MTP accept length**: 프로덕션 수치가 0.57–0.85/token으로 워크로드 의존(수학/코드가 높음, CNN/DM 낮음 — ST-MoE). bit-exact 검증 규칙 하에서 depth 3–4 체인의 실측 분포 필요.
5. **100GB/s flash의 실효 지속 대역폭**: 실측 사례는 정격 대비 크게 낮음(Gen5 NVMe에서 2–5GB/s 지속, R1 mmap). expert 크기 단위 읽기(≥32KiB 규칙, Apple), row-column bundling식 레이아웃, ECC/컨트롤러 오버헤드 하의 지속 대역폭을 자체 검증해야 함.
6. **요청 경계에서의 캐시 재워밍 비용**: hot set이 요청마다 바뀐다(MoE-Infinity) — 단일 사용자 멀티턴에서 prefill-기반 워밍(top-20→90% 커버)이 우리 모델·캐시 크기에서 몇 토큰 만에 h를 회복시키는지.
7. **h와 tok/s의 결합 검증**: 문헌 최고 조합(spec prefetch + 예측 캐시)도 40% hit/13 tok/s대 — h≈0.9 × A≈3 × 100GB/s가 실제 100 tok/s로 환산되는지는 본 프로젝트의 시뮬레이션/RTL 측정으로만 답할 수 있는 미답 영역.