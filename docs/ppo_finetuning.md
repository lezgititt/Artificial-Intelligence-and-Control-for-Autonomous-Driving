# BC→PPO 파인튜닝

최종 BC(`results/BC_20261122_220742_calibrated_4740_signed_gaps_spawn20`)에서 시작해 이벤트 보상으로 PPO 파인튜닝을 한다. 학습과 평가 모두 ego의 규칙 기반 장치를 끈다(`docs/final_BC_trianing.md`의 "규칙 장치 없는 재평가").

## 최종 모델

`results/PPO_overspeed_20261124_c00/model.pt` (과속 패널티 없음, 학습 시드 0, 반복 100)

| 평가 | BC | 최종 PPO |
|---|---|---|
| 무작위 출발 차선 테스트 (시드 52101–52125, 7,050회) | 90.03% | **97.02%** |
| └ 충돌 / 적신호 위반 / 차선 실패 / 시간 초과 | 186 / 199 / 293 / 25 | 65 / 7 / 130 / 8 |
| 검증 세트 (시드 62101–62110, 2,820회) | 90.25% | 96.38% |

## 이전 PPO가 실패한 원인

첫 PPO(`results/PPO_20261124_100850`)는 혼합 평가에서 BC 97.7%를 80.2%로 떨어뜨렸다.

- 탐험 노이즈가 너무 컸다. 가속 표준편차 0.61(약 ±3.3m/s²)로 BC를 고정한 채 학습 환경을 돌리면 에피소드의 84%가 충돌로 끝났다. 노이즈를 0.05로 줄이면 87%가 도착했다. PPO가 결정적 BC와 전혀 다른 정책의 데이터로 학습하고 있었다.
- 속도 상한을 켠 채 학습해서, 상한을 끈 평가에서 최고 57m/s까지 가속했다.

## 방법

`algorithms/anchored_ppo.py`, `scripts/finetune_bc_ppo.py`

- BC actor에서 시작한다. 가속 표준편차 0.05, 차선 logit을 2배로 키워 샘플링되는 차선 변경을 줄인다(argmax는 그대로라 결정적 정책은 BC와 같다).
- critic은 별도 신경망이다. BC에는 가치 출력이 없어서, 몸통을 공유하면 초기 가치 학습이 BC 특징을 바꾼다. 처음 5회 반복은 critic만 학습한다.
- actor 손실에 BC 정책과의 KL(가속 평균 차이와 차선 분포)을 `anchor_coef`로 더한다.
- 학습률 5e-5, `target_kl` 0.01, 반복당 5 epoch, 512개 환경 × 64스텝 = 32,768스텝, 100회 반복.
- 저장되는 체크포인트는 `HybridActorCritic` 형식이라 기존 평가 스크립트가 그대로 읽는다.

## 보상

`env/bc_ppo_reward.py`. 스텝 −0.0001, 종료 시 도착 +1 / 차선 실패·시간 초과 −1 / 충돌·적신호 위반 −2.

- 정차 보너스: 적신호 접근로에서 정지선 1.0m 앞(대기열이면 앞차 뒤 2.0m)에 멈추면 최대 0.01. 직접 측정과 보상 계산이 일치했지만, BC가 이미 최대치의 96%를 받아 학습에는 거의 영향이 없다.
- 과속 패널티(`--overspeed-penalty`, 기본 0): 스텝마다 `c × max(0, v − 제한속도×1.1) / 제한속도`. 제한속도는 현재 차선의 값이다(일반 도로 13.89m/s, 원형도로 8.33m/s, 교차로 내부는 회전 속도로 유턴 3.0m/s). 허용 오차 10%는 IDM 전문가의 속도 편차(`speed_dev` 0.1)와 같다.

과속 패널티 계수는 학습 환경에서 BC의 과속을 재서 정했다. 과속으로 도착이 빨라져 얻는 이득을 넘으려면 c ≥ 0.009이고, c = 0.05면 최악의 에피소드 패널티가 도착 보상(+1)과 같아진다.

## 실험

모든 실행은 같은 BC에서 시작하고 앵커 0.03을 쓴다(첫 행 제외). 검증은 별도 시드 세트(2,820회)다.

| 실행 | 조건 | 검증 최고 | 경로 |
|---|---|---|---|
| 앵커 0.1, 0.03 | 학습 중 교차로 내부 차선 변경 금지 켜짐 | (혼합 검증으로 선택) | `results/PPO_anchored_20261124_a10`, `_a03` |
| c=0.02 | 과속 패널티 | 95.85% (반복 70) | `results/PPO_overspeed_20261124_c02` |
| c=0.05 | 과속 패널티 | 94.82% (반복 100) | `results/PPO_overspeed_20261124_c05` |
| c=0, 시드 0 | 패널티 없음 | **96.38% (반복 100)** | `results/PPO_overspeed_20261124_c00` |
| c=0, 시드 1 | 패널티 없음 | 94.54% (반복 70) | `results/PPO_overspeed_20261124_c00s1` |
| c=0, 시드 2 | 패널티 없음 | 94.89% (반복 100) | `results/PPO_overspeed_20261124_c00s2` |

같은 검증 세트에서 앵커 0.03 반복 50(학습 중 교차로 내부 차선 변경 금지)은 96.88%였다.

규칙 장치 없는 테스트(모든 평가에서 장치 끔):

| 모델 | 혼합 교통 (시드 21701–21712) | 무작위 출발 차선 | 고정 경로 |
|---|---|---|---|
| BC | 97.93% | 90.03% | 94.56% |
| 앵커 0.03, 반복 50 | 98.60% | 97.84% | 98.94% |
| c=0.02, 반복 70 | 98.27% | 96.20% | 98.67% |
| c=0, 시드 0, 반복 100 (최종) | - | 97.02% | - |

검증·테스트 그림: `results/PPO_anchored_20261124_validation/learning_curves.png`, `results/PPO_overspeed_20261124_validation/learning_curves.png`.

## 알게 된 점

- 같은 설정이라도 학습 시드에 따라 검증 최고가 94.5~96.4%로 달라진다. 설정을 비교하려면 시드를 여러 개 돌려야 한다.
- 앵커 0.03 반복 50은 무작위 출발 차선 테스트에서 최종 모델보다 높았다(짝지은 비교에서 그 모델만 도착 150건, 최종 모델만 도착 92건). 이 모델은 학습 중 교차로 내부 차선 변경 금지가 켜져 있어 최종 후보에서 뺐다. 교차로 내부 차선 변경은 에피소드의 4~6%에서만 일어났고, 그런 에피소드의 실패율이 오히려 낮아서 이 금지가 성능 차이의 원인이라는 근거는 찾지 못했다.
- 과속 패널티 c=0.02는 패널티 없는 실행보다 과속 평균 초과량을 약 9% 줄였고, 검증 성공률은 평균 0.4%p 낮았다. c=0.05는 평균 1.6%p 낮았다. 정책이 차선별 제한속도를 관측하지 못하는 것이 효과가 작은 원인으로 보인다.
- 시드 1은 반복 80 이후 평균 속도가 오르면서 적신호 위반이 14건에서 79건으로 늘었다.

## 남은 문제

- 과속: 최종 모델도 BC보다 과속이 많다.
- 차선 실패: 테스트 실패 210건 중 130건이다. A4→A3(78.7%)과 A1→A5(92.3%)가 가장 약하다.
- 무작위 출발 차선 평가에는 ego가 한 대뿐이라, ego 여러 대가 만드는 정체는 혼합 교통 평가에서만 드러난다.

## 재현

```bash
python -m scripts.finetune_bc_ppo results/BC_20261122_220742_calibrated_4740_signed_gaps_spawn20/training/model.pt \
  results/eval_fixed_routes_4740_idm_only_speedgain_20261121_013051 \
  --iterations 100 --init-std 0.05 --lane-sharpness 2 --lr 5e-5 --anchor-coef 0.03 --overspeed-penalty 0 --seed 0
python scripts/evaluate_diverse_start_lanes.py prepare results/eval_fixed_routes_4740_idm_only_speedgain_20261121_013051 \
  results/eval_diverse_start_lanes_4740_validation_20261124 --seeds 62101 62102 62103 62104 62105 62106 62107 62108 62109 62110
python scripts/evaluate_diverse_start_lanes.py evaluate results/eval_diverse_start_lanes_4740_20261123_172237 \
  --model results/PPO_overspeed_20261124_c00/eval_runs/iter100/model.pt
python scripts/plot_anchored_ppo.py results/PPO_overspeed_20261124_validation \
  results/PPO_overspeed_20261124_c00 results/PPO_overspeed_20261124_c00s1 results/PPO_overspeed_20261124_c00s2 \
  results/PPO_overspeed_20261124_c02 --labels "c=0 seed 0 (final)" "c=0 seed 1" "c=0 seed 2" "c=0.02 seed 0" \
  --output results/PPO_overspeed_20261124_validation/learning_curves.png
```
