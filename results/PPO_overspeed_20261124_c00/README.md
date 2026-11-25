# 규칙 장치 없는 BC→PPO 최종 모델

- 모델: `model.pt` (반복 100, `model_iter100.pt`와 가중치가 같다)
- critic: `critic.pt` (같은 시점의 별도 critic, 이어서 학습할 때 사용)
- 초기 모델: `results/BC_20261122_220742_calibrated_4740_signed_gaps_spawn20/training/model.pt`
- 학습: `python -m scripts.finetune_bc_ppo <BC model.pt> results/eval_fixed_routes_4740_idm_only_speedgain_20261121_013051 --iterations 100 --init-std 0.05 --lane-sharpness 2 --lr 5e-5 --anchor-coef 0.03 --overspeed-penalty 0 --seed 0`
- 학습과 평가 모두 ego 규칙 장치(차선 변경 안전 게이트, 출발 구간 차선 변경 금지, 속도 상한, 교차로 내부 차선 변경 금지)를 끈 상태다.
- 선택: 별도 시드 검증 세트(`results/eval_diverse_start_lanes_4740_validation_20261124`, 2,820회)에서 반복 50~100 중 최고 96.38%.
- 무작위 출발 차선 테스트(7,050회): 97.02% (BC 90.03%). 결과는 `eval_runs/iter100/evaluation_diverse_start_lanes_no_rule_barriers/`. 사례들이 SUMO를 공유하던 이전 방식이다.
- 같은 테스트를 사례별 새 SUMO(현재 기본, SUMO seed 0)로 돌린 결과: 97.06%. `eval_runs/iter100/evaluation_diverse_start_lanes_isolated/`.

`eval_runs/iter100/`은 고정 경로·무작위 출발 차선 평가 스크립트가 요구하는 BC 학습 폴더 형식(설정 파일 + `config.json`)에 이 모델의 가중치를 `model.pt`로 넣은 것이다. 이 PPO는 BC와 신경망 구조와 관측 설정이 같다.

자세한 내용은 `docs/ppo_finetuning.md`.
