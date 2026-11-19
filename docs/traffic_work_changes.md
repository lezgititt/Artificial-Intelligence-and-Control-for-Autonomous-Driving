# 이번 교통량 작업의 파일 정리

## 유지하는 구현

| 파일 | 역할 |
|---|---|
| `train_parallel.py` | 사용자가 요청한 Phase 2 전체 교통량 상한 6000 설정. 뷰어의 적응형 제어는 학습에 연결하지 않음 |
| `view_road.py` | 기본 6000 상한, 적응형 뷰어 실행, 고정 수요 비교, 차선/네트워크 측정 기록 |
| `env/lane_inflow.py` | 연속 진입 차선 측정, 녹색 배출량 학습, 대기열·잔여 차량 기반 공유 차선 유입 제어 |
| `tests/test_lane_inflow.py` | 유입 제어와 측정 회귀 검사 |
| `scripts/analyze_lane_traffic.py` | 초기에 사용한 고정 수요의 차선별 진입/배출 진단 도구 |
| `docs/adaptive_traffic.md` | 뷰어 제어 기준, 실행법, 버전별 검증 결과 |
| `scripts/evaluate_fixed_routes.py` | 사용자 지정 정식 평가. Windows SQLite 종료 처리와 선택적 실패 사례 추적(`--trace-failures-from`) 추가. 판정식은 유지 |

`env/sumo/highway.con.xml`, `highway.net.xml`, `highway.rou.xml`은 도로 생성/뷰어 실행에 따라
변하는 생성물이다. 기존 사용자 변경을 보존하기 위해 Git 기준으로 일괄 복원하지 않았다.
이번 작업으로 `road_config.py`의 도로 형상이나 신호 계획을 새로 바꾸지는 않았다.

## 보조 평가의 보관

이전에 별도로 만든 전체 배경 차량 추적 평가를 정식 평가와 구분했다.

- 코드: `analysis/archive/background_cohort_evaluation/evaluate_background_traffic.py.txt`
- 해당 검사: `analysis/archive/background_cohort_evaluation/test_evaluate_background_traffic.py.txt`
- 기존 결과: `analysis/archive/traffic_diagnostics/background_completion_6000/`

이 보조 평가의 100% 결과는 실제 진입한 차량을 추적하고 마지막에 유입을 끊어 배출시킨
결과다. 아래 고정 출발/고정 경로 벤치마크의 완주율과 직접 비교하면 안 된다.
보관한 테스트는 자동 테스트 탐색에서 제외되도록 텍스트 확장자로 저장했다.

## 사용자 지정 평가

Git에 포함하는 평가 요약·실패 원인 보고서:
`docs/analysis/traffic_fixed_routes_6000_20261120.md` 및 같은 이름의 `.json`.
아래 `analysis/` 경로는 로컬 원자료 위치이며 Git에서는 제외한다.

- 실행 스크립트: `scripts/evaluate_fixed_routes.py`
- 새 평가 묶음: `analysis/fixed_routes_current_6000_20261120/`
- 사용한 도로 스냅샷: `analysis/fixed_routes_current_6000_20261120/reference/road_config.py`
  준비용 원본은 `analysis/archive/traffic_diagnostics/fixed_routes_current_reference_20261120/`에 보관.
- 나머지 평가 문맥: 기존 `results/eval_fixed_routes_20261119_214714/reference/`의 저장 설정.
  `--expert` 모드에서는 체크포인트 정책을 실행하지 않으며 SUMO 운전 모델로 주행한다.
- 고정 6000대/시간, 800초 준비 운행, 802~950초의 75개 교통 상태.
- 24경로 × 75상태 = 운전 모델별 1800회, IDM/Krauss/EIDM/ACC 총 7200회.
- 중앙 차선 고정 출발, 출발 후 30m 차선변경 보호, 제한 시간 500초.
- **뷰어의 `LaneInflowController`는 이 평가에 연결되지 않는다.** 평가 스크립트 원래의
  `FixedTrafficSchedule`을 사용하므로 적응형 유입 제어의 효과를 측정하는 결과는 아니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -B scripts/evaluate_fixed_routes.py evaluate analysis/fixed_routes_current_6000_20261120 IDM --expert car_idm --replicas 32
```

각 운전 모델 폴더의 `evaluation_summary.json`, `route_evaluation.csv`, `summary.png`가 정식 결과다.
전체 보고서는 `analysis/fixed_routes_current_6000_20261120/REPORT.md`, 합산 수치는
같은 폴더의 `combined_summary.json`에 있다. 7200회 중 6906회 완주(95.92%)였고,
IDM 97.22%, Krauss 95.89%, EIDM 96.00%, ACC 94.56%였다.
모델별 1800개 case ID의 누락/중복이 없고, 각 경로가 정확히 75회씩 평가됐음을 확인했다.
이후 실패 294건을 추적했고 7200회 재현 결과가 원래와 일치했다. 보고서는 같은 폴더의
`audit/REPORT.md`다. 시간초과는 다중 녹색/비신호 구간 정체였고, 신호 위반 32건은
녹색 전환 경계의 판정 시점 오탐이 확인돼 실제 적색 통과로 해석하면 안 된다.
`env/sumo_env.py`의 판정식과 기존 결과 숫자는 아직 수정하지 않았다.
처음 준비하다 실패한 자료는 `analysis/archive/fixed_routes_prepare_failed_20261120/`에 보존했다.
이전 단계별 진단 결과는 `analysis/archive/traffic_diagnostics/`로 모았다.
루트에는 정식 평가 결과와 현재 뷰어 출력만 남겼다. `/analysis/` 전체를 `.gitignore`에
추가했으며, 로컬 결과 위치는 `analysis/README.md`에 정리했다.
