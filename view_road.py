"""도로 구조 확인 스크립트 — 학습 없이 도로만 눈으로 확인한다.

env/road_config.py 를 수정한 뒤, 학습을 돌리기 전에
"도로가 의도대로 만들어졌는지 / 교통 밀도가 적당한지"를
sumo-gui 화면으로 빠르게 확인하는 용도.

사용법:
    python view_road.py             # 차선별 적응형 유입으로 sumo-gui 실행
    python view_road.py --fixed-traffic  # 고정 6000대/시간 수요와 비교
    python view_road.py --netedit   # netedit(SUMO 도로 편집기)로 도로망 열기
    python view_road.py --nogui     # 화면 없이 설정 검증만 (GUI 없는 서버용)

기본 모드(sumo-gui) 동작:
    적응형 모드는 자동 재생하며 차선별 배출량과 대기열로 유입을 조절한다.
    --fixed-traffic은 기존처럼 ▶ 버튼으로 재생하는 고정 수요 비교 모드다.
    차선별 측정값은 --metrics-dir 아래 CSV로 저장한다.

관찰 포인트:
    - 차선 수 / 도로 길이가 road_config.ROAD 대로인지  ← 재생 없이 바로 확인 가능
    - (▶ 재생 후) 전체 교통량(--traffic-vph, 기본 6000)이 너무 한산하거나 막히지 않는지
      → 정체가 심하면 학습 난이도가 급상승하니 여기서 미리 조정
    - 재생 속도는 상단 Delay(ms) 값으로 조절 (크게 잡을수록 느리게)

※ netedit 관련 주의:
    netedit에서 도로를 수정하고 저장해도, 다음번 train.py 실행 시
    road_builder가 env/road_config.py 기준으로 파일을 **덮어쓴다**.
    이 프로젝트에서 도로의 원본(source of truth)은 어디까지나
    env/road_config.py 이므로, 영구적인 수정은 그 파일에서 할 것.
    (netedit은 구조를 뜯어보거나 좌표를 확인하는 용도로 쓰면 좋다)
"""
import argparse
import csv
import json
import math
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from env.road_config import (ROAD, EGO, TRAFFIC, NOHYEONG,
                             configure_parallel_training)
from env.mdp_config import SIMULATION
from env import road_builder
# sumo_env를 import하면 SUMO_HOME 자동 감지 로직이 함께 실행된다
from env.sumo_env import SumoHighwayEnv  # noqa: F401  (자동 감지 목적의 import)
from sumolib import checkBinary
from env.parallel_traffic import TrafficCurriculum, parse_traffic_flows
from env.lane_inflow import LaneInflowController

BASE = os.path.dirname(os.path.abspath(__file__))

# road_config의 신호 실험(SIGNAL_EXPERIMENT)이 켜져 있으면 import 시점에
# 배경 교통이 전부 0으로 꺼진다(학습 실험용 설정). view_road의 목적은
# "도로 구조 + 배경 교통 밀도"를 눈으로 확인하는 것이므로, 배경 교통을
# 원래 값(vehs_per_hour / 원형도로 vph)으로 되살린다.
# ego는 여기서 투입하지 않으므로 화면에는 배경 차량만 보인다.
configure_parallel_training(phase=2)


def configure_view_traffic(sumocfg, total_vph, adaptive=False):
    """Use the Phase 2 route proportions and Poisson demand in standalone SUMO."""
    directory = os.path.dirname(sumocfg)
    route_path = os.path.join(directory, "highway.rou.xml")
    tree = ET.parse(route_path)
    root = tree.getroot()
    routes = {r.get("id"): r.get("edges").split() for r in root.findall("route")}
    nodes = root.findall("flow")
    profile = TrafficCurriculum(max_vph=6000, total_vph=True, ring_reference_vph=1500)
    flows = parse_traffic_flows(nodes, routes, os.path.join(directory, "highway.net.xml"),
                               TRAFFIC["vehs_per_hour"], profile)
    for node in nodes:
        root.remove(node)
    for name, edges in routes.items():
        if root.find(f"route[@id='{name}']") is None:
            ET.SubElement(root, "route", id=name, edges=" ".join(edges))
    for index, flow in enumerate(flows):
        if adaptive:
            continue
        # SUMO exp(rate) takes vehicles/second. Independent Poisson route streams
        # have the same distribution as training's combined Poisson schedule.
        ET.SubElement(root, "flow", id=f"view_{index}", type=flow.vehicle_type,
                      route=flow.route, begin="0", end="100000",
                      period=f"exp({total_vph * flow.rate_per_central_vph / 3600:.12g})",
                      departLane=flow.lane, departSpeed=flow.speed)
    ET.indent(tree, space="    ")
    tree.write(route_path, encoding="utf-8", xml_declaration=True)
    central_vph = total_vph * sum(flow.rate_per_central_vph for flow in flows
                                if not flow.route.startswith("background_"))
    return central_vph, total_vph - central_vph, flows, routes


def run_adaptive(sumocfg, flows, routes, args):
    """Drive the viewer through TraCI; live measurements remain outside training."""
    import traci
    from traci.constants import VAR_LANE_ID, VAR_LANEPOSITION, VAR_SPEED, VAR_LENGTH

    controller = LaneInflowController(Path(sumocfg).with_name('highway.net.xml'), flows,
                                      routes, args.traffic_vph, np.random.default_rng(args.seed),
                                      args.queue_short, args.queue_long)
    output = args.metrics_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'queue_corridors.json').write_text(json.dumps({
        lane: dict(central_lanes=controller.queue_roots[lane],
                   length_m=max(offset + length for _, offset, length in segments),
                   segments=[dict(lane=name, distance_from_central_stop_m=offset, length_m=length)
                             for name, offset, length in segments])
        for lane, segments in controller.corridors.items()}, indent=2), encoding='utf-8')
    print(f'실시간 측정 기록: {output}', flush=True)
    options = [checkBinary('sumo' if args.nogui else 'sumo-gui'), '-c', sumocfg,
               '--step-length', str(SIMULATION['step_length']), '--seed', str(args.seed),
               '--max-depart-delay', '125', '--collision.action', 'remove',
               '--no-step-log', 'true', '--no-warnings', 'true']
    if not args.nogui:
        options += ['--start', '--delay', str(SIMULATION['gui_delay'])]
    started = False
    states = {}
    totals = dict(entered=0, arrived=0, active=0, pending=0, teleports=0, time=0.)
    try:
        traci.start(options)
        started = True
        tls_ids = traci.trafficlight.getIDList()
        signals = {tid: traci.trafficlight.getRedYellowGreenState(tid) for tid in tls_ids}
        controller.observe({}, signals)
        fields = list(controller.rows(0)[0])
        with (output / 'lanes.csv').open('w', encoding='utf-8-sig', newline='') as lane_file, \
                (output / 'network.csv').open('w', encoding='utf-8-sig', newline='') as network_file:
            lane_writer = csv.DictWriter(lane_file, fieldnames=fields)
            network_writer = csv.DictWriter(network_file, fieldnames=[*totals, 'requested_vph',
                                                                     'requested', 'admissions_withheld'])
            lane_writer.writeheader()
            network_writer.writeheader()
            while not args.nogui or traci.simulation.getTime() < args.seconds:
                now = traci.simulation.getTime()
                controller.spawn_due(traci, now, states)
                traci.simulationStep()
                for vehicle in traci.simulation.getDepartedIDList():
                    traci.vehicle.subscribe(vehicle, (VAR_LANE_ID, VAR_LANEPOSITION, VAR_SPEED, VAR_LENGTH))
                subscriptions = traci.vehicle.getAllSubscriptionResults()
                states = {v: (s[VAR_LANE_ID], s[VAR_LANEPOSITION], s[VAR_SPEED], s[VAR_LENGTH])
                          for v, s in subscriptions.items()}
                signals = {tid: traci.trafficlight.getRedYellowGreenState(tid) for tid in tls_ids}
                controller.observe(states, signals, traci.simulation.getStartingTeleportIDList())
                # Prune completed/discarded IDs; retain pending vehicles for admission checks.
                pending = set(traci.simulation.getPendingVehicles())
                controller.route_by_id = {v: route for v, route in controller.route_by_id.items()
                                          if v in states or v in pending}
                totals['entered'] += traci.simulation.getDepartedNumber()
                totals['arrived'] += traci.simulation.getArrivedNumber()
                totals['teleports'] += traci.simulation.getStartingTeleportNumber()
                totals.update(time=traci.simulation.getTime(), active=len(states), pending=len(pending))
                if controller.update_rates(totals['time']):
                    lane_writer.writerows(controller.rows(totals['time']))
                    network_writer.writerow(totals | dict(requested_vph=sum(controller.rates),
                                                         requested=controller.stats['requested'],
                                                         admissions_withheld=controller.stats['admissions_withheld']))
                    lane_file.flush()
                    network_file.flush()
    except traci.exceptions.FatalTraCIError:
        if args.nogui or not started:
            raise
        print('GUI 연결이 종료되었습니다.')
    finally:
        if started:
            try:
                traci.close()
            except traci.exceptions.FatalTraCIError:
                pass
        summary = dict(totals, **controller.stats, max_requested_vph=args.traffic_vph,
                       queue_short=args.queue_short, queue_long=args.queue_long,
                       seed=args.seed, lanes=controller.rows(totals['time']))
        (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"측정 결과: 유입 {totals['entered']}대 / 배출 {totals['arrived']}대 / "
          f"도로 안 {totals['active']}대 / 진입 대기 {totals['pending']}대")
    print(f'차선별 녹색 배출량·대기열 기록: {output}')


def print_summary(total_vph, central_vph, right_vph, adaptive=False):
    """현재 road_config 내용을 표로 요약 출력."""
    print("─" * 46)
    print(" 현재 도로 설정 (env/road_config.py)")
    print("─" * 46)
    print(f"  도로 길이      : {ROAD['length']} m")
    print(f"  차선 수        : {ROAD['num_lanes']}")
    print(f"  제한속도       : {ROAD['speed_limit']} m/s "
          f"(≈ {float(ROAD['speed_limit']) * 3.6:.0f} km/h)")
    label = '전체 유입 상한' if adaptive else '전체 요청 수요'
    print(f"  배경 교통량    : {total_vph:g} 대/시간 ({label}, 무작위 유입)")
    print(f"  경로 비중 기준 : 중앙 {central_vph:.1f} + 우회도로 {right_vph:.1f}")
    print("  시작 조건      : 빈 도로에서 시작; 250초 이상 재생 후 정체 확인")
    print(f"  배경차 최고속도: {TRAFFIC['max_speed']} m/s")
    print(f"  ego 가속/감속  : +{EGO['accel']} / -{EGO['decel']} m/s²")
    print("─" * 46)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="도로 구조를 GUI로 확인")
    parser.add_argument("--netedit", action="store_true",
                        help="sumo-gui 대신 netedit(도로 편집기)로 열기")
    parser.add_argument("--nogui", action="store_true",
                        help="화면 없이 설정 검증만 (GUI 없는 환경용)")
    parser.add_argument("--seconds", type=int, default=120,
                        help="--nogui 검증 시 시뮬레이션 길이(초). 기본 120초. "
                             "(GUI 모드는 시간 제한 없이 사용자가 직접 제어)")
    parser.add_argument("--traffic-vph", type=float, default=6000,
                        help="전체 요청 교통량 상한(대/시간, 기본 6000)")
    parser.add_argument('--fixed-traffic', action='store_true', help='적응형 제어 없이 고정 수요로 비교')
    parser.add_argument('--queue-short', type=float, default=.3, help='짧은 대기열 기준 비율(기본 0.3)')
    parser.add_argument('--queue-long', type=float, default=.6, help='긴 대기열 기준 비율(기본 0.6)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--metrics-dir', type=Path, default=Path(BASE) / 'analysis' / 'view_road_adaptive')
    args = parser.parse_args()
    if not math.isfinite(args.traffic_vph) or args.traffic_vph <= 0:
        parser.error("--traffic-vph는 0보다 커야 합니다.")
    if not 0 <= args.queue_short < args.queue_long < .9:
        parser.error('--queue-short < --queue-long < 0.9, 두 기준은 0 이상이어야 합니다.')
    if args.seconds <= 0:
        parser.error('--seconds는 0보다 커야 합니다.')

    # 1) road_config 기준으로 SUMO 파일 (재)생성 — 수정사항이 즉시 반영된다
    sumocfg = road_builder.build(ROAD, EGO, TRAFFIC, os.path.join(BASE, "env", "sumo"))
    net_file = os.path.join(BASE, "env", "sumo", "highway.net.xml")
    print(f"도로 생성 완료: {sumocfg}")
    adaptive = not (args.fixed_traffic or args.netedit)
    central_vph, right_vph, flows, routes = configure_view_traffic(sumocfg, args.traffic_vph, adaptive)
    print_summary(args.traffic_vph, central_vph, right_vph, adaptive)
    if adaptive:
        print('  유입 제어: 차선별 녹색 평균 배출량 × 짧음 1.10 / 김 0.90, 포화 90% 이상 보류')
        print('  누적 방지: 녹색 종료 후 미통과 차량이 남으면 공유 차선 유입 추가 억제')
        print('  시작 시 기준: 공유 차선별 초기 예산 제한; 해당 경로 대기 차량이 있던 녹색부터 평균 갱신')
    traffic_options = ["--max-depart-delay", "125", "--collision.action", "remove",
                       '--seed', str(args.seed)]

    # 2) 요청한 모드로 열기
    if args.netedit:
        # netedit: SUMO에 포함된 도로망 편집기. 노드/엣지/차선을 시각적으로 확인.
        print("netedit 실행 중... (창을 닫으면 종료)")
        subprocess.run([checkBinary("netedit"), net_file])

    elif adaptive:
        run_adaptive(sumocfg, flows, routes, args)

    elif args.nogui:
        # 화면 없이 짧게 돌려서 "설정이 유효하고 차가 실제로 투입되는지"만 검증
        result = subprocess.run(
            [checkBinary("sumo"), "-c", sumocfg,
             "--step-length", str(SIMULATION["step_length"]),
             "--end", str(args.seconds),
             "--no-warnings", "true", "--no-step-log", "true",
             "--duration-log.statistics", "true", *traffic_options],
            capture_output=True, text=True)
        # SUMO의 통계 요약(투입/도착 차량 수 등)을 그대로 보여준다
        print(result.stderr.strip() or result.stdout.strip())
        print("설정 검증 OK" if result.returncode == 0 else "설정 오류!")

    else:
        # sumo-gui: 도로를 "정적으로" 띄운다.
        #   --start, --quit-on-end, --end 를 모두 주지 않는 것이 핵심:
        #     --start 없음      → 자동 재생 안 함. 창이 뜨면 도로만 보이고,
        #                         사용자가 ▶(플레이) 버튼을 눌러야 차가 움직인다.
        #     --quit-on-end 없음 → 시뮬레이션이 끝나도 창이 저절로 닫히지 않는다.
        #     --end 없음        → 시간 제한 없음. 원하는 만큼 돌려볼 수 있다.
        #   --delay 100 : 재생 시 스텝당 100ms (0이면 눈으로 못 따라갈 만큼 빠름).
        #                 GUI 상단 Delay 입력칸에서 실행 중에도 바꿀 수 있다.
        print("sumo-gui 실행 중...")
        print("  · 창이 뜨면 도로만 보입니다. ▶(플레이) 버튼을 눌러야 차가 움직입니다.")
        print(f"  · 확대: env/mdp_config.py의 gui_zoom={SIMULATION['gui_zoom']}")
        print("  · 속도 조절: 상단 Delay(ms) 값을 조정 (크게 = 느리게)")
        print("  · 종료하려면 창을 닫으세요.")
        subprocess.run(
            [checkBinary("sumo-gui"), "-c", sumocfg,
             "--step-length", str(SIMULATION["step_length"]),
             "--delay", str(SIMULATION["gui_delay"]), *traffic_options])
