"""Measure background-only lane flows using the Phase 2 demand profile.

Run from the repository root. No ego vehicles or episode resets are included.
Lane indices follow SUMO: zero is the rightmost lane in the travel direction.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import libsumo as api
except ImportError:
    import traci as api
import numpy as np
from sumolib import checkBinary
from traci.constants import VAR_LANE_ID, VAR_SPEED, VAR_ROUTE_ID

from env import road_config
from env.mdp_config import SIMULATION
from env.parallel_traffic import TrafficCurriculum, TrafficSchedule, parse_traffic_flows
from env.road_builder import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vph', type=int, default=6000)
    parser.add_argument('--seconds', type=int, default=3600)
    parser.add_argument('--warmup', type=int, default=250)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, default=Path('analysis/traffic_6000'))
    args = parser.parse_args()
    if args.vph <= 0 or args.seconds <= 0 or args.warmup < 0:
        parser.error('Demand and measurement duration must be positive; warmup must be nonnegative')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    road_config.configure_parallel_training(2)
    config = build(road_config.ROAD, road_config.EGO, road_config.TRAFFIC, str(out / 'sumo'))
    directory = Path(config).parent
    root = ET.parse(directory / 'highway.rou.xml').getroot()
    routes = {r.get('id'): r.get('edges').split() for r in root.findall('route')}
    curriculum = TrafficCurriculum(max_vph=6000, total_vph=True,
                                  max_depart_delay_seconds=125, ring_reference_vph=1500)
    flows = parse_traffic_flows(root.findall('flow'), routes, directory / 'highway.net.xml',
                               road_config.TRAFFIC['vehs_per_hour'], curriculum)
    for node in root.findall('flow'):
        root.remove(node)
    for route, edges in routes.items():
        if root.find(f"route[@id='{route}']") is None:
            ET.SubElement(root, 'route', id=route, edges=' '.join(edges))
    ET.ElementTree(root).write(directory / 'highway.rou.xml', encoding='utf-8', xml_declaration=True)
    net = ET.parse(directory / 'highway.net.xml').getroot()
    lanes = {lane.get('id'): edge.get('id') for edge in net.findall('edge')
             if not edge.get('id').startswith(':') for lane in edge.findall('lane')}
    entering, leaving, departures, arrivals = (Counter() for _ in range(4))
    samples, halted = Counter(), Counter()
    stopped_routes = Counter()
    uturn_lanes = {c.get('from') + '_' + c.get('fromLane'): int(c.get('linkIndex'))
                   for c in net.findall('connection') if c.get('tl') == 'nC' and c.get('dir') == 't'}
    stopped_by_signal = Counter()
    totals = Counter()
    previous = {}
    step = SIMULATION['step_length']
    schedule = TrafficSchedule(flows)
    rng = np.random.default_rng(args.seed)
    api.start([checkBinary('sumo'), '-c', config, '--step-length', str(step), '--seed', str(args.seed),
               '--collision.action', 'remove', '--no-warnings', 'true', '--no-step-log', 'true',
               *curriculum.sumo_options()])
    try:
        schedule.reset(api, 0., args.vph, rng, set())
        baseline = None
        for _ in range(round((args.warmup + args.seconds) / step)):
            now = api.simulation.getTime()
            measured = now >= args.warmup
            if measured and baseline is None:
                baseline = dict(active=api.vehicle.getIDCount(), pending=len(api.simulation.getPendingVehicles()))
            serial = schedule.serial
            schedule.spawn_due(api, now, lambda name: name, rng)
            api.simulationStep()
            for v in api.simulation.getDepartedIDList():
                api.vehicle.subscribe(v, (VAR_LANE_ID, VAR_SPEED, VAR_ROUTE_ID))
            states = api.vehicle.getAllSubscriptionResults()
            current = {v: state[VAR_LANE_ID] for v, state in states.items()}
            if measured:
                signal_state = api.trafficlight.getRedYellowGreenState('nC')
                totals['requested'] += schedule.serial - serial
                departed = api.simulation.getDepartedIDList()
                arrived = api.simulation.getArrivedIDList()
                totals['departed'] += len(departed)
                totals['arrived'] += len(arrived)
                totals['teleports'] += api.simulation.getStartingTeleportNumber()
                totals['colliding_vehicle_events'] += api.simulation.getCollidingVehiclesNumber()
                for v in departed:
                    departures[current.get(v, 'unknown')] += 1
                for v in arrived:
                    arrivals[previous.get(v, 'unknown')] += 1
                # Count longitudinal edge transitions, not lane changes within an edge.
                for v, lane in current.items():
                    old = previous.get(v)
                    if lane in lanes and lanes.get(old) != lanes[lane]:
                        entering[lane] += 1
                    if lane in lanes:
                        samples[lane] += 1
                        halted[lane] += states[v][VAR_SPEED] < 0.1
                        if states[v][VAR_SPEED] < 0.1 and lane in uturn_lanes:
                            stopped_routes[lane, states[v][VAR_ROUTE_ID]] += step
                            signal = signal_state[uturn_lanes[lane]]
                            stopped_by_signal[lane, signal] += step
                for v, old in previous.items():
                    if old in lanes and lanes.get(current.get(v)) != lanes[old]:
                        leaving[old] += 1
            previous = current
        totals['active_end'] = api.vehicle.getIDCount()
        totals['pending_end'] = len(api.simulation.getPendingVehicles())
    finally:
        api.close()
    factor = 3600 / args.seconds
    with (out / 'lane_counts.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['edge', 'lane', 'entered', 'left', 'entered_per_hour', 'left_per_hour',
                         'network_departures', 'network_arrivals', 'mean_vehicles', 'mean_halted'])
        for lane, edge in sorted(lanes.items()):
            writer.writerow([edge, lane, entering[lane], leaving[lane], entering[lane] * factor,
                             leaving[lane] * factor, departures[lane], arrivals[lane],
                             samples[lane] * step / args.seconds, halted[lane] * step / args.seconds])
    demand = [{'route': flow.route, 'from': routes[flow.route][0], 'to': routes[flow.route][-1],
               'depart_lane': flow.lane, 'requested_vph': args.vph * flow.rate_per_central_vph}
              for flow in flows]
    with (out / 'boundary_counts.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['direction', 'edge', 'lane', 'count', 'vehicles_per_hour'])
        for direction, field, counts in [('in', 'from', departures), ('out', 'to', arrivals)]:
            edges = {route[field] for route in demand}
            for lane, edge in sorted(lanes.items()):
                if edge in edges:
                    writer.writerow([direction, edge, lane, counts[lane], counts[lane] * factor])
    report = dict(settings=vars(args) | {'output': str(out), 'step': step},
                  baseline=baseline, totals=dict(totals), routes=demand,
                  network_departures_by_lane=dict(departures), network_arrivals_by_lane=dict(arrivals),
                  note='Background only; continuous simulation without ego or episode resets. '
                       'Arrivals use the last observed lane before exit; collisions/teleports can '
                       'also cause edge departures. Entry/exit counts exclude same-edge lane changes.')
    report['uturn_lane_diagnostics'] = {
        lane: {'stopped_vehicle_seconds_by_route': {route: value for (name, route), value
                                                   in stopped_routes.items() if name == lane},
               'stopped_vehicle_seconds_by_uturn_signal': {signal: value for (name, signal), value
                                                           in stopped_by_signal.items() if name == lane}}
        for lane in sorted(uturn_lanes)}
    (out / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('baseline', 'totals', 'network_departures_by_lane',
                                           'network_arrivals_by_lane')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
