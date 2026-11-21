"""Prepare and evaluate a shared fixed-center, 24-route SUMO benchmark."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
import csv
from dataclasses import asdict, replace
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import shutil
import sqlite3
import sys
from time import perf_counter
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms.ppo import PPO
from env import road_builder
from env.adaptive_traffic_schedule import AdaptiveTrafficSchedule, adaptive_schedule_factory
from env.background_demand import CALIBRATED_VPH, set_weighted_od_flow_rates
from env.collision_detection import configure_collision_gaps
from env.control_timing import lane_change_timing_allowed
from env.ego_routes import read_ego_route_groups
from env.parallel_sumo import ParallelSumo, libsumo
from env.parallel_traffic import TrafficCurriculum, TrafficSchedule
from env.traffic_snapshots import TrafficSnapshotBank
from env.uniform_idm import separate_idm_type
from scripts.evaluate import OUTCOMES, plt
from scripts.evaluate_quick_bc import plot_bc_training
from scripts.evaluate_quick_phase1 import STEP, summarize
from scripts.run_bc51_fixed_start import fixed_start_context
from test import load_saved_configs
from utils.evaluator import _DrivingStats, parallel_route_rows

DEMAND = CALIBRATED_VPH
TRAFFIC_SEED = 21701
MAX_REPLICAS_PER_WORKER = 64
SNAPSHOT_TIMES = tuple(range(802, 951, 2))
EXPERTS = ("car_idm", "car_krauss", "car_eidm", "car_acc")
STATUSES = ("arrival", "collision", "red_light", "wrong_lane", "timeout")


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def include_central_a1_a3(background_routes, ego_routes):
    """Use the central route for the direct ID and keep the explicit ring variant."""
    central = ET.parse(background_routes).find("./route[@id='r_A1_A3']")
    if central is None:
        raise ValueError("Missing central A1→A3 background route")
    tree = ET.parse(ego_routes)
    direct = tree.find("./route[@id='ego_d_A1_A3']")
    if direct is None:
        direct = ET.SubElement(tree.getroot(), "route", id="ego_d_A1_A3")
    direct.set("edges", central.attrib["edges"])
    tree.write(ego_routes, encoding="utf-8", xml_declaration=True)


def use_current_road(context):
    """Replace a copied reference's road snapshot with env/road_config.py, keeping its evaluation tail."""
    saved = (context / "road_config.py").read_text(encoding="utf-8")
    marker = "# BC evaluation context"
    if marker not in saved:
        raise ValueError("Reference road_config.py has no evaluation context block")
    current = (Path(__file__).resolve().parents[1] / "env/road_config.py").read_text(encoding="utf-8")
    (context / "road_config.py").write_text(
        current.rstrip() + "\n\n" + marker + saved.split(marker, 1)[1], encoding="utf-8")


def prepare(suite, reference, traffic_vph, adaptive=False, uniform_idm_vph=0, *,
            traffic_seed=0, current_road=False):
    context = suite / "reference"
    if (reference / "evaluation_config.json").is_file():
        shutil.copytree(reference, context)
        curriculum = TrafficCurriculum(**read_json(context / "evaluation_config.json")["traffic_curriculum"])
    else:
        _, curriculum, _ = fixed_start_context(reference, context, max_vph=traffic_vph)
    if current_road:
        use_current_road(context)
    curriculum = replace(curriculum, max_vph=traffic_vph, snapshot_warmup_seconds=800)
    if adaptive:
        curriculum = replace(curriculum, total_vph=True, max_depart_delay_seconds=125)
    road, _, _ = load_saved_configs(str(context / "model.pt"))
    config = road_builder.build(road["ROAD"], road["EGO"], road["TRAFFIC"], str(suite / "sumo"))
    background_routes = suite / "sumo/highway.rou.xml"
    ego_routes = suite / "sumo/highway.ego.rou.xml"
    set_weighted_od_flow_rates(background_routes, traffic_vph)
    if uniform_idm_vph:
        tree = ET.parse(background_routes)
        separate_idm_type(tree)
        tree.write(background_routes, encoding="utf-8", xml_declaration=True)
    include_central_a1_a3(background_routes, ego_routes)
    gaps = configure_collision_gaps(config, ego_gap_m=road["EGO"]["collision_gap_m"],
                                    background_gap_m=road["TRAFFIC"]["collision_gap_m"])
    central, ring = read_ego_route_groups(ego_routes)
    if (len(central), len(ring)) != (19, 5):
        raise ValueError(f"Expected 19 central and 5 ring routes, found {len(central)} and {len(ring)}")
    routes = central + ring
    entries = {r.get("id"): r.get("edges").split()[0]
               for r in ET.parse(ego_routes).iter("route")}
    lane_counts = {e.get("id"): len(e.findall("lane"))
                   for e in ET.parse(suite / "sumo/highway.net.xml").iter("edge")}
    bank = TrafficSnapshotBank(config, curriculum, road["TRAFFIC"]["vehs_per_hour"],
                               (traffic_vph,), STEP, traffic_seed, refresh=False,
                               capture_steps=[int(time / STEP) for time in SNAPSHOT_TIMES], adaptive=adaptive,
                               uniform_idm_vph=uniform_idm_vph)
    try:
        destination = suite / "traffic_states.sqlite"
        with closing(sqlite3.connect(bank.path)) as source, closing(sqlite3.connect(destination)) as target:
            source.backup(target)
    finally:
        bank.close()
    cases = []
    with closing(sqlite3.connect(destination)) as db:
        signal_ticks = dict(db.execute("SELECT snapshot_tick, signal_tick FROM signal_ticks"))
        for worker, times in enumerate(np.array_split(SNAPSHOT_TIMES, 4)):
            for value in times:
                secs = int(value)
                snapshot_tick = int(secs / STEP)
                if db.execute("SELECT count(*) FROM states WHERE vph=? AND tick=?", (traffic_vph, snapshot_tick)).fetchone()[0] != 1:
                    raise ValueError(f"Expected exactly one traffic state at {secs}s")
                for route in routes:
                    cases.append(dict(case_id=len(cases), worker=worker, snapshot_time_s=secs,
                                      snapshot_tick=snapshot_tick, tick=signal_ticks[snapshot_tick],
                                      seed=0, traffic_seed=[TRAFFIC_SEED + traffic_seed, secs, 0], route=route,
                                      start_lane=lane_counts[entries[route]] // 2))
    if len({(c["snapshot_tick"], c["route"]) for c in cases}) != len(SNAPSHOT_TIMES) * len(routes):
        raise ValueError("Fixed case list must contain exactly one case per saved state and route")
    write_json(suite / "cases.json", cases)
    write_json(suite / "suite.json", dict(traffic_vph=traffic_vph, adaptive_traffic=adaptive,
        uniform_idm_vph=uniform_idm_vph, traffic_seed=traffic_seed, current_road=current_road,
        warmup_seconds=800, routes=routes,
        settings=asdict(curriculum), vehicle_types=road["TRAFFIC"]["controllers"],
        collision_detection=gaps, simulation_step_seconds=STEP, max_seconds=500,
        max_replicas_per_worker=MAX_REPLICAS_PER_WORKER, workers=4, seeds_per_snapshot=1,
        snapshot_times_seconds=SNAPSHOT_TIMES,
        spawn_protect_m=30))
    print(f"Prepared {len(cases):,} cases in {suite}", flush=True)


class FixedTrafficSchedule(TrafficSchedule):
    """Make arrival, routing, type and lane requests use only the case's RNG."""

    def __init__(self, flows, controllers, origin_lanes):
        super().__init__(flows)
        self.types = [f"car_{name}" for name in controllers]
        self.deviations = [c["speed_dev"] for c in controllers.values()]
        self.probabilities = np.asarray([c["probability"] for c in controllers.values()])
        self.probabilities /= self.probabilities.sum()
        self.origin_lanes = origin_lanes

    def spawn_due(self, api, now, prefix, rng):
        requests = []
        while self.next_departure <= now:
            idx = self.cumulative.searchsorted(rng.random() * self.cumulative[-1], side="right")
            flow = self.flows[idx]
            type_index = int(rng.choice(len(self.types), p=self.probabilities))
            lane = (str(rng.integers(self.origin_lanes[flow.route])) if flow.lane == "random" else flow.lane)
            factor = float(rng.normal(1.0, self.deviations[type_index]))
            vid = prefix(f"bg_{self.serial}")
            self.serial += 1
            api.vehicle.add(vid, prefix(flow.route), typeID=self.types[type_index],
                            departLane=lane, departSpeed=flow.speed)
            api.vehicle.setSpeedFactor(vid, float(np.clip(factor, .2, 2.0)))
            self.ids.add(vid)
            requests.append((vid, flow.route, self.types[type_index]))
            self.next_departure += self._headway(rng)
        return requests


class FixedCaseSumo(ParallelSumo):
    def _respawn(self, ego, route=None, **kwargs):
        if self.adaptive_factory:
            if not isinstance(ego.background, AdaptiveTrafficSchedule):
                ego.background = self.adaptive_factory()
        elif not isinstance(ego.background, FixedTrafficSchedule):
            ego.background = FixedTrafficSchedule(self.traffic_flows, self.prototype.traffic["controllers"],
                                                 self.origin_lanes)
        obs = super()._respawn(ego, route, **kwargs)
        if self.adaptive_factory:
            key = kwargs["snapshot_tick"]
            state, = self._snapshot_db.execute("SELECT state FROM adaptive_states WHERE vph=? AND tick=?",
                                               (ego.background.vph, key)).fetchone()
            ego.background.restore_checkpoint(json.loads(state), libsumo.simulation.getTime())
        if self.expert_type:
            vehicle = libsumo.vehicle
            vehicle.setType(ego.ego_id, self.expert_type)
            vehicle.setSpeedMode(ego.ego_id, 31)
            vehicle.setSpeed(ego.ego_id, -1)
            return ego._get_obs()
        return obs

    @staticmethod
    def _record_adaptive_requests(ego, info):
        schedule = ego.background
        idm_requests = sum(schedule.uniform_idm.requests.values()) if schedule.uniform_idm else 0
        info["background_requests"] = schedule.controller.stats["requested"] + idm_requests
        info["uniform_idm_requests"] = idm_requests

    def _advance(self, egos, actions):
        if not self.expert_type:
            for ego, transition in zip(egos, super()._advance(egos, actions), strict=True):
                if self.adaptive_factory:
                    ego.background.ids.intersection_update(self._live_background)
                    self._record_adaptive_requests(ego, transition[4])
                if ego.replica_index in getattr(self, "audit_before", {}):
                    self.audit_after[ego.replica_index] = audit_state(
                        ego, self.audit_before[ego.replica_index].get("signals", []))
                yield transition
            return

        vehicle = libsumo.vehicle
        before = []
        for ego in egos:
            vid = ego.ego_id
            edge = vehicle.getRoadID(vid)
            allowed = lane_change_timing_allowed(edge, vehicle.getLanePosition(vid),
                                                 vehicle.getRouteIndex(vid), ego.lc_spawn_protect)
            vehicle.setLaneChangeMode(vid, 1621 if allowed else 0)
            signals = ego._upcoming_signals()
            connected = bool(signals) and ego._lane_follows_route(vehicle.getLaneID(vid))
            distance = vehicle.getDistance(vid) if signals else None
            before.append((edge, vehicle.getLaneIndex(vid), vehicle.getSpeed(vid), signals, connected, distance))
            ego.background.spawn_due(libsumo, libsumo.simulation.getTime(), ego.prefix, ego.traffic_random)
        libsumo.simulationStep()
        active = set(vehicle.getIDList())
        self._live_background = active | set(libsumo.simulation.getPendingVehicles())
        collided = set(libsumo.simulation.getCollidingVehiclesIDList())
        arrived = set(libsumo.simulation.getArrivedIDList())
        for ego, (edge, lane, speed, signals, connected, distance) in zip(egos, before, strict=True):
            vid = ego.ego_id
            ego.present = vid in active
            delta = (vehicle.getLaneIndex(vid) - lane
                     if ego.present and vehicle.getRoadID(vid) == edge and not edge.startswith(":") else 0)
            state = (0., delta, delta, bool(delta), lane, speed, signals, connected, distance, None)
            transition = ego._finish_step(state, collided=vid in collided, arrived=vid in arrived)
            if self.adaptive_factory:
                ego.background.ids.intersection_update(self._live_background)
                self._record_adaptive_requests(ego, transition[4])
            if ego.replica_index in getattr(self, 'audit_before', {}):
                # Capture before step_evaluation removes a terminated vehicle.
                self.audit_after[ego.replica_index] = audit_state(
                    ego, self.audit_before[ego.replica_index].get('signals', []))
            yield transition


def audit_state(ego, reference_signals=()):
    """Read-only diagnostics; do not alter vehicle controls or consume RNG."""
    vehicle = libsumo.vehicle
    signal_states = [dict(tls=s['tls'], link=s['link'],
                         state=libsumo.trafficlight.getRedYellowGreenState(s['tls'])[s['link']])
                     for s in reference_signals]
    if not ego.present:
        return dict(present=False, reference_signal_states=signal_states)
    vid = ego.ego_id
    lane = vehicle.getLaneID(vid)
    route = vehicle.getRoute(vid)
    index = vehicle.getRouteIndex(vid)
    signals = []
    for tid, link, distance, state in vehicle.getNextTLS(vid):
        key = tid, link
        offset = ego._signal_stop_offsets.get(key, 0.)
        signals.append(dict(tls=tid, link=link, raw_distance=distance, offset=offset,
                            adjusted_distance=distance-offset, state=state,
                            stop_line=ego._signal_stop_lines.get(key)))
    leader = vehicle.getLeader(vid, 1e6)
    leader_data = None
    if leader and leader[0]:
        other, gap = leader
        leader_data = dict(vehicle=other, gap=gap, speed=vehicle.getSpeed(other),
                           lane=vehicle.getLaneID(other), route=vehicle.getRouteID(other),
                           signals=vehicle.getNextTLS(other))
    connected = ego._lane_follows_route(lane)
    lc = None
    if not connected and vehicle.getSpeed(vid) < .5 and not lane.startswith(':'):
        lc = {str(direction): vehicle.getLaneChangeStatePretty(vid, direction) for direction in (-1, 1)}
    return dict(present=True, time=libsumo.simulation.getTime(), road=vehicle.getRoadID(vid), lane=lane,
                position=vehicle.getLanePosition(vid), remaining=libsumo.lane.getLength(lane)-vehicle.getLanePosition(vid),
                speed=vehicle.getSpeed(vid), distance=vehicle.getDistance(vid), route_index=index,
                next_edge=route[index+1] if 0 <= index < len(route)-1 else None,
                connected=connected, signals=signals, leader=leader_data, lane_change=lc,
                reference_signal_states=signal_states)


def evaluate_worker(suite, context, scratch, cases, expert_type, replicas, trace_ids=()):
    import torch

    torch.set_num_threads(1)
    model = context / "model.pt"
    _, mdp, train = load_saved_configs(str(model))
    reference, _, _ = load_saved_configs(str(suite / "reference/model.pt"))
    obs_config = mdp["OBSERVATION"]
    action = mdp["ACTION"]
    reward = train["REWARD"]
    spec = read_json(suite / "suite.json")
    traffic_vph = spec["traffic_vph"]
    curriculum = TrafficCurriculum(**spec["settings"])
    records = _DrivingStats(len(cases))
    env = FixedCaseSumo(min(replicas, len(cases)), dict(
        cfg_path=suite / "sumo/highway.sumocfg", road=reference["ROAD"], ego=reference["EGO"],
        traffic=reference["TRAFFIC"], mdp_sim=mdp["SIMULATION"], mdp_obs=obs_config,
        action=action, reward=reward), scratch, seed=0, randomize_signals=True,
        randomize_departures=False, independent_episode_rng=True,
        traffic_curriculum=curriculum, fixed_traffic_vph=traffic_vph,
        traffic_snapshot_path=str(suite / "traffic_states.sqlite"))
    env.expert_type = expert_type
    env.adaptive_factory = None
    if spec.get("adaptive_traffic"):
        env.adaptive_factory = adaptive_schedule_factory(
            suite / "sumo/highway.sumocfg", scratch, env.traffic_flows, env.signal_ids, spec.get("uniform_idm_vph", 0))
    if env.prototype.step_length != STEP or env.prototype.max_steps * STEP != 500:
        raise ValueError("Fixed evaluation requires a 0.5-second step and a 500-second time limit")
    routes_xml = ET.parse(scratch / "parallel.rou.xml")
    net_xml = ET.parse(suite / "sumo/highway.net.xml")
    lane_counts = {e.get("id"): len(e.findall("lane")) for e in net_xml.iter("edge")}
    env.origin_lanes = {r.get("id").removeprefix("r0_"):
                        lane_counts[r.get("edges").split()[0].removeprefix("r0_")]
                        for r in routes_xml.iter("route") if r.get("id").startswith("r0_")}
    policy = None
    if expert_type is None:
        policy = PPO(obs_dim=env.prototype.obs_dim, act_dim=env.prototype.act_dim,
                     **(train["HPARAMS"] | {"device": "cpu"})).policy
        policy.load_state_dict(torch.load(model, map_location="cpu", weights_only=True))
        policy.eval()
    case_seeds = [np.random.SeedSequence(c["traffic_seed"]) for c in cases]
    queue = iter(range(len(cases)))
    active = {slot: next(queue) for slot in range(env.num_envs)}
    completed = 0
    adaptive_requests = {}
    trace_ids = set(trace_ids)
    trace_file = None
    try:
        if trace_ids:
            trace_file = (scratch.parent / f'{scratch.name}_trace.jsonl').open('w', encoding='utf-8')
        obs = env.reset([cases[j]["route"] for j in active.values()],
                        signal_ticks=[cases[j]["tick"] for j in active.values()],
                        snapshot_ticks=[cases[j].get("snapshot_tick", cases[j]["tick"]) for j in active.values()],
                        replica_seeds=[case_seeds[j] for j in active.values()])
        if any(env.egos[i].start_lane != cases[j]["start_lane"] for i, j in active.items()):
            raise RuntimeError("Departure lane differs from the fixed case list")
        while active:
            slots = list(active)
            with torch.inference_mode():
                actions = (policy.det_action(torch.from_numpy(obs[slots])).numpy() if policy is not None
                           else np.zeros((len(slots), env.prototype.act_dim), np.float32))
            traced = {slot: audit_state(env.egos[slot]) for slot in slots
                      if cases[active[slot]]['case_id'] in trace_ids}
            env.audit_before = traced
            env.audit_after = {}
            transitions = list(env.step_evaluation(slots, actions))
            for slot, (x, reward, terminated, truncated, info) in zip(slots, transitions, strict=True):
                index = active[slot]
                if slot in traced:
                    trace_file.write(json.dumps(dict(case_id=cases[index]['case_id'],
                        step=env.egos[slot].step_count, before=traced[slot],
                        after=env.audit_after[slot] if slot in env.audit_after else audit_state(env.egos[slot]),
                        terminated=terminated, truncated=truncated,
                        flags={key: info.get(key) for key in ('arrived', 'collided', 'wrong_lane', 'red_light_violation')})) + '\n')
                records.record(index, reward, truncated, info)
                if not (terminated or truncated):
                    obs[slot] = x
                    continue
                if env.adaptive_factory:
                    adaptive_requests[index] = {key: info[key] for key in
                                                ("background_requests", "uniform_idm_requests")}
                completed += 1
                if completed % 250 == 0:
                    print(f"{scratch.name}: {completed}/{len(cases)}", flush=True)
                following = next(queue, None)
                if following is None:
                    del active[slot]
                    continue
                active[slot] = following
                case = cases[following]
                ego = env.egos[slot]
                ego.traffic_random = np.random.default_rng(case_seeds[following].spawn(4)[3])
                obs[slot] = env._respawn(ego, case["route"], traffic_vph=traffic_vph, signal_tick=case["tick"],
                                         snapshot_tick=case.get("snapshot_tick"))
                if ego.start_lane != case["start_lane"]:
                    raise RuntimeError("Refilled case used the wrong departure lane")
        rows = parallel_route_rows(records, env.prototype.visibility, env.prototype.step_length)
        return [case | row | adaptive_requests.get(i, {})
                for i, (case, row) in enumerate(zip(cases, rows, strict=True))]
    finally:
        if trace_file is not None:
            trace_file.close()
        env.close()


def plot_summary(out, label, rows, summary, routes, model_run):
    fig = plt.figure(figsize=(24, 14 if model_run else 11), layout="constrained")
    outer = fig.add_gridspec(3 if model_run else 2, 1,
                             height_ratios=(1.6, 4.4, 1) if model_run else (4.4, 1))
    if model_run:
        plot_bc_training(fig, outer[0].subgridspec(3, 2), model_run, evaluated_epoch=summary["evaluated_epoch"])
    charts = outer[-2].subgridspec(1, 2, width_ratios=(1, 2))
    bars, heat = (fig.add_subplot(charts[i]) for i in range(2))
    routes = sorted(routes, key=lambda route: (summary["by_route"][route]["success_rate"], route))
    labels = [route.removeprefix("ego_d_").replace("ego_r_RING_", "Ring ").replace("_", " → ", 1)
              if not route.startswith("ego_r_") else f"Ring {'→'.join(route.split('_')[-2:])}" for route in routes]
    left = np.zeros(len(routes))
    for status in STATUSES:
        counts = np.array([summary["by_route"][route]["statuses"].get(status, 0) for route in routes])
        width = 100 * counts / np.array([summary["by_route"][route]["episodes"] for route in routes])
        if counts.any():
            bars.barh(labels, width, left=left, color=OUTCOMES[status][2], label=OUTCOMES[status][0])
        left += width
    bars.invert_yaxis()
    bars.set(xlabel="Share of runs (%)", title="Outcomes by route", xlim=(0, 100))
    bars.legend(loc="lower center", bbox_to_anchor=(.5, 1.04), ncol=3, fontsize=8)
    counts = {}
    for row in rows:
        key = row["route"], row["snapshot_time_s"]
        counts[key] = counts.get(key, 0) + (row["status"] == "arrival")
    times = sorted({row["snapshot_time_s"] for row in rows})
    grid = [[counts.get((route, time), np.nan) for time in times] for route in routes]
    max_arrivals = summary["conditions"].get("seeds_per_snapshot", summary["conditions"].get("seeds_per_offset", 1))
    raster = heat.imshow(grid, cmap=plt.get_cmap("YlGn", max_arrivals + 1), vmin=-.5, vmax=max_arrivals+.5,
                          interpolation="nearest", aspect="auto")
    ticks = range(0, len(times), max(1, len(times) // 8))
    heat.set_xticks(list(ticks), [times[i] for i in ticks])
    heat.set_yticks(range(len(routes)), labels)
    heat.set(xlabel="Saved simulation time (s)", title="Arrivals by saved state")
    fig.colorbar(raster, ax=heat, ticks=range(max_arrivals + 1), label=f"Arrivals / {max_arrivals}")
    clearance = outer[-1].subgridspec(1, 2)
    for i, (reference, title) in enumerate((("stop_line", "Stop line"), ("queued_vehicle", "Stopped leader"))):
        ax = fig.add_subplot(clearance[i])
        stat = summary["overall"]["red_wait_clearance_by_reference"][reference]
        if stat["n"]:
            med, low, high = (stat[k] for k in ("median_m", "q1_m", "q3_m"))
            ax.errorbar(med, 0, xerr=[[med-low], [high-med]], marker="o", capsize=5)
            ax.set_title(f"{title} · median {med:.2f} m · n={stat['n']}")
        else:
            ax.set_title(f"{title} · no qualifying stops")
        ax.set(xlabel="Clearance (m), median and 25th–75th percentiles", yticks=[])
        ax.set_xlim(left=0)
    counts = summary["overall"]["statuses"]
    total = len(rows)
    headline = f"{label} · {total:,} runs · {summary['conditions']['traffic_vph']:,} veh/h · fixed center · first 30 m protected"
    outcomes = " · ".join(f"{OUTCOMES[s][0]} {counts.get(s, 0):,} ({counts.get(s, 0) / total:.1%})" for s in STATUSES)
    fig.suptitle(f"{headline}\n{outcomes}", fontsize=15)
    fig.savefig(out / "summary.png", dpi=125)
    plt.close(fig)


def run(suite, label, expert, model, replicas, smoke, trace_failures_from=None, *, output=None):
    started = perf_counter()
    reference = suite / "reference"
    spec = read_json(suite / "suite.json")
    cases = read_json(suite / "cases.json")
    if output is None:
        if model is None:
            output = suite / label
        else:
            experiment = model.parent
            if experiment.name == "training":
                experiment = experiment.parent
            output = experiment / (f"evaluation_{label}" if label else "evaluation")
    out = output
    label = label or out.name
    out.mkdir()
    context = out / "context"
    if model is not None:
        fixed_start_context(model.parent, context, max_vph=spec["traffic_vph"])
        shutil.copy2(model, context / "model.pt")
    else:
        shutil.copytree(reference, context)
    shards = [[c for c in cases if c["worker"] == worker] for worker in range(4)]
    if smoke:
        shards = [shard[:smoke] for shard in shards]
    trace_ids = []
    if trace_failures_from is not None:
        with (trace_failures_from / 'route_evaluation.csv').open(newline='') as file:
            trace_ids = [int(row['case_id']) for row in csv.DictReader(file) if row['status'] != 'arrival']
    print(f"{label}: {sum(map(len, shards)):,} cases, {replicas} replicas/worker", flush=True)
    with ProcessPoolExecutor(4, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(evaluate_worker, suite, context, out / f"worker_{i}",
                               shard, expert, replicas, trace_ids) for i, shard in enumerate(shards)]
        rows = sorted((row for job in as_completed(futures) for row in job.result()), key=lambda r: r["case_id"])
    expected = {c["case_id"] for shard in shards for c in shard}
    if len(rows) != len(expected) or {r["case_id"] for r in rows} != expected:
        raise RuntimeError("Missing or duplicated evaluation results")
    if any(row["status"] not in STATUSES for row in rows):
        raise RuntimeError("An episode ended without a recognized outcome")
    routes = [route for route in spec["routes"] if any(r["route"] == route for r in rows)]
    result = read_json(model.parent / "result.json") if model and (model.parent / "result.json").is_file() else {}
    evaluated_epoch = (result.get("epochs_completed") if model and model.name == "last_model.pt"
                       else result.get("best_epoch"))
    summary = dict(label=label, expert=expert, checkpoint=str(model) if model else None,
                   checkpoint_sha256=hashlib.sha256(model.read_bytes()).hexdigest() if model else None,
                   evaluated_epoch=evaluated_epoch,
                   conditions=spec | {"replicas_per_worker": replicas}, overall=summarize(rows),
                   by_route={route: summarize([r for r in rows if r["route"] == route]) for route in routes},
                   elapsed_seconds=round(perf_counter() - started, 2))
    if trace_failures_from is not None:
        with (trace_failures_from / 'route_evaluation.csv').open(newline='') as file:
            previous = {int(row['case_id']): row for row in csv.DictReader(file)}
        mismatches = [dict(case_id=row['case_id'], original=previous[row['case_id']]['status'], replay=row['status'])
                      for row in rows if previous[row['case_id']]['status'] != row['status']]
        summary['audit_replay'] = dict(source=str(trace_failures_from), traced_cases=len(trace_ids),
                                      status_mismatches=mismatches)
    write_json(out / "evaluation_summary.json", summary)
    with (out / "route_evaluation.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plot_summary(out, label, rows, summary, routes, model.parent if model else None)
    print(f"{label}: {summary['overall']['statuses']} · {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("prepare")
    init.add_argument("suite", type=Path)
    init.add_argument("--reference", type=Path, required=True, help="Saved suite reference directory or BC results directory")
    init.add_argument("--traffic-vph", type=int, default=DEMAND)
    init.add_argument("--adaptive-traffic", action="store_true")
    init.add_argument("--uniform-idm-vph", type=int, default=0)
    init.add_argument("--traffic-seed", type=int, default=0,
                      help="Vary the warm-up traffic states and spawn streams (0 = original benchmark)")
    init.add_argument("--current-road", action="store_true",
                      help="Build the road from env/road_config.py instead of the reference's saved snapshot")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("suite", type=Path)
    evaluate.add_argument("label", nargs="?",
                          help="Optional BC evaluation suffix (evaluation_<label>); required for expert evaluation")
    target = evaluate.add_mutually_exclusive_group(required=True)
    target.add_argument("--expert", choices=EXPERTS)
    target.add_argument("--model", type=Path)
    evaluate.add_argument("--output", type=Path, help="Override the evaluation output directory")
    evaluate.add_argument("--replicas", type=int, default=MAX_REPLICAS_PER_WORKER,
                          help=f"Replicas per worker (default and maximum: {MAX_REPLICAS_PER_WORKER})")
    evaluate.add_argument("--smoke", type=int, default=0, help="Use only the first N cases per worker")
    evaluate.add_argument('--trace-failures-from', type=Path,
                          help='Replay every case in its original order, logging failed case steps from this result directory')
    args = parser.parse_args()
    suite = args.suite.resolve()
    if args.command == "prepare":
        if args.traffic_vph < 1:
            parser.error("--traffic-vph must be positive")
        if not 0 <= args.uniform_idm_vph < args.traffic_vph or (args.uniform_idm_vph and not args.adaptive_traffic):
            parser.error("--uniform-idm-vph requires adaptive traffic and must be below --traffic-vph")
        prepare(suite, args.reference.resolve(strict=True), args.traffic_vph, args.adaptive_traffic,
                args.uniform_idm_vph, traffic_seed=args.traffic_seed, current_road=args.current_road)
    else:
        if args.expert and args.label is None:
            parser.error("expert evaluation requires a label")
        if not 1 <= args.replicas <= MAX_REPLICAS_PER_WORKER or args.smoke < 0:
            parser.error(f"Expected 1–{MAX_REPLICAS_PER_WORKER} replicas and a nonnegative smoke-test count")
        run(suite, args.label, args.expert, args.model.resolve(strict=True) if args.model else None,
            args.replicas, args.smoke,
            args.trace_failures_from.resolve(strict=True) if args.trace_failures_from else None,
            output=args.output.resolve() if args.output else None)


if __name__ == "__main__":
    main()
