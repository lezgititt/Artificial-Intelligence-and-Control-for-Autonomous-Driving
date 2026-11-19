"""Prepare and evaluate a shared fixed-center, 23-route SUMO benchmark."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from env.background_demand import set_weighted_od_flow_rates
from env.collision_detection import configure_collision_gaps
from env.control_timing import lane_change_timing_allowed
from env.ego_routes import read_ego_route_groups
from env.parallel_sumo import ParallelSumo, libsumo
from env.parallel_traffic import TrafficCurriculum, TrafficSchedule
from env.traffic_snapshots import TrafficSnapshotBank
from scripts.evaluate import OUTCOMES, plt
from scripts.evaluate_quick_bc import plot_bc_training
from scripts.evaluate_quick_phase1 import STEP, summarize
from scripts.run_bc51_fixed_start import fixed_start_context
from test import load_saved_configs
from utils.evaluator import _DrivingStats, parallel_route_rows

DEMAND = 6000
TRAFFIC_SEED = 21701
MAX_REPLICAS_PER_WORKER = 64
SNAPSHOT_TIMES = tuple(range(802, 951, 2))
EXPERTS = ("car_idm", "car_krauss", "car_eidm", "car_acc")
STATUSES = ("arrival", "collision", "red_light", "wrong_lane", "timeout")


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(suite, reference, traffic_vph):
    context = suite / "reference"
    if (reference / "evaluation_config.json").is_file():
        shutil.copytree(reference, context)
        curriculum = TrafficCurriculum(**read_json(context / "evaluation_config.json")["traffic_curriculum"])
    else:
        _, curriculum, _ = fixed_start_context(reference, context, max_vph=traffic_vph)
    curriculum = replace(curriculum, max_vph=traffic_vph, snapshot_warmup_seconds=800)
    road, _, _ = load_saved_configs(str(context / "model.pt"))
    config = road_builder.build(road["ROAD"], road["EGO"], road["TRAFFIC"], str(suite / "sumo"))
    set_weighted_od_flow_rates(suite / "sumo/highway.rou.xml", traffic_vph)
    gaps = configure_collision_gaps(config, ego_gap_m=road["EGO"]["collision_gap_m"],
                                    background_gap_m=road["TRAFFIC"]["collision_gap_m"])
    routes = sum(read_ego_route_groups(suite / "sumo/highway.ego.rou.xml"), ())
    if len(routes) != 23:
        raise ValueError(f"Expected 23 routes, found {len(routes)}")
    entries = {r.get("id"): r.get("edges").split()[0]
               for r in ET.parse(suite / "sumo/highway.ego.rou.xml").iter("route")}
    lane_counts = {e.get("id"): len(e.findall("lane"))
                   for e in ET.parse(suite / "sumo/highway.net.xml").iter("edge")}
    bank = TrafficSnapshotBank(config, curriculum, road["TRAFFIC"]["vehs_per_hour"],
                               (traffic_vph,), STEP, 0, refresh=False,
                               capture_steps=[int(time / STEP) for time in SNAPSHOT_TIMES])
    try:
        destination = suite / "traffic_states.sqlite"
        with sqlite3.connect(bank.path) as source, sqlite3.connect(destination) as target:
            source.backup(target)
    finally:
        bank.close()
    cases = []
    with sqlite3.connect(destination) as db:
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
                                      seed=0, traffic_seed=[TRAFFIC_SEED, secs, 0], route=route,
                                      start_lane=lane_counts[entries[route]] // 2))
    if len({(c["snapshot_tick"], c["route"]) for c in cases}) != len(SNAPSHOT_TIMES) * len(routes):
        raise ValueError("Fixed case list must contain exactly one case per saved state and route")
    write_json(suite / "cases.json", cases)
    write_json(suite / "suite.json", dict(traffic_vph=traffic_vph, warmup_seconds=800, routes=routes,
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
            self.next_departure += self._headway(rng)


class FixedCaseSumo(ParallelSumo):
    def _respawn(self, ego, route=None, **kwargs):
        if not isinstance(ego.background, FixedTrafficSchedule):
            ego.background = FixedTrafficSchedule(self.traffic_flows, self.prototype.traffic["controllers"],
                                                 self.origin_lanes)
        obs = super()._respawn(ego, route, **kwargs)
        if self.expert_type:
            vehicle = libsumo.vehicle
            vehicle.setType(ego.ego_id, self.expert_type)
            vehicle.setSpeedMode(ego.ego_id, 31)
            vehicle.setSpeed(ego.ego_id, -1)
            return ego._get_obs()
        return obs

    def _advance(self, egos, actions):
        if not self.expert_type:
            yield from super()._advance(egos, actions)
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
            yield ego._finish_step(state, collided=vid in collided, arrived=vid in arrived)


def evaluate_worker(suite, context, scratch, cases, expert_type, replicas):
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
    try:
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
            transitions = list(env.step_evaluation(slots, actions))
            for slot, (x, reward, terminated, truncated, info) in zip(slots, transitions, strict=True):
                index = active[slot]
                records.record(index, reward, truncated, info)
                if not (terminated or truncated):
                    obs[slot] = x
                    continue
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
        return [case | row for case, row in zip(cases, rows, strict=True)]
    finally:
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


def run(suite, label, expert, model, replicas, smoke):
    started = perf_counter()
    reference = suite / "reference"
    spec = read_json(suite / "suite.json")
    cases = read_json(suite / "cases.json")
    out = suite / label
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
    print(f"{label}: {sum(map(len, shards)):,} cases, {replicas} replicas/worker", flush=True)
    with ProcessPoolExecutor(4, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(evaluate_worker, suite, context, out / f"worker_{i}",
                               shard, expert, replicas) for i, shard in enumerate(shards)]
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
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("suite", type=Path)
    evaluate.add_argument("label")
    target = evaluate.add_mutually_exclusive_group(required=True)
    target.add_argument("--expert", choices=EXPERTS)
    target.add_argument("--model", type=Path)
    evaluate.add_argument("--replicas", type=int, default=MAX_REPLICAS_PER_WORKER,
                          help=f"Replicas per worker (default and maximum: {MAX_REPLICAS_PER_WORKER})")
    evaluate.add_argument("--smoke", type=int, default=0, help="Use only the first N cases per worker")
    args = parser.parse_args()
    suite = args.suite.resolve()
    if args.command == "prepare":
        if args.traffic_vph < 1:
            parser.error("--traffic-vph must be positive")
        prepare(suite, args.reference.resolve(strict=True), args.traffic_vph)
    else:
        if not 1 <= args.replicas <= MAX_REPLICAS_PER_WORKER or args.smoke < 0:
            parser.error(f"Expected 1–{MAX_REPLICAS_PER_WORKER} replicas and a nonnegative smoke-test count")
        run(suite, args.label, args.expert, args.model.resolve(strict=True) if args.model else None,
            args.replicas, args.smoke)


if __name__ == "__main__":
    main()
