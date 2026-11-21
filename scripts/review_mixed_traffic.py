"""Re-run the mixed-traffic BC benchmark while logging how every vehicle drives.

Same traffic, seeds and outcome rules as scripts/evaluate_mixed_traffic.py. In addition it writes

  trips.csv   one row per finished trip (BC and background) with driving-quality measures
  events.json every BC trip that did not arrive: the last 20 s step by step, and for
              collisions who hit whom, on which lane, at which speeds

usage (repository root):
    python scripts/review_mixed_traffic.py <suite> --model <model.pt> --output <dir> [--seeds ...] [--seconds 3600]
"""
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_fixed_routes import FixedTrafficSchedule, read_json  # noqa: E402
from scripts.evaluate_mixed_traffic import (DEFAULT_SECONDS, DEFAULT_SEEDS, EGO_TYPE, MAX_DEPART_DELAY,  # noqa: E402
                                            MAX_TRIP_SECONDS, MIX, prepare_road)

HARD_BRAKE = 4.0        # m/s^2
HISTORY_STEPS = 40      # 20 s at 0.5 s steps


class Trip:
    """Driving-quality counters for one vehicle, updated once per step."""
    __slots__ = ("kind", "route", "start", "distance0", "last_speed", "last_lane", "last_edge", "hard_brakes",
                 "braking", "free_stop", "green_stop", "tailgate", "lane_changes", "reversals", "change_log",
                 "flips", "last_sign", "stopped", "steps")

    def __init__(self, kind, route, now, distance):
        self.kind, self.route, self.start, self.distance0 = kind, route, now, distance
        self.last_speed = self.last_lane = self.last_edge = None
        self.hard_brakes = self.free_stop = self.green_stop = self.tailgate = 0
        self.lane_changes = self.reversals = self.flips = self.stopped = self.steps = 0
        self.braking = False
        self.change_log = deque(maxlen=4)
        self.last_sign = 0

    def update(self, api, vid, now, step):
        speed = api.vehicle.getSpeed(vid)
        edge = api.vehicle.getRoadID(vid)
        lane = api.vehicle.getLaneIndex(vid)
        self.steps += 1
        if self.last_speed is not None:
            accel = (speed - self.last_speed) / step
            hard = accel <= -HARD_BRAKE
            self.hard_brakes += hard and not self.braking      # count events, not steps
            self.braking = hard
            sign = 1 if accel > 1.5 else (-1 if accel < -1.5 else 0)
            if sign and self.last_sign and sign != self.last_sign:
                self.flips += 1
            if sign:
                self.last_sign = sign
            if edge == self.last_edge and not edge.startswith(":") and lane != self.last_lane:
                self.lane_changes += 1
                direction = lane - self.last_lane
                if self.change_log and now - self.change_log[-1][0] <= 6.0 and self.change_log[-1][1] == -direction:
                    self.reversals += 1
                self.change_log.append((now, direction))
        if speed < 0.5:
            self.stopped += 1
            leader = api.vehicle.getLeader(vid, 30.0)
            gap = leader[1] if leader and leader[0] else None
            signals = api.vehicle.getNextTLS(vid)
            near = signals[0] if signals and signals[0][2] <= 40.0 else None
            if near is None and (gap is None or gap > 15.0):
                self.free_stop += 1                             # standing with nothing ahead
            elif near is not None and near[3].lower() == "g" and (gap is None or gap > 10.0):
                self.green_stop += 1                            # standing at a green light, road clear
        elif speed > 5.0:
            leader = api.vehicle.getLeader(vid, 5.0)
            if leader and leader[0] and leader[1] < 2.0:
                self.tailgate += 1
        self.last_speed, self.last_lane, self.last_edge = speed, lane, edge

    def row(self, api, vid, seed, now, status, step, distance):
        seconds = max(now - self.start, step)
        return dict(seed=seed, vehicle=vid, kind=self.kind, route=self.route, status=status,
                    start_s=self.start, seconds=round(seconds, 1), distance_m=round(distance - self.distance0, 1),
                    mean_speed=round((distance - self.distance0) / seconds, 2), hard_brakes=int(self.hard_brakes),
                    accel_flips=self.flips, lane_changes=self.lane_changes, lane_reversals=self.reversals,
                    stopped_s=self.stopped * step, free_stop_s=self.free_stop * step,
                    green_stop_s=self.green_stop * step, tailgate_s=self.tailgate * step)


def snapshot(api, vid, now, action=None):
    leader = api.vehicle.getLeader(vid, 60.0)
    signals = api.vehicle.getNextTLS(vid)
    x, y = api.vehicle.getPosition(vid)
    geometric = None
    if leader and leader[0]:
        # Bumper gap from lane geometry, independent of SUMO's reported leader gap.
        own_lane, lead_lane = api.vehicle.getLaneID(vid), api.vehicle.getLaneID(leader[0])
        lead_back = api.vehicle.getLanePosition(leader[0]) - api.vehicle.getLength(leader[0])
        if lead_lane == own_lane:
            geometric = lead_back - api.vehicle.getLanePosition(vid)
        elif any(lead_lane in (link[0], link[4]) for link in api.lane.getLinks(own_lane)):
            geometric = api.lane.getLength(own_lane) - api.vehicle.getLanePosition(vid) + lead_back
    return dict(t=now, lane=api.vehicle.getLaneID(vid), pos=round(api.vehicle.getLanePosition(vid), 1),
                x=round(x, 1), y=round(y, 1), speed=round(api.vehicle.getSpeed(vid), 2),
                accel_cmd=None if action is None else round(float(action[0]), 2),
                lane_cmd=None if action is None else int(round(float(action[1]))),
                leader=None if not leader or not leader[0] else dict(
                    geometric_gap=None if geometric is None else round(geometric, 2),
                    id=leader[0], type=api.vehicle.getTypeID(leader[0]), gap=round(leader[1], 1),
                    speed=round(api.vehicle.getSpeed(leader[0]), 2)),
                signal=None if not signals else dict(state=signals[0][3], distance=round(signals[0][2], 1)))


def run_road(cfg, flows, lanes, suite, model, vph, seconds, seed, capture=None, options=None):
    import torch
    from collect_background_data import libsumo as api
    if capture:
        import traci as api  # the GUI needs the TraCI client

    from collect_background_data import BackgroundObserver
    from scripts.run_bc51_dagger import load_actor
    from test import load_saved_configs

    torch.set_num_threads(1)
    road, _, _ = load_saved_configs(str(suite / "reference/model.pt"))
    _, mdp, train = load_saved_configs(str(model))
    actor = load_actor(model, train["HPARAMS"], "cpu").eval()
    gui_api = api

    class Observer(BackgroundObserver):
        @property
        def _api(self):
            return gui_api

    env = Observer(str(cfg), road["ROAD"], road["EGO"], mdp["SIMULATION"], mdp["OBSERVATION"],
                             mdp["ACTION"], train["REWARD"], traffic=road["TRAFFIC"])
    options = options or {}
    env.leader_gap_geometry = bool(options.get("leader_gap_geometry"))
    lane_shift = None
    if options.get("lane_prior"):
        # Training weighted the lane classes (sqrt inverse frequency), which moves the decision
        # boundary towards "change". Subtracting scale*log(weight) undoes that at inference.
        weights = read_json(model.parent / "config.json")["lane_weights"]
        lane_shift = torch.tensor(np.log(weights) * float(options["lane_prior"]), dtype=torch.float32)
    env.lc_safety_enabled = False
    env.lc_spawn_protect_only = True
    env.lc_spawn_protect = 30.0
    controllers = road["TRAFFIC"]["controllers"]
    mix = {name: dict(controllers[name], probability=share) if name != "ego"
           else dict(probability=share, speed_dev=0.0) for name, share in MIX.items()}
    schedule = FixedTrafficSchedule(flows, mix, lanes)
    schedule.types[-1] = EGO_TYPE
    from env.parallel_sumo import _signal_schedule
    traffic_seed, signal_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(traffic_seed)
    signal_ids, signal_ends = _signal_schedule(ET.parse(cfg.with_name("highway.net.xml")).getroot(),
                                              env.step_length, True)
    signal_tick = int(np.random.default_rng(signal_seed).integers(signal_ends[-1]))
    phase = int(np.searchsorted(signal_ends, signal_tick, side="right"))
    remaining = (signal_ends[phase] - signal_tick) * env.step_length
    step = env.step_length
    pending, active = {}, {}
    trips, history, rows, events = {}, {}, [], []
    command = ["sumo-gui" if capture else "sumo", "-c", str(cfg), "--step-length", str(step), "--seed", str(seed),
               "--no-step-log", "true", "--no-warnings", "true", "--collision.action", "remove",
               "--max-depart-delay", str(MAX_DEPART_DELAY)]
    if capture:
        command += ["--start", "--delay", "0", "--quit-on-end", "--window-size", "1100,800"]
        scenes = sorted(capture["scenes"], key=lambda scene: scene[1])   # (vehicle, time, label)
        seconds = min(seconds, scenes[-1][1] + 3)
        Path(capture["out"]).mkdir(parents=True, exist_ok=True)
    api.start(command)
    try:
        for signal in signal_ids:
            api.trafficlight.setPhase(signal, phase)
            api.trafficlight.setPhaseDuration(signal, remaining)
        schedule.reset(api, 0, vph, rng, set())
        for _ in range(round(seconds / step)):
            now = api.simulation.getTime()
            for vid, route, vehicle_type in schedule.spawn_due(api, now, str, rng):
                if vehicle_type == EGO_TYPE:
                    pending[vid] = dict(route=route, request_s=now)
            env._active_ids = set(api.vehicle.getIDList())
            before, commands = {}, {}
            if active:
                with torch.inference_mode():
                    obs = np.stack([env.observe_vehicle(vid) for vid in active])
                    if lane_shift is None:
                        actions = actor.det_action(torch.from_numpy(obs)).numpy()
                    else:
                        mu, _, logits, _ = actor.forward(torch.from_numpy(obs))
                        lane = torch.as_tensor(actor.LANE_CMDS)[torch.argmax(logits - lane_shift, dim=-1)]
                        actions = torch.cat([mu, lane.unsqueeze(-1)], dim=-1).numpy()
                for vid, action, row in zip(active, actions, obs, strict=True):
                    env.ego_id = vid
                    before[vid] = env._prepare_step(action)
                    commands[vid] = action
                    record = snapshot(api, vid, now, action)
                    # What the policy saw: own speed and the 5-lane neighbour block (left2..right2).
                    record["obs_neighbors"] = [round(float(x), 3) for x in row[:21]]
                    history[vid].append(record)

            # Last known state of every vehicle, needed after SUMO removes collided ones.
            previous = {vid: (api.vehicle.getTypeID(vid), api.vehicle.getLaneID(vid),
                              api.vehicle.getSpeed(vid), api.vehicle.getDistance(vid), api.vehicle.getRouteID(vid))
                        for vid in env._active_ids}
            if capture:
                for vid, when, label in scenes:
                    lead = when - now
                    if vid in env._active_ids and 0 <= lead <= capture.get("before", 10.0) and (
                            lead <= 2.0 or abs(lead - round(lead)) < 1e-6):
                        x, y = api.vehicle.getPosition(vid)
                        api.vehicle.setColor(vid, (255, 0, 255, 255))
                        api.gui.setZoom("View #0", capture.get("zoom", 1400))
                        api.gui.setOffset("View #0", x, y)
                        api.gui.screenshot("View #0", str(Path(capture["out"]) / f"{label}_{vid}_t{now:07.1f}.png"))
            api.simulationStep()
            now = api.simulation.getTime()
            live = set(api.vehicle.getIDList())
            env._active_ids = live
            collisions = api.simulation.getCollisions()
            collided = set(api.simulation.getCollidingVehiclesIDList())
            arrived = set(api.simulation.getArrivedIDList())
            teleported = set(api.simulation.getStartingTeleportIDList())
            for vid in api.simulation.getDepartedIDList():
                if vid in pending:
                    entry = pending.pop(vid)
                    entry["departure_s"] = now
                    active[vid] = entry
                    history[vid] = deque(maxlen=HISTORY_STEPS)
                    if vid in live:
                        api.vehicle.setSpeedMode(vid, 0)
                        api.vehicle.setLaneChangeMode(vid, 0)
                if vid in live:
                    kind = api.vehicle.getTypeID(vid)
                    trips[vid] = Trip("bc" if kind == EGO_TYPE else kind.removeprefix("car_"),
                                      api.vehicle.getRouteID(vid), now, api.vehicle.getDistance(vid))
            if not capture:
                for vid in live:
                    if vid in trips:
                        trips[vid].update(api, vid, now, step)

            def close(vid, status):
                trip = trips.pop(vid, None)
                if trip is not None and vid in previous:
                    rows.append(trip.row(api, vid, seed, now, status, step, previous[vid][3]))

            for vid, entry in list(active.items()):
                env.ego_id = vid
                status = None
                if vid in before and vid in live:
                    signals, _, distance = before[vid][6:9]
                    if signals:
                        traveled = api.vehicle.getDistance(vid) - distance
                        if any(traveled > stop + 1e-6 and env._crossing_signal_color(tls, link, color) == "r"
                               for tls, link, stop, color in signals):
                            status = "red_light"
                if status is None:
                    if vid in collided:
                        status = "collision"
                    elif vid in arrived:
                        status = "arrival"
                    elif vid in teleported:
                        status = "teleport"
                    elif vid not in live:
                        status = "removed"
                    elif env._stuck_in_wrong_lane():
                        status = "wrong_lane"
                    elif now - entry["departure_s"] >= MAX_TRIP_SECONDS:
                        status = "timeout"
                if status:
                    if status != "arrival":
                        event = dict(seed=seed, vehicle=vid, route=entry["route"], status=status, time_s=now,
                                     departure_s=entry["departure_s"], history=list(history[vid]))
                        for c in collisions:
                            if vid in (c.collider, c.victim):
                                other = c.victim if vid == c.collider else c.collider
                                event["collision"] = dict(
                                    role="collider" if vid == c.collider else "victim", type=c.type, lane=c.lane,
                                    pos=round(c.pos, 1), other=other,
                                    other_type=previous.get(other, ("?",))[0],
                                    other_lane=previous.get(other, ("?", "?"))[1],
                                    own_speed=round(c.colliderSpeed if vid == c.collider else c.victimSpeed, 2),
                                    other_speed=round(c.victimSpeed if vid == c.collider else c.colliderSpeed, 2))
                        events.append(event)
                    close(vid, status)
                    del active[vid], history[vid]
                    if vid in live or vid in teleported:
                        api.vehicle.remove(vid)
            for vid in list(trips):
                if vid not in live and vid not in active:
                    close(vid, "arrival" if vid in arrived else ("collision" if vid in collided else "removed"))
            for vid, entry in list(pending.items()):
                if now - entry["request_s"] > MAX_DEPART_DELAY + 2 * step:
                    del pending[vid]
            schedule.ids.clear()
    finally:
        api.close()
    print(f"Seed {seed}: BC {dict(Counter(r['status'] for r in rows if r['kind'] == 'bc'))}", flush=True)
    return rows, events


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vph", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seconds", type=int, default=DEFAULT_SECONDS)
    parser.add_argument("--capture", nargs="+", metavar="VEHICLE:TIME:LABEL",
                        help="Replay the first seed in SUMO GUI and save screenshots following each vehicle "
                             "during the 10 s before TIME (every 1 s, then every step for the last 2 s)")
    parser.add_argument("--zoom", type=float, default=1400)
    parser.add_argument("--leader-gap-geometry", action="store_true",
                        help="Measure the front gap from lane geometry when SUMO clamps it at 0")
    parser.add_argument("--lane-prior", type=float, default=0.0,
                        help="Subtract this multiple of log(training lane class weight) from the lane logits "
                             "(1.0 removes the class weighting; 0 keeps the trained decision)")
    args = parser.parse_args()
    suite, model = args.suite.resolve(strict=True), args.model.resolve(strict=True)
    spec = read_json(suite / "suite.json")
    vph = spec["traffic_vph"] if args.vph is None else args.vph
    out = args.output.resolve()
    if args.capture:
        from tempfile import TemporaryDirectory
        scenes = [(v, float(t), label) for v, t, label in (item.split(":") for item in args.capture)]
        with TemporaryDirectory(prefix="sumo-review-") as work:
            cfg, flows, lanes = prepare_road(suite, Path(work), spec)
            run_road(cfg, flows, lanes, suite, model, vph, args.seconds, args.seeds[0],
                     capture=dict(out=out, scenes=scenes, zoom=args.zoom))
        print(f"Screenshots: {out}", flush=True)
        return
    out.mkdir(parents=True)
    cfg, flows, lanes = prepare_road(suite, out, spec)
    rows, events = [], []
    with ProcessPoolExecutor(min(args.workers, len(args.seeds)), mp_context=get_context("spawn")) as pool:
        options = dict(leader_gap_geometry=args.leader_gap_geometry, lane_prior=args.lane_prior)
        for job in [pool.submit(run_road, cfg, flows, lanes, suite, model, vph, args.seconds, seed, None, options)
                    for seed in args.seeds]:
            r, e = job.result()
            rows.extend(r)
            events.extend(e)
    with (out / "trips.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (out / "events.json").write_text(json.dumps(events, indent=1), encoding="utf-8")
    bc = Counter(r["status"] for r in rows if r["kind"] == "bc")
    print(f"BC trips: {dict(bc)} | events: {len(events)} | {out}", flush=True)


if __name__ == "__main__":
    main()
