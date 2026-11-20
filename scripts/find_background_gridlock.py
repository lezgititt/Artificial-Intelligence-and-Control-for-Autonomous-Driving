"""Hunt for rare background-traffic gridlocks and report what blocks.

usage (repository root):
    python scripts/find_background_gridlock.py <suite_dir> <shares.json|-> [seconds=3600] [seeds=24] [first_seed=101]

"-" uses env.background_demand.FLOW_SHARES.

Runs background-only SUMO (Poisson arrivals, no teleport) for many seeds. When a
vehicle has been standing for more than 300 s, the run stops and prints every
long-waiting vehicle: where it is, whether its lane leads to its route, and why
it cannot change lanes.
"""
import xml.parsers.expat  # noqa: F401
import json, os, shutil, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import xml.etree.ElementTree as ET

LIMIT = 300.0


def hunt(args):
    sumo_dir, shares, seconds, seed = args
    import libsumo as api
    work = Path(tempfile.mkdtemp(prefix=f"hunt_{seed}_"))
    tree = ET.parse(Path(sumo_dir) / "highway.rou.xml")
    for flow in tree.getroot().findall("flow"):
        flow.set("period", f"exp({shares[flow.get('id')] / 3600:.6f})")
        flow.set("end", str(seconds))
    rou = work / "rou.xml"
    tree.write(rou, encoding="utf-8", xml_declaration=True)
    api.start(["sumo", "-n", str(Path(sumo_dir) / "highway.net.xml"), "-r", str(rou),
               "--step-length", "0.5", "--seed", str(seed), "--time-to-teleport", "-1",
               "--no-warnings", "true", "--no-step-log", "true", "--max-depart-delay", "100000"])
    report = None
    step = 0
    while api.simulation.getTime() < seconds:
        api.simulationStep(); step += 1
        if step % 20:
            continue
        ids = api.vehicle.getIDList()
        waits = {v: api.vehicle.getWaitingTime(v) for v in ids}
        if waits and max(waits.values()) > LIMIT:
            rows = []
            for v, w in sorted(waits.items(), key=lambda kv: -kv[1]):
                if w < 100:
                    continue
                lane = api.vehicle.getLaneID(v)
                route = api.vehicle.getRoute(v)
                index = api.vehicle.getRouteIndex(v)
                nxt = route[index + 1] if 0 <= index < len(route) - 1 else None
                follows = (lane.startswith(":") or nxt is None
                           or any(api.lane.getEdgeID(link[0]) == nxt for link in api.lane.getLinks(lane)))
                leader = api.vehicle.getLeader(v, 50.0)
                rows.append(dict(
                    flow=v.rsplit(".", 1)[0], wait=round(w), lane=lane,
                    remaining=round(api.lane.getLength(lane) - api.vehicle.getLanePosition(v), 1),
                    next=nxt, lane_ok=bool(follows),
                    leader=(leader[0].rsplit(".", 1)[0], round(leader[1], 1)) if leader and leader[0] else None,
                    lc_right=api.vehicle.getLaneChangeStatePretty(v, -1)[0],
                    lc_left=api.vehicle.getLaneChangeStatePretty(v, 1)[0]))
            report = dict(seed=seed, time=api.simulation.getTime(), vehicles=len(ids), stuck=rows)
            break
    api.close()
    shutil.rmtree(work, ignore_errors=True)
    return report or dict(seed=seed, time=None)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from env.background_demand import FLOW_SHARES
    suite = Path(sys.argv[1])
    shares = FLOW_SHARES if sys.argv[2] == "-" else json.loads(Path(sys.argv[2]).read_text())
    seconds = int(sys.argv[3]) if len(sys.argv) > 3 else 3600
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    first = int(sys.argv[5]) if len(sys.argv) > 5 else 101
    jobs = [(str(suite / "sumo"), shares, seconds, seed) for seed in range(first, first + n)]
    with ProcessPoolExecutor(6) as pool:
        results = list(pool.map(hunt, jobs))
    locked = [r for r in results if r["time"] is not None]
    print(f"{len(locked)} of {n} seeds gridlocked within {seconds} s (total {sum(shares.values()):.0f} vph)")
    for r in locked:
        print(f"\nseed {r['seed']} at t={r['time']:.0f}s, {r['vehicles']} vehicles in network; waiting > 100 s:")
        for s in r["stuck"][:14]:
            print(f"  {s['flow']:16s} wait={s['wait']:4d}s {s['lane']:16s} rem={s['remaining']:5.1f}m next={s['next']} "
                  f"lane_ok={s['lane_ok']} leader={s['leader']} lcR={s['lc_right']} lcL={s['lc_left']}")
