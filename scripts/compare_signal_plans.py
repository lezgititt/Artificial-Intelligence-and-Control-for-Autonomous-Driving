"""Compare signal plans for the central junction (nC) with background traffic only.

The road file is not changed: each plan is written as a separate SUMO additional
file holding a new program for traffic light nC, and SUMO uses the program loaded
last. The same calibrated demand is then simulated under every plan.

    python scripts/compare_signal_plans.py analysis/yj_bc_suite_4740/sumo

A plan is a list of stages. A stage names the lanes that get green, as
(incoming edge, lane indices); None means every lane of that edge. Lanes that are
green in two consecutive stages stay green through the yellow and all-red between
them. U-turns only get a permissive green (they yield), as in road_builder.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import itertools
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

TLS = "nC"
YELLOW, ALL_RED = 3, 2
# U-turns made at the node before the junction; controlled by the same program.
UPSTREAM_UTURNS = {"A2": "A2_in1", "A5": "A5_in1"}

# Clock positions: A1 = 8, A2 = 2, A3 = 5, A4 = 10, A5 = 3 o'clock.
SEQUENTIAL = [
    ("A1", [("A1_in", None)]),
    ("A2", [("A2_in", None), ("A2_in1", None)]),
    ("A3", [("A3_in", None)]),
    ("A4", [("A4_in", None)]),
    ("A5", [("A5_in", None), ("A5_in1", None)]),
]
# Nohyeong-ro (A1/A2) runs both ways at once, then its left turns together.
# The right-turn lanes of A3 and A1 keep green wherever nothing crosses them.
A3_RIGHT, A1_RIGHT = ("A3_in", [0]), ("A1_in", [0])
GROUPED = [
    ("A1A2_through", [("A1_in", [0, 1, 2, 3]), ("A2_in", [1, 2, 3]), A3_RIGHT]),
    ("A1A2_left", [("A1_in", [4]), ("A2_in", [4]), ("A2_in1", None), A3_RIGHT, A1_RIGHT]),
    ("A3", [("A3_in", None)]),
    ("A4", [("A4_in", None)]),
    ("A5", [("A5_in", None), ("A5_in1", None), A1_RIGHT]),
]
PLANS = {
    "current": (SEQUENTIAL, [20, 20, 20, 20, 20]),
    "current_weighted": (SEQUENTIAL, [25, 17, 23, 19, 16]),
    "grouped": (GROUPED, [20, 10, 20, 20, 20]),
    "grouped_weighted": (GROUPED, [23, 12, 21, 18, 16]),
    "current_short": (SEQUENTIAL, [15, 15, 15, 15, 15]),
    "grouped_short": (GROUPED, [15, 8, 15, 15, 15]),
}


def junction_links(net_path):
    """Signal links of the junction: index -> connection attributes, with the full path through it."""
    root = ET.parse(net_path).getroot()
    shape = {lane.get("id"): [tuple(map(float, p.split(","))) for p in lane.get("shape").split()]
             for lane in root.iter("lane")}
    following = {f'{c.get("from")}_{c.get("fromLane")}': c.get("via") for c in root.findall("connection")
                 if c.get("from").startswith(":") and c.get("via")}
    links = {}
    for c in root.findall("connection"):
        if c.get("tl") != TLS:
            continue
        path, lane = [], c.get("via")
        while lane:
            path += shape[lane]
            lane = following.get(lane)
        links[int(c.get("linkIndex"))] = dict(c.attrib, path=path)
    return links


def stage_links(links, lanes):
    chosen = set()
    for edge, indices in lanes:
        found = {i for i, c in links.items() if c["from"] == edge
                 and (indices is None or int(c["fromLane"]) in indices)}
        if not found:
            raise ValueError(f"No signal link for {edge} lanes {indices}")
        chosen |= found
    return chosen


def green_state(links, green):
    """'G' for protected green; 'g' (yield) for U-turns and a second stream into the same lane."""
    state, taken = ["r"] * (max(links) + 1), set()
    for i in sorted(green):
        c = links[i]
        target = (c["to"], c["toLane"])
        yields = c["dir"] == "t" or c["from"] in UPSTREAM_UTURNS.values() or target in taken
        state[i] = "g" if yields else "G"
        if not yields:
            taken.add(target)
    return state


def program(links, stages, greens):
    """SUMO tlLogic for the plan: green, yellow and all-red for each stage."""
    sets = [stage_links(links, lanes) for _, lanes in stages]
    logic = ET.Element("tlLogic", id=TLS, type="static", programID="compare", offset="0")
    for k, ((name, _), seconds) in enumerate(zip(stages, greens, strict=True)):
        state = green_state(links, sets[k])
        ending = sets[k] - sets[(k + 1) % len(sets)]
        ET.SubElement(logic, "phase", duration=str(seconds), state="".join(state), name=f"{name}_green")
        ET.SubElement(logic, "phase", duration=str(YELLOW), name=f"{name}_yellow",
                      state="".join("y" if i in ending else s for i, s in enumerate(state)))
        ET.SubElement(logic, "phase", duration=str(ALL_RED), name=f"{name}_clearance",
                      state="".join("r" if i in ending else s for i, s in enumerate(state)))
    return logic, sets


def _cross(p, q):
    def side(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return any(side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0
               for a, b in zip(p[:-1], p[1:]) for c, d in zip(q[:-1], q[1:]))


def conflicts(links, green):
    """Pairs of protected greens whose paths cross, merge into one lane, or feed a U-turn's exit."""
    state = green_state(links, green)
    found = []
    for i, j in itertools.combinations(sorted(green), 2):
        a, b = links[i], links[j]
        if a["from"] == b["from"] or "g" in (state[i], state[j]):
            continue
        if _cross(a["path"], b["path"]) or (a["to"], a["toLane"]) == (b["to"], b["toLane"]):
            found.append((i, j))
    for i in green:   # a U-turn must not run while the junction feeds the road it turns into
        arm = next((arm for arm, edge in UPSTREAM_UTURNS.items() if links[i]["from"] == edge), None)
        found += [(i, j) for j in green if arm and links[j]["to"] == f"{arm}_out"]
    return found


def simulate(sumo_dir, plan_file, seed, seconds, out):
    cfg = sumo_dir / "highway.sumocfg"
    root = ET.parse(cfg).getroot()
    node = root.find("input/additional-files")
    additional = [str(sumo_dir / name) for name in (node.get("value").split(",") if node is not None else [])]
    stats = out / f"{plan_file.stem}_{seed}.xml"
    subprocess.run(
        ["sumo", "-c", str(cfg), "--additional-files", ",".join(additional + [str(plan_file)]),
         "--step-length", "0.5", "--seed", str(seed), "--end", str(seconds),
         "--max-depart-delay", "125", "--collision.action", "warn", "--collision.check-junctions", "true",
         "--no-step-log", "true", "--no-warnings", "true", "--duration-log.statistics", "true",
         "--statistic-output", str(stats)], check=True, capture_output=True)
    s = ET.parse(stats).getroot()
    trips = s.find("vehicleTripStatistics")
    return dict(inserted=int(s.find("vehicles").get("inserted")), arrived=int(trips.get("count")),
                wait=float(trips.get("waitingTime")), loss=float(trips.get("timeLoss")),
                collisions=int(s.find("safety").get("collisions")),
                teleports=int(s.find("teleports").get("total")))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sumo_dir", type=Path, help="Folder with highway.sumocfg, network and background flows")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--keep", type=Path, help="Folder to keep the plan files in (default: temporary)")
    args = parser.parse_args()
    sumo_dir = args.sumo_dir.resolve(strict=True)
    links = junction_links(sumo_dir / "highway.net.xml")

    with tempfile.TemporaryDirectory(prefix="signal-plans-") as work:
        out = args.keep.resolve() if args.keep else Path(work)
        out.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, (stages, greens) in PLANS.items():
            logic, sets = program(links, stages, greens)
            for (stage, _), green in zip(stages, sets):
                for i, j in conflicts(links, green):
                    a, b = links[i], links[j]
                    raise ValueError(f"{name}/{stage}: {a['from']}>{a['to']} conflicts with {b['from']}>{b['to']}")
            ET.indent(logic, space="    ")
            wrapper = ET.Element("additional")
            wrapper.append(logic)
            files[name] = out / f"{name}.add.xml"
            ET.ElementTree(wrapper).write(files[name], encoding="utf-8")
            cycle = sum(greens) + len(greens) * (YELLOW + ALL_RED)
            print(f"{name}: green {greens}, cycle {cycle} s")

        jobs = [(name, seed) for name in PLANS for seed in args.seeds]
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda job: simulate(sumo_dir, files[job[0]], job[1], args.seconds, out), jobs))
        by_plan = defaultdict(list)
        for (name, _), result in zip(jobs, results):
            by_plan[name].append(result)
        print(f"\nmean of {len(args.seeds)} seeds, {args.seconds} s each")
        print(f"{'plan':18s} {'inserted':>9s} {'arrived':>8s} {'wait s':>7s} {'loss s':>7s} {'collisions':>10s} {'teleports':>9s}")
        for name, rows in by_plan.items():
            mean = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
            print(f"{name:18s} {mean['inserted']:9.0f} {mean['arrived']:8.0f} {mean['wait']:7.1f} "
                  f"{mean['loss']:7.1f} {mean['collisions']:10.1f} {mean['teleports']:9.1f}")


if __name__ == "__main__":
    main()
