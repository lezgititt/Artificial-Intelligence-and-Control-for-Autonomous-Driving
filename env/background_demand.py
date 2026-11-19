"""Background demand with fixed U-turn and ring probabilities."""
import xml.etree.ElementTree as ET

import numpy as np


def weighted_od_flows(root):
    """Sample route variants with a reduced U-turn share and increased ring share."""
    routes = {route.get("id"): route.get("edges").split() for route in root.findall("route")}
    by_pair = {}
    for flow in root.findall("flow"):
        if flow.get("route"):
            edges = routes[flow.get("route")]
            pair = (edges[0], edges[-1])
            variant = ("vehicle", {"route": flow.get("route")})
        else:
            pair = (flow.get("from"), flow.get("to"))
            variant = ("trip", {key: flow.get(key) for key in ("from", "to", "via") if key in flow.attrib})
        by_pair.setdefault(pair, []).append((*variant, flow))

    variants = []
    for (origin, destination), paths in by_pair.items():
        u_turn = origin.split("_")[0] == destination.split("_")[0]
        for kind, route, flow in paths:
            edges = routes[route["route"]] if kind == "vehicle" else route.get("via", "").split()
            ring = any(edge.startswith("RING_") for edge in edges)
            variants.append((kind, route, u_turn, ring, 1 / (len(by_pair) * len(paths)), flow))

    base = np.array([row[4] for row in variants])
    u_turn = np.array([row[2] for row in variants])
    ring = np.array([row[3] for row in variants])
    ordinary = ~(u_turn | ring)
    probabilities = base * np.where(u_turn, 0.5, 1.0) * np.where(ring, 2.0, 1.0)
    if ordinary.any():
        probabilities[ordinary] *= (1 - probabilities[~ordinary].sum()) / base[ordinary].sum()
    previous_u_turn_share = probabilities[u_turn].sum()
    probabilities[u_turn] = base[u_turn] * len(by_pair) * 0.0125
    released_share = previous_u_turn_share - probabilities[u_turn].sum()
    ring_share = probabilities[ring].sum()
    probabilities[ring] *= (ring_share + released_share) / ring_share
    return variants, probabilities, len(by_pair), probabilities[u_turn].sum(), probabilities[ring].sum()


def set_weighted_od_flow_rates(route_file, traffic_vph):
    tree = ET.parse(route_file)
    variants, probabilities, *_ = weighted_od_flows(tree.getroot())
    for row, probability in zip(variants, probabilities, strict=True):
        row[-1].set("period", f"{3600 / (traffic_vph * probability):.12f}")
    tree.write(route_file, encoding="utf-8", xml_declaration=True)


def sample_background_demand(route_file, traffic_vph, duration, seed, *, output=None):
    """Generate independent route draws at a fixed total request rate."""
    tree = ET.parse(route_file)
    root = tree.getroot()
    variants, probabilities, pairs, u_turn_share, ring_share = weighted_od_flows(root)
    for flow in root.findall("flow"):
        root.remove(flow)
    rng = np.random.default_rng(seed)
    count = int(traffic_vph * duration / 3600)
    for i, variant_index in enumerate(rng.choice(len(variants), size=count, p=probabilities)):
        kind, route, *_ = variants[variant_index]
        ET.SubElement(root, kind, dict(id=f"background_{i}", type="carDist",
                                      depart=f"{i * 3600 / traffic_vph:.6f}",
                                      departLane="random", departSpeed="8.0", **route))
    tree.write(output if output is not None else route_file, encoding="utf-8", xml_declaration=True)
    return pairs, u_turn_share, ring_share, count
