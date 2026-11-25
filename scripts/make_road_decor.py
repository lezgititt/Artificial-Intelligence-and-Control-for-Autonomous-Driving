"""Write map-style decoration (solid background, grass, roadside trees) for sumo-gui.

The output is a separate SUMO additional file (``highway.decor.xml``) made only of
polygons. It is loaded by the GUI playback tools and never by training or headless
evaluation, so it cannot change simulation results.

    python scripts/make_road_decor.py env/sumo/highway.net.xml
    sumo-gui -c env/sumo/highway.sumocfg -a env/sumo/highway.scenery.xml,env/sumo/highway.decor.xml

Geometry is computed on a 0.5 m grid: the distance from every cell to the nearest
road surface gives the grass band (closed loops of road are filled, so the empty
land inside the intersection becomes grass) and the line where trees are planted.
"""
import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

CELL = 0.5               # grid resolution (m)
GRASS_WIDTH = 14.0       # grass band beside the road (m)
TREE_OFFSET = 5.0        # distance from road edge to tree centers (m)
TREE_SPACING = 14.0      # distance between neighboring trees (m)
TREE_RADIUS = 2.3        # crown radius (m)
DEAD_END_CLEARANCE = 12.0  # no trees in front of the open road ends
BACKGROUND_EXTENT = 5000.0  # half-size of the background sheet (m)

BACKGROUND_COLOR = "238,236,229"   # light warm gray, like a paper map
GRASS_COLOR = "168,196,150"        # same green as the existing traffic islands
TREE_COLOR = "86,140,82"
TREE_TOP_COLOR = "112,163,100"

# Lanes are drawn at layer 0; negative layers stay underneath the road.
BACKGROUND_LAYER = -5
GRASS_LAYER = -4
TREE_LAYER = -0.5


def _road_pieces(net):
    """Yield (polyline, half width) for every lane and junction outline."""
    for edge in net.getEdges(withInternal=True):
        for lane in edge.getLanes():
            yield np.asarray(lane.getShape(), float), lane.getWidth() / 2
    for node in net.getNodes():
        shape = node.getShape()
        if len(shape) >= 3:
            yield np.asarray(list(shape) + [shape[0]], float), 0.0


def _distance_field(net, margin):
    """Signed-ish distance (m) from each grid cell to the road surface (<= 0 on the road)."""
    from matplotlib.path import Path as MplPath

    xmin, ymin, xmax, ymax = net.getBoundary()
    xs = np.arange(xmin - margin, xmax + margin + CELL, CELL)
    ys = np.arange(ymin - margin, ymax + margin + CELL, CELL)
    field = np.full((len(ys), len(xs)), np.inf)
    reach = GRASS_WIDTH + 2.0
    for line, half in _road_pieces(net):
        for (x0, y0), (x1, y1) in zip(line[:-1], line[1:]):
            lo_x, hi_x = min(x0, x1) - half - reach, max(x0, x1) + half + reach
            lo_y, hi_y = min(y0, y1) - half - reach, max(y0, y1) + half + reach
            i0, i1 = np.searchsorted(xs, [lo_x, hi_x])
            j0, j1 = np.searchsorted(ys, [lo_y, hi_y])
            if i0 >= i1 or j0 >= j1:
                continue
            gx, gy = np.meshgrid(xs[i0:i1], ys[j0:j1])
            dx, dy = x1 - x0, y1 - y0
            length2 = dx * dx + dy * dy
            t = 0.0 if length2 == 0 else np.clip(((gx - x0) * dx + (gy - y0) * dy) / length2, 0, 1)
            dist = np.hypot(gx - (x0 + t * dx), gy - (y0 + t * dy)) - half
            np.minimum(field[j0:j1, i0:i1], dist, out=field[j0:j1, i0:i1])
    gx, gy = np.meshgrid(xs, ys)
    cells = np.column_stack([gx.ravel(), gy.ravel()])
    for node in net.getNodes():
        shape = node.getShape()
        if len(shape) >= 3:
            inside = MplPath(shape).contains_points(cells).reshape(field.shape)
            field[inside] = np.minimum(field[inside], 0.0)
    return xs, ys, np.minimum(field, 1e3)


def _simplify(points, tolerance):
    """Douglas-Peucker line simplification."""
    points = np.asarray(points, float)
    if len(points) < 3:
        return points
    keep = np.zeros(len(points), bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        p, q = points[a], points[b]
        seg = q - p
        rel = points[a + 1:b] - p
        norm = math.hypot(*seg)
        dist = (np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / norm if norm
                else np.hypot(rel[:, 0], rel[:, 1]))
        k = int(np.argmax(dist))
        if dist[k] > tolerance:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return points[keep]


def _fmt(points):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _circle(cx, cy, r, n=14):
    return [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n))
            for k in range(n)]


def _grass_outlines(xs, ys, field):
    """Outer rings of the grass band; holes (land enclosed by roads) are filled."""
    from contourpy import FillType, contour_generator

    gen = contour_generator(xs, ys, field, fill_type=FillType.OuterOffset)
    points, offsets = gen.filled(-np.inf, GRASS_WIDTH)
    outlines = []
    for pts, off in zip(points, offsets):
        outer = pts[off[0]:off[1]]
        if len(outer) >= 3:
            outlines.append(_simplify(outer, CELL * 0.6))
    return outlines


def _tree_spots(net, xs, ys, field):
    """Points on the TREE_OFFSET line around the road, TREE_SPACING apart."""
    from contourpy import contour_generator

    dead_ends = np.array([node.getCoord()[:2] for node in net.getNodes()
                          if node.getType() == "dead_end"], float).reshape(-1, 2)
    spots = []
    for line in contour_generator(xs, ys, field).lines(TREE_OFFSET):
        seg = np.hypot(*np.diff(line, axis=0).T)
        along = np.concatenate([[0], np.cumsum(seg)])
        if along[-1] < TREE_SPACING:
            continue
        for s in np.arange(TREE_SPACING / 2, along[-1], TREE_SPACING):
            x = np.interp(s, along, line[:, 0])
            y = np.interp(s, along, line[:, 1])
            if len(dead_ends) and np.min(np.hypot(*(dead_ends - (x, y)).T)) < DEAD_END_CLEARANCE:
                continue
            if all(math.hypot(x - u, y - v) >= 0.8 * TREE_SPACING for u, v in spots):
                spots.append((x, y))
    return spots


def write_decor(net_path, out_path=None):
    """Write the decoration file next to the network (or to out_path) and return its path."""
    import sumolib

    net_path = Path(net_path)
    out_path = Path(out_path) if out_path else net_path.with_name("highway.decor.xml")
    net = sumolib.net.readNet(str(net_path), withInternal=True)
    xs, ys, field = _distance_field(net, GRASS_WIDTH + 4.0)
    xmin, ymin, xmax, ymax = net.getBoundary()
    cx, cy, e = (xmin + xmax) / 2, (ymin + ymax) / 2, BACKGROUND_EXTENT

    lines = ["<additional>",
             "    <!-- Generated by scripts/make_road_decor.py; GUI decoration only. -->",
             f'    <poly id="decor_background" color="{BACKGROUND_COLOR}" fill="1" '
             f'layer="{BACKGROUND_LAYER}" shape="{_fmt([(cx - e, cy - e), (cx + e, cy - e), (cx + e, cy + e), (cx - e, cy + e)])}"/>']
    for k, outline in enumerate(_grass_outlines(xs, ys, field)):
        lines.append(f'    <poly id="decor_grass_{k}" color="{GRASS_COLOR}" fill="1" '
                     f'layer="{GRASS_LAYER}" shape="{_fmt(outline)}"/>')
    for k, (x, y) in enumerate(_tree_spots(net, xs, ys, field)):
        lines.append(f'    <poly id="decor_tree_{k}" color="{TREE_COLOR}" fill="1" '
                     f'layer="{TREE_LAYER}" shape="{_fmt(_circle(x, y, TREE_RADIUS))}"/>')
        lines.append(f'    <poly id="decor_tree_{k}_top" color="{TREE_TOP_COLOR}" fill="1" '
                     f'layer="{TREE_LAYER + 0.1:.1f}" '
                     f'shape="{_fmt(_circle(x - 0.5, y + 0.5, TREE_RADIUS * 0.6, 10))}"/>')
    lines.append("</additional>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def gui_decor_options(sumocfg):
    """sumo-gui options that add the decoration to the additional files of sumocfg.

    The decoration is regenerated from the configuration's own network, so it fits
    whichever road (current road, evaluation suite copy, preview) is being shown.
    """
    sumocfg = Path(sumocfg).resolve()
    root = ET.parse(sumocfg).getroot()
    net_node = root.find("input/net-file")
    add_node = root.find("input/additional-files")
    net = sumocfg.parent / net_node.get("value")
    additional = [] if add_node is None else [
        str(sumocfg.parent / name.strip()) for name in add_node.get("value").split(",") if name.strip()]
    decor = write_decor(net, sumocfg.with_name("highway.decor.xml"))
    return ["--additional-files", ",".join(additional + [str(decor)])]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("net", type=Path, help="SUMO network (highway.net.xml)")
    parser.add_argument("-o", "--output", type=Path, help="Output file (default: highway.decor.xml beside the network)")
    args = parser.parse_args()
    print(f"Wrote {write_decor(args.net, args.output)}")


if __name__ == "__main__":
    main()
