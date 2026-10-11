"""Original two-junction geometry and reproducible, genuinely connected routes."""

from __future__ import annotations

import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from fair_signal.demand import (
    DIRECTIONS,
    EXITS,
    GREEN_STATES,
    LINK_BASE,
    save_json,
    sha256,
    write_xml,
)
from fair_signal.runtime import configure_sumo

JUNCTIONS = ("a", "b")
NEIGHBORS = {
    "a": {"n": "an", "s": "as", "e": "b", "w": "w"},
    "b": {"n": "bn", "s": "bs", "e": "e", "w": "a"},
}
ORIGINS = ("w", "e", "an", "as", "bn", "bs")
ORIGIN_JUNCTION = {"w": "a", "e": "b", "an": "a", "as": "a", "bn": "b", "bs": "b"}
SCENARIOS = ("steady", "tidal", "downstream_slowdown")
INCOMING = {
    j: tuple(f"{NEIGHBORS[j][d]}_{j}_{lane}" for d in DIRECTIONS for lane in (0, 1))
    for j in JUNCTIONS
}
OUTGOING = {
    j: tuple(f"{j}_{NEIGHBORS[j][d]}_{lane}" for d in DIRECTIONS for lane in (0, 1))
    for j in JUNCTIONS
}
ALL_LANES = tuple(sorted(set(sum(INCOMING.values(), ()) + sum(OUTGOING.values(), ()))))
CONNECTOR_LANES = ("a_b_0", "a_b_1", "b_a_0", "b_a_1")


def build_network(directory: Path) -> Path:
    configure_sumo()
    directory.mkdir(parents=True, exist_ok=True)
    network = directory / "arterial.net.xml"
    if network.exists():
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["sha256"] != sha256(network):
            raise ValueError("Arterial network checksum mismatch")
        return network
    nodes, edges, connections, logic = (
        ET.Element(n) for n in ("nodes", "edges", "connections", "tlLogics")
    )
    positions = {
        "a": (0, 0),
        "b": (350, 0),
        "w": (-300, 0),
        "e": (650, 0),
        "an": (0, 300),
        "as": (0, -300),
        "bn": (350, 300),
        "bs": (350, -300),
    }
    for name, (x, y) in positions.items():
        ET.SubElement(
            nodes,
            "node",
            id=name,
            x=str(x),
            y=str(y),
            type="traffic_light" if name in JUNCTIONS else "priority",
        )
    pairs = {(j, n) for j in JUNCTIONS for n in NEIGHBORS[j].values()}
    pairs |= {(v, u) for u, v in pairs}
    for u, v in sorted(pairs):
        ET.SubElement(
            edges,
            "edge",
            id=f"{u}_{v}",
            **{"from": u, "to": v, "numLanes": "2", "speed": "13.89", "priority": "1"},
        )
    for j in JUNCTIONS:
        for d in DIRECTIONS:
            for movement, dest in enumerate(EXITS[d]):
                lane = 1 if movement == 2 else 0
                ET.SubElement(
                    connections,
                    "connection",
                    **{
                        "from": f"{NEIGHBORS[j][d]}_{j}",
                        "to": f"{j}_{NEIGHBORS[j][dest]}",
                        "fromLane": str(lane),
                        "toLane": str(lane),
                        "tl": j,
                        "linkIndex": str(LINK_BASE[d] + movement),
                    },
                )
        tls = ET.SubElement(logic, "tlLogic", id=j, type="static", programID="0", offset="0")
        for green in GREEN_STATES:
            for state, duration in ((green, 30), (green.replace("G", "y"), 3), ("r" * 12, 1)):
                ET.SubElement(tls, "phase", duration=str(duration), state=state)
    for root, name in (
        (nodes, "nodes.nod.xml"),
        (edges, "edges.edg.xml"),
        (connections, "connections.con.xml"),
        (logic, "signals.tll.xml"),
    ):
        write_xml(root, directory / name)
    subprocess.run(
        [
            shutil.which("netconvert") or "netconvert",
            "--node-files",
            str(directory / "nodes.nod.xml"),
            "--edge-files",
            str(directory / "edges.edg.xml"),
            "--connection-files",
            str(directory / "connections.con.xml"),
            "--tllogic-files",
            str(directory / "signals.tll.xml"),
            "--output-file",
            str(network),
            "--no-turnarounds",
            "true",
            "--no-warnings",
            "true",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    root = ET.parse(network).getroot()
    write_xml(root, network)
    for j in JUNCTIONS:
        tls = root.find(f"tlLogic[@id='{j}']")
        if tls is None or [p.get("state") for p in tls.findall("phase")[::3]] != list(GREEN_STATES):
            raise RuntimeError("Arterial phase ordering mismatch")
        actual = {
            (c.get("from"), c.get("to")): int(c.get("linkIndex"))
            for c in root.findall("connection")
            if c.get("tl") == j
        }
        expected = {
            (f"{NEIGHBORS[j][d]}_{j}", f"{j}_{NEIGHBORS[j][dest]}"): LINK_BASE[d] + m
            for d in DIRECTIONS
            for m, dest in enumerate(EXITS[d])
        }
        if actual != expected:
            raise RuntimeError("Arterial controlled link ordering mismatch")
    save_json(
        directory / "manifest.json",
        {
            "sha256": sha256(network),
            "sumo": "1.28.0",
            "junction_spacing_m": 350,
            "boundary_arm_m": 300,
            "original_generator": True,
        },
    )
    return network


def rates_for(scenario: str, split: str, rng: np.random.Generator) -> np.ndarray:
    if split not in ("train", "tuning", "validation", "test"):
        raise ValueError("Unknown split")
    if scenario == "steady":
        rates = np.tile([600, 600, 120, 120, 120, 120], (3, 1)).astype(float)
    elif scenario == "tidal":
        rates = np.array(
            [
                [850, 300, 130, 130, 130, 130],
                [300, 850, 130, 130, 130, 130],
                [600, 600, 130, 130, 130, 130],
            ],
            dtype=float,
        )
    elif scenario == "downstream_slowdown":
        rates = np.array(
            [
                [650, 650, 140, 140, 140, 140],
                [650, 650, 140, 140, 400, 400],
                [650, 650, 140, 140, 140, 140],
            ],
            dtype=float,
        )
    else:
        raise ValueError("Unknown scenario")
    if split == "train":
        rates *= rng.uniform(0.8, 1.2, size=(1, 6))
    return rates * {"train": 1.0, "tuning": 1.0, "validation": 1.05, "test": 1.12}[split]


def route_edges(origin: str, movements: tuple[int, int]) -> list[str]:
    junction = ORIGIN_JUNCTION[origin]
    previous = origin
    route = [f"{previous}_{junction}"]
    for movement in movements:
        direction = next(d for d, node in NEIGHBORS[junction].items() if node == previous)
        destination = NEIGHBORS[junction][EXITS[direction][movement]]
        route.append(f"{junction}_{destination}")
        if destination not in JUNCTIONS:
            return route
        previous, junction = junction, destination
    raise RuntimeError("Two non-U-turn movements must exit the two-node arterial")


def generate_routes(directory: Path, scenario: str, split: str, seed: int, seconds: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rates = rates_for(scenario, split, rng)
    root = ET.Element("routes")
    ET.SubElement(
        root,
        "vType",
        id="car",
        accel="2.6",
        decel="4.5",
        sigma="0.5",
        length="5",
        minGap="2.5",
        maxSpeed="13.89",
        speedFactor="1.0",
        speedDev="0.1",
    )
    vehicles = []
    for third in range(3):
        start, end = third * seconds / 3, (third + 1) * seconds / 3
        for i, origin in enumerate(ORIGINS):
            t = start
            while True:
                t += rng.exponential(3600 / rates[third, i])
                if t >= end:
                    break
                movements = tuple(int(x) for x in rng.choice(3, size=2, p=[0.1, 0.7, 0.2]))
                vehicles.append((t, origin, route_edges(origin, movements)))
    vehicles.sort(key=lambda v: v[0])
    traversals = 0
    for i, (t, origin, edges) in enumerate(vehicles):
        vehicle = ET.SubElement(
            root,
            "vehicle",
            id=f"{origin}_{i:06d}",
            type="car",
            depart=f"{t:.3f}",
            departLane="best",
            departSpeed="max",
        )
        ET.SubElement(vehicle, "route", edges=" ".join(edges))
        traversals += len(edges) == 3
    route = directory / "vehicles.rou.xml"
    write_xml(root, route)
    save_json(
        directory / "manifest.json",
        {
            "sha256": sha256(route),
            "scenario": scenario,
            "split": split,
            "seed": seed,
            "demand_seconds": seconds,
            "planned": len(vehicles),
            "two_junction_trips": traversals,
            "rates_vph": rates.tolist(),
        },
    )
    return route
