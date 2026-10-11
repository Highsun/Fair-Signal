"""Original intersection geometry and seeded, explicit vehicle arrival lists."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from fair_signal.runtime import configure_sumo

DIRECTIONS = ("n", "s", "e", "w")
LANES = tuple(f"{direction}_t_{lane}" for direction in DIRECTIONS for lane in (0, 1))
# Phase order: NS through/right, NS left, EW through/right, EW left.
PHASE_LANES = ((0, 2), (1, 3), (4, 6), (5, 7))
GREEN_STATES = ("GGrrrrGGrrrr", "rrGrrrrrGrrr", "rrrGGrrrrGGr", "rrrrrGrrrrrG")
# Connections are explicitly indexed clockwise, independent of netconvert's ordering.
EXITS = {"n": ("w", "s", "e"), "s": ("e", "n", "w"), "e": ("n", "w", "s"), "w": ("s", "e", "n")}
LINK_BASE = {"n": 0, "e": 3, "s": 6, "w": 9}


def write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: dict | list) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def build_network(directory: Path) -> Path:
    configure_sumo()
    directory.mkdir(parents=True, exist_ok=True)
    net = directory / "intersection.net.xml"
    if net.exists():
        manifest = json.loads((directory / "manifest.json").read_text())
        if sha256(net) != manifest["sha256"]:
            raise ValueError(f"Network checksum mismatch: {net}")
        return net
    nodes = ET.Element("nodes")
    ET.SubElement(nodes, "node", id="t", x="0", y="0", type="traffic_light")
    for name, (x, y) in {"n": (0, 300), "s": (0, -300), "e": (300, 0), "w": (-300, 0)}.items():
        ET.SubElement(nodes, "node", id=name, x=str(x), y=str(y), type="priority")
    edges, connections, logic = (
        ET.Element("edges"),
        ET.Element("connections"),
        ET.Element("tlLogics"),
    )
    for direction in DIRECTIONS:
        for start, end in ((direction, "t"), ("t", direction)):
            ET.SubElement(
                edges,
                "edge",
                {
                    "id": f"{start}_{end}",
                    "from": start,
                    "to": end,
                    "numLanes": "2",
                    "speed": "13.89",
                    "priority": "1",
                },
            )
        for movement, dest in enumerate(EXITS[direction]):
            lane = 1 if movement == 2 else 0
            ET.SubElement(
                connections,
                "connection",
                {
                    "from": f"{direction}_t",
                    "to": f"t_{dest}",
                    "fromLane": str(lane),
                    "toLane": str(lane),
                    "tl": "t",
                    "linkIndex": str(LINK_BASE[direction] + movement),
                },
            )
    tls = ET.SubElement(logic, "tlLogic", id="t", type="static", programID="0", offset="0")
    for green in GREEN_STATES:
        ET.SubElement(tls, "phase", duration="30", state=green)
        ET.SubElement(tls, "phase", duration="3", state=green.replace("G", "y"))
        ET.SubElement(tls, "phase", duration="1", state="r" * 12)
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
            str(net),
            "--no-turnarounds",
            "true",
            "--no-warnings",
            "true",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    # Remove generated timestamp/absolute command paths from the XML header.
    root = ET.parse(net).getroot()
    write_xml(root, net)
    phases = root.find("tlLogic")
    if phases is None or [p.get("state") for p in phases.findall("phase")[::3]] != list(
        GREEN_STATES
    ):
        raise RuntimeError("Generated traffic light phases do not match the controller")
    actual = {
        (c.get("from"), c.get("to")): int(c.get("linkIndex", "-1"))
        for c in root.findall("connection")
        if c.get("tl") == "t"
    }
    expected = {
        (f"{d}_t", f"t_{dest}"): LINK_BASE[d] + m
        for d in DIRECTIONS
        for m, dest in enumerate(EXITS[d])
    }
    if actual != expected:
        raise RuntimeError("Generated traffic light link mapping differs from the design")
    save_json(
        directory / "manifest.json",
        {
            "sha256": sha256(net),
            "source": "original generator",
            "sumo": "1.28.0",
            "arm_length_m": 300,
        },
    )
    return net


def rates_for(scenario: str, split: str, rng: np.random.Generator) -> np.ndarray:
    """Vehicles/hour per direction; rows represent successive thirds of an episode."""
    if split not in ("train", "validation", "test"):
        raise ValueError(f"Unknown split: {split}")
    # Held-out fixed parameter combinations; random training uses continuous perturbations.
    scale = {"train": 1.0, "validation": 1.05, "test": 1.12}[split]
    if scenario == "balanced":
        rates = np.full((3, 4), 270.0)
    elif scenario == "imbalanced":
        rates = np.tile([110.0, 110.0, 450.0, 450.0], (3, 1))
    elif scenario == "changing_peak":
        rates = np.array(
            [[380, 380, 120, 120], [120, 120, 420, 420], [280, 280, 280, 280]], dtype=float
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    if split == "train":
        rates *= rng.uniform(0.8, 1.2, size=(1, 4))
    return rates * scale


def generate_routes(
    directory: Path, scenario: str, split: str, seed: int, demand_seconds: int
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    route_path = directory / "vehicles.rou.xml"
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
        speedFactor="1",
        speedDev="0",
    )
    for d in DIRECTIONS:
        for move, dest in enumerate(EXITS[d]):
            ET.SubElement(root, "route", id=f"{d}_{move}", edges=f"{d}_t t_{dest}")
    vehicles = []
    for part, row in enumerate(rates):
        start, end = demand_seconds * part / 3, demand_seconds * (part + 1) / 3
        for d, rate in zip(DIRECTIONS, row, strict=True):
            at = start + rng.exponential(3600 / rate)
            while at < end:
                # right/straight/left shares: 10/70/20 percent.
                move = int(rng.choice(3, p=[0.1, 0.7, 0.2]))
                vehicles.append((round(at, 3), d, move))
                at += rng.exponential(3600 / rate)
    for index, (at, d, move) in enumerate(sorted(vehicles)):
        ET.SubElement(
            root,
            "vehicle",
            id=f"{d}_{index:06d}",
            type="car",
            route=f"{d}_{move}",
            depart=f"{at:.3f}",
            departLane="1" if move == 2 else "0",
            departSpeed="max",
        )
    write_xml(root, route_path)
    save_json(
        directory / "manifest.json",
        {
            "scenario": scenario,
            "split": split,
            "seed": seed,
            "demand_seconds": demand_seconds,
            "vehicles": len(vehicles),
            "rates_veh_per_hour_n_s_e_w": rates.tolist(),
            "turn_shares_right_straight_left": [0.1, 0.7, 0.2],
            "sha256": sha256(route_path),
        },
    )
    return route_path
