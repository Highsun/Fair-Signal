"""Frozen Poisson/turn generator with explicit demand scaling."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from fair_signal.arterial.demand import ORIGINS, rates_for, route_edges
from fair_signal.demand import save_json, sha256, write_xml


def generate_scaled_routes(
    directory: Path, scenario: str, split: str, seed: int, seconds: int, multiplier: float = 1.0
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Demand multiplier must be finite and positive")
    rates = rates_for(scenario, split, rng) * multiplier
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
            "demand_multiplier": multiplier,
        },
    )
    return route
