"""Audit second-resolution signal traces independently of controller decisions."""


def audit_signal_rows(rows, config):
    intervals = 0
    for junction in ("a", "b"):
        blocks = []
        for row in rows:
            key = (row[f"{junction}_mode"], int(row[f"{junction}_phase"]))
            if not blocks or blocks[-1][0] != key:
                blocks.append([key, []])
            blocks[-1][1].append(row)
        for i, ((mode, phase), block) in enumerate(blocks):
            elapsed = [int(r[f"{junction}_elapsed"]) for r in block]
            if elapsed != list(range(len(block))):
                raise ValueError("Non-contiguous/reset signal timer")
            if i == len(blocks) - 1:
                continue  # Simulation can end within a physical interval.
            intervals += 1
            duration = len(block)
            next_mode, next_phase = blocks[i + 1][0]
            if mode == "green":
                if not config.min_green <= duration <= config.max_green:
                    raise ValueError("Invalid green duration")
                if next_mode != "yellow" or next_phase != phase:
                    raise ValueError("Green phase changed without yellow")
            elif mode == "yellow":
                if (
                    duration != config.yellow_seconds
                    or next_mode != "all_red"
                    or next_phase != phase
                ):
                    raise ValueError("Invalid yellow clearance")
            elif mode == "all_red":
                if duration != config.all_red_seconds or next_mode != "green":
                    raise ValueError("Invalid all-red clearance")
                if next_phase != int(block[-1][f"{junction}_target"]):
                    raise ValueError("Clearance target was not committed")
            else:
                raise ValueError("Unknown signal mode")
    return intervals
