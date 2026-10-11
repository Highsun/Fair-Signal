"""Pure signal state machine, shared by learned and traditional policies."""

from dataclasses import dataclass

from fair_signal.config import EnvConfig
from fair_signal.demand import GREEN_STATES


@dataclass
class Signal:
    config: EnvConfig
    phase: int = 0
    target: int = 0
    mode: str = "green"
    elapsed: int = 0
    switches: int = 0
    forced_switches: int = 0
    blocked_requests: int = 0

    @property
    def state(self) -> str:
        if self.mode == "green":
            return GREEN_STATES[self.phase]
        if self.mode == "yellow":
            return GREEN_STATES[self.phase].replace("G", "y")
        return "r" * 12

    def request(self, action: int) -> None:
        if self.mode != "green":
            return
        if action != self.phase:
            if self.elapsed >= self.config.min_green:
                self._switch(action)
            else:
                self.blocked_requests += 1

    def _switch(self, target: int) -> None:
        self.target, self.mode, self.elapsed = target, "yellow", 0
        self.switches += 1

    def before_second(self) -> str:
        # Enforce at one-second resolution, even between decision boundaries.
        if self.mode == "green" and self.elapsed >= self.config.max_green:
            self._switch((self.phase + 1) % 4)
            self.forced_switches += 1
        return self.state

    def after_second(self) -> None:
        self.elapsed += 1
        if self.mode == "yellow" and self.elapsed >= self.config.yellow_seconds:
            self.mode, self.elapsed = "all_red", 0
        elif self.mode == "all_red" and self.elapsed >= self.config.all_red_seconds:
            self.phase, self.mode, self.elapsed = self.target, "green", 0
