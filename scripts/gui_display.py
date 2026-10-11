"""Find a reachable macOS XQuartz display without changing global settings."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

XDPYINFO = "/opt/X11/bin/xdpyinfo"


def command_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=2, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def resolve_display(wait_seconds: float = 10) -> str:
    if not Path(XDPYINFO).is_file():
        raise RuntimeError("未找到 XQuartz：请先安装并打开 XQuartz。")
    inherited = os.environ.get("DISPLAY", "")
    deadline = time.monotonic() + wait_seconds
    started = False
    while True:
        candidates = [inherited, command_output(["launchctl", "getenv", "DISPLAY"])]
        if Path("/private/tmp/.X11-unix/X0").is_socket():
            candidates.append(":0")
        for candidate in dict.fromkeys(candidates):
            if candidate and command_output([XDPYINFO, "-display", candidate]) is not None:
                return candidate
        if not started:
            if command_output(["open", "-a", "XQuartz"]) is None:
                raise RuntimeError("无法打开 XQuartz，请手动启动后重试。")
            started = True
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "XQuartz 显示服务仍不可连接，请重启 XQuartz 后重试；"
                "首次安装可能需要注销并重新登录。"
            )
        time.sleep(0.5)


if __name__ == "__main__":
    try:
        display = resolve_display()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
    if display != os.environ.get("DISPLAY", ""):
        print(f"SUMO GUI 使用已验证的 XQuartz 显示地址：{display}", file=sys.stderr)
    print(display)
