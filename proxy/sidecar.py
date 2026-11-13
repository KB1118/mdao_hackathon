"""Starts, stops, and restarts the Headroom sidecar as a child process.

The child's environment is built only from the dashboard settings. HEADROOM_* variables from
the shell are dropped, so the saved settings are the single source of truth.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _base_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("HEADROOM_") and key != "HF_HUB_OFFLINE"
    }


class Sidecar:
    def __init__(self, log_path: Path, python_exe: str | None = None) -> None:
        self._log_path = log_path
        self._python = python_exe or sys.executable
        self._proc: subprocess.Popen[bytes] | None = None
        self._port: int | None = None
        self._started_at: float | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "pid": self._proc.pid if self.running and self._proc else None,
            "port": self._port,
            "uptime_seconds": round(time.time() - self._started_at, 1)
            if self.running and self._started_at
            else None,
            "log_path": str(self._log_path),
        }

    def start(self, env_values: dict[str, str], port: int) -> None:
        if self.running:
            return
        env = _base_env()
        env.update(env_values)
        env["HEADROOM_SIDECAR_PORT"] = str(port)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self._log_path.open("ab")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        self._proc = subprocess.Popen(
            [self._python, "-m", "proxy.sidecar_main"],
            cwd=str(PROJECT_DIR),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        self._port = port
        self._started_at = time.time()

    def stop(self, timeout: float = 15.0) -> None:
        if not self.running or self._proc is None:
            self._proc = None
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5)
        self._proc = None
        self._started_at = None

    def restart(self, env_values: dict[str, str], port: int) -> None:
        self.stop()
        self.start(env_values, port)
