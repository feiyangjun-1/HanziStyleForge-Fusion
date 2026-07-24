from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


class SafeStopRequested(RuntimeError):
    """Raised only after a durable checkpoint or per-glyph state save."""


class LongRunGuard:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        runtime = cfg.get("runtime", {})
        self.stop_file = Path(runtime.get("safe_stop_file", "")) if runtime.get("safe_stop_file") else None
        self.minimum_free_gb = float(runtime.get("minimum_free_disk_gb", 0.0))
        thermal = runtime.get("thermal_guard", {})
        self.thermal_enabled = bool(thermal.get("enabled", False))
        self.pause_above = int(thermal.get("pause_above_c", 88))
        self.resume_below = int(thermal.get("resume_below_c", 80))
        self.poll_seconds = max(5, int(thermal.get("poll_seconds", 30)))
        self.runtime_check_interval = max(5.0, float(thermal.get("check_interval_seconds", 10.0)))
        self._last_runtime_check = 0.0
        self._last_disk_check = 0.0
        self.work_dir = Path(cfg["paths"]["work_dir"])

    def _temperature(self) -> int | None:
        if not self.thermal_enabled:
            return None
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if completed.returncode != 0:
                return None
            first = completed.stdout.strip().splitlines()[0]
            return int(float(first.strip()))
        except Exception:
            return None

    def _check_disk(self) -> None:
        if self.minimum_free_gb <= 0:
            return
        target = self.work_dir if self.work_dir.exists() else self.work_dir.parent
        free = shutil.disk_usage(target).free / (1024 ** 3)
        if free < self.minimum_free_gb:
            raise RuntimeError(
                f"Free disk space is {free:.1f} GB, below the safety threshold of {self.minimum_free_gb:.1f} GB."
                "Free disk space and run again; the program will resume from its checkpoint."
            )

    def _thermal_pause(self) -> None:
        temperature = self._temperature()
        if temperature is None or temperature < self.pause_above:
            return
        print(
            f"\nGPU temperature {temperature}°C reached the safety limit; execution paused at a safe checkpoint. "
            f"Execution will resume automatically below {self.resume_below}°C.",
            flush=True,
        )
        while True:
            time.sleep(self.poll_seconds)
            temperature = self._temperature()
            if temperature is None or temperature <= self.resume_below:
                print("GPU temperature recovered; resuming.", flush=True)
                return

    def runtime_boundary(self) -> None:
        """Periodically apply thermal protection between training batches.

        This method intentionally does not honour the safe-stop file because a
        durable checkpoint may not have been written yet.  It is inexpensive to
        call every batch: nvidia-smi is launched only when the configured timer
        expires.
        """
        now = time.monotonic()
        if now - self._last_runtime_check < self.runtime_check_interval:
            return
        self._last_runtime_check = now
        self._thermal_pause()
        # Disk usage changes much more slowly than temperature.
        if now - self._last_disk_check >= 120.0:
            self._check_disk()
            self._last_disk_check = now

    def checkpoint_boundary(self) -> None:
        """Call after an atomic checkpoint/state write, never before it."""
        self._check_disk()
        self._last_disk_check = time.monotonic()
        self._thermal_pause()
        self._last_runtime_check = time.monotonic()
        if self.stop_file is not None and self.stop_file.exists():
            raise SafeStopRequested(
                f"Safe-stop file detected: {self.stop_file}. Current progress has been saved."
            )
