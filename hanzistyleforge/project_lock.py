from __future__ import annotations

import os
from pathlib import Path
from typing import IO


class ProjectLock:
    """Single-writer lock for one work directory.

    The operating system releases the lock automatically when Python exits or
    the computer loses power.  The small lock file may remain, but a stale file
    without an active OS lock does not block the next run.
    """

    def __init__(self, work_dir: str | Path) -> None:
        work = Path(work_dir)
        self.path = work.parent / f".{work.name}.hanzistyleforge.lock"
        self.handle: IO[bytes] | None = None

    def __enter__(self) -> "ProjectLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0)
        if handle.read(1) == b"":
            handle.seek(0)
            handle.write(b"0")
            handle.flush()
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            handle.close()
            raise RuntimeError(
                "Another HanziStyleForge process is already using the same work directory. Do not launch multiple training or generation jobs at the same time. "
                "Run run_status.bat separately to inspect progress."
            ) from exc
        handle.seek(0)
        payload = f"PID={os.getpid()}\n".encode("ascii", errors="replace")
        handle.write(payload)
        handle.truncate()
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
        self.handle = handle
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        handle = self.handle
        self.handle = None
        if handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            # Closing the descriptor releases an OS lock.  Antivirus, cloud
            # synchronization, or an already-released Windows byte-range lock
            # can make LK_UNLCK fail with PermissionError.  Cleanup must never
            # hide the original training exception or turn a recoverable error
            # into a second traceback.
            pass
        finally:
            try:
                handle.close()
            except OSError:
                pass
