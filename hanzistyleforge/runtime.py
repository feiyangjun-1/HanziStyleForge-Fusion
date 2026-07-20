from __future__ import annotations

import os
from typing import Any


def configure_runtime(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply conservative process-wide thread settings.

    Font conversion performs thousands of small OpenCV operations between GPU
    batches.  On high-core-count systems, allowing OpenCV, OpenBLAS and PyTorch
    to each create a full thread pool can make inference dramatically slower or
    appear to stall.  Final therefore uses a small CPU pool and a single OpenCV
    worker while leaving CUDA kernels unaffected.
    """

    training = cfg.get("training", {})
    requested = max(1, int(training.get("cpu_threads", 4)))
    torch_threads = max(1, min(6, requested))
    opencv_threads = max(1, min(2, int(training.get("opencv_threads", 1))))
    interop_threads = max(1, min(2, int(training.get("interop_threads", 1))))

    # These environment variables mainly protect libraries initialised after
    # startup.  Explicit APIs below are the source of truth for this process.
    os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(torch_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(torch_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(torch_threads))

    result: dict[str, Any] = {
        "requested_cpu_threads": requested,
        "torch_threads": None,
        "torch_interop_threads": None,
        "cudnn_benchmark": None,
        "cuda_tf32_matmul": None,
        "cudnn_tf32": None,
        "float32_matmul_precision": None,
        "opencv_threads": None,
        "opencv_opencl": None,
        "windows_sleep_prevention": False,
    }

    try:
        import torch

        torch.set_num_threads(torch_threads)
        result["torch_threads"] = int(torch.get_num_threads())
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError:
            # PyTorch allows changing the inter-op pool only before the first
            # parallel operation.  Re-entered library use may legitimately hit
            # this path; keep the already-initialised value.
            pass
        result["torch_interop_threads"] = int(torch.get_num_interop_threads())
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except (AttributeError, RuntimeError):
                pass
            result["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
            result["cuda_tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
            result["cudnn_tf32"] = bool(torch.backends.cudnn.allow_tf32)
            try:
                result["float32_matmul_precision"] = torch.get_float32_matmul_precision()
            except AttributeError:
                result["float32_matmul_precision"] = "unsupported"
    except Exception as exc:  # pragma: no cover - environment diagnostic
        result["torch_error"] = str(exc)

    try:
        import cv2

        cv2.setNumThreads(opencv_threads)
        try:
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass
        result["opencv_threads"] = int(cv2.getNumThreads())
        try:
            result["opencv_opencl"] = bool(cv2.ocl.useOpenCL())
        except Exception:
            result["opencv_opencl"] = False
    except Exception as exc:  # pragma: no cover - environment diagnostic
        result["opencv_error"] = str(exc)

    if os.name == "nt" and bool(cfg.get("runtime", {}).get("prevent_system_sleep", True)):
        try:
            import ctypes

            # Keep the computer awake while this Python process is running. The
            # display may still turn off. The execution state is thread-scoped
            # and is released when the process exits.
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            value = ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            )
            result["windows_sleep_prevention"] = bool(value)
        except Exception as exc:  # pragma: no cover - Windows-only diagnostic
            result["windows_sleep_error"] = str(exc)

    return result
