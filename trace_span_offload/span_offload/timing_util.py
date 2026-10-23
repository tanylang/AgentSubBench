"""A freezegun-immune monotonic wall clock (ms).

The AppWorld environment uses freezegun to freeze simulated time; freezegun 1.5.x
patches every clock in the time module (perf_counter/monotonic/monotonic_ns/time_ns)
to a fake epoch (2023-05-18). During restore/world construction the frozen clock can
leak beyond the call boundary, and perf_counter differences across that boundary are
garbage values (measured restore_ms ≈ 1.68e12 ms). This module takes the monotonic
clock via ctypes syscalls, unaffected by module-level patching:
  Windows: QueryPerformanceCounter (measured normal differences inside the frozen region)
  Linux:   clock_gettime(CLOCK_MONOTONIC)
  Fallback: time.perf_counter
"""

from __future__ import annotations

import ctypes
import os
import time

_IMPL = None


def _load():
    global _IMPL
    if _IMPL is not None:
        return _IMPL
    try:
        if os.name == "nt":
            _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            _qpc = _k32.QueryPerformanceCounter
            _qpf = _k32.QueryPerformanceFrequency

            def _mono_win():
                c = ctypes.c_int64()
                f = ctypes.c_int64()
                _qpc(ctypes.byref(c))
                _qpf(ctypes.byref(f))
                return c.value * 1000.0 / max(f.value, 1)

            _IMPL = _mono_win
        else:
            _libc = ctypes.CDLL(None)
            _clock_gettime = _libc.clock_gettime
            _clock_gettime.argtypes = [ctypes.c_int, ctypes.c_void_p]
            _clock_gettime.restype = ctypes.c_int

            class _Timespec(ctypes.Structure):
                _fields_ = [("tv_sec", ctypes.c_longlong),
                            ("tv_nsec", ctypes.c_longlong)]

            def _mono_posix():
                ts = _Timespec()
                _clock_gettime(1, ctypes.byref(ts))  # CLOCK_MONOTONIC = 1
                return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6

            _IMPL = _mono_posix
    except Exception:
        _IMPL = lambda: time.perf_counter() * 1000.0  # fallback (will be patched by freezegun)
    return _IMPL


def monotonic_ms() -> float:
    """freezegun-immune monotonic wall clock, in milliseconds."""
    return _load()()
