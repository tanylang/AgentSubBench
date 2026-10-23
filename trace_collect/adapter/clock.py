"""freezegun-immune kernel-level clock (calls the OS directly via ctypes).

AppWorld uses freezegun to freeze Python's time/datetime modules (in this
environment, even time.perf_counter / time_ns / monotonic_ns are frozen in
practice); while frozen, all time-module-based timestamps and duration
measurements are constant simulated-world values (the constant 2023-05-18 →
gap/duration are all 0). freezegun can only patch the Python layer, not the
kernel — here we call the OS clock directly:
  - real_time_ns(): wall clock (Windows GetSystemTimePreciseAsFileTime /
                    Linux clock_gettime(CLOCK_REALTIME)) → Unix nanoseconds
  - monotonic_ns(): monotonic clock (Windows QueryPerformanceCounter /
                    Linux clock_gettime(CLOCK_MONOTONIC)) → for duration
                    measurement only; only deltas are meaningful, and it is
                    unrelated to the wall-clock epoch

Rule: every timestamp / duration measurement point that runs inside the AppWorld
world must use this module, never time.*.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import ctypes.wintypes as wt
import sys

_WIN = sys.platform.startswith("win")

if _WIN:
    _GetSystemTimePreciseAsFileTime = ctypes.windll.kernel32.GetSystemTimePreciseAsFileTime  # noqa: N816
    _GetSystemTimePreciseAsFileTime.argtypes = [ctypes.POINTER(wt.FILETIME)]
    _GetSystemTimePreciseAsFileTime.restype = None

    _QueryPerformanceCounter = ctypes.windll.kernel32.QueryPerformanceCounter  # noqa: N816
    _QueryPerformanceCounter.argtypes = [ctypes.POINTER(ctypes.c_int64)]
    _QueryPerformanceCounter.restype = ctypes.c_int
    _QueryPerformanceFrequency = ctypes.windll.kernel32.QueryPerformanceFrequency  # noqa: N816
    _QueryPerformanceFrequency.argtypes = [ctypes.POINTER(ctypes.c_int64)]
    _QueryPerformanceFrequency.restype = ctypes.c_int
    _QPF = ctypes.c_int64()
    _QueryPerformanceFrequency(ctypes.byref(_QPF))
else:
    class _Timespec(ctypes.Structure):
        _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

    _librt = ctypes.CDLL(ctypes.util.find_library("rt") or ctypes.util.find_library("c"),
                         use_errno=True)

_EPOCH_1601_OFFSET_100NS = 116444736000000000  # 1601-01-01 → 1970-01-01


def real_time_ns() -> int:
    """Wall clock in nanoseconds since the Unix epoch (freezegun-immune)."""
    if _WIN:
        ft = wt.FILETIME()
        _GetSystemTimePreciseAsFileTime(ctypes.byref(ft))
        t100ns = (ft.dwHighDateTime << 32) + ft.dwLowDateTime
        return (t100ns - _EPOCH_1601_OFFSET_100NS) * 100
    ts = _Timespec()
    if _librt.clock_gettime(0, ctypes.byref(ts)) != 0:  # 0 = CLOCK_REALTIME
        raise OSError(ctypes.get_errno(), "clock_gettime failed")
    return ts.tv_sec * 1_000_000_000 + ts.tv_nsec


def monotonic_ns() -> int:
    """Monotonic clock in nanoseconds (freezegun-immune), for duration measurement only (unrelated to the wall clock; only deltas are meaningful)."""
    if _WIN:
        counter = ctypes.c_int64()
        if not _QueryPerformanceCounter(ctypes.byref(counter)):
            raise OSError("QueryPerformanceCounter failed")
        return int(counter.value * 1_000_000_000 / _QPF.value)
    ts = _Timespec()
    if _librt.clock_gettime(1, ctypes.byref(ts)) != 0:  # 1 = CLOCK_MONOTONIC
        raise OSError(ctypes.get_errno(), "clock_gettime failed")
    return ts.tv_sec * 1_000_000_000 + ts.tv_nsec


def monotonic_ms() -> float:
    """Monotonic clock in milliseconds (convenient form for duration measurement)."""
    return monotonic_ns() / 1e6
