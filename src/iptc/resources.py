"""实际进程峰值工作集测量和合作式资源预算检查。"""
from __future__ import annotations
import ctypes
import os
import time
from pathlib import Path

_active = None

if os.name == "nt":
    from ctypes import wintypes
    class _Counters(ctypes.Structure):
        _fields_ = [("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD),
                    ("PeakWorkingSetSize",ctypes.c_size_t),("WorkingSetSize",ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage",ctypes.c_size_t),("QuotaPagedPoolUsage",ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),("QuotaNonPagedPoolUsage",ctypes.c_size_t),
                    ("PagefileUsage",ctypes.c_size_t),("PeakPagefileUsage",ctypes.c_size_t)]
    _kernel = ctypes.WinDLL("kernel32",use_last_error=True)
    _kernel.GetCurrentProcess.restype = wintypes.HANDLE
    _psapi = ctypes.WinDLL("psapi",use_last_error=True)
    _psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,ctypes.POINTER(_Counters),wintypes.DWORD]


def memory_bytes() -> int:
    if os.name != "nt":
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    counters = _Counters()
    counters.cb = ctypes.sizeof(counters)
    if not _psapi.GetProcessMemoryInfo(_kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
        raise OSError(ctypes.get_last_error(),"GetProcessMemoryInfo failed")
    return int(counters.PeakWorkingSetSize)


class Budget:
    def __init__(self, cfg: dict, run_dir: Path):
        self.deadline = time.perf_counter()+float(cfg["seconds"])
        self.memory_limit = int(cfg["memory_mb"])*1024**2
        self.output_limit = int(cfg["output_mb"])*1024**2
        self.run_dir = run_dir
        if min(float(cfg["seconds"]),cfg["memory_mb"],cfg["output_mb"]) <= 0:
            raise ValueError("Resource budgets must be positive")

    def check(self, disk: bool = False) -> None:
        if time.perf_counter() > self.deadline:
            raise TimeoutError("Run wall-time budget exceeded")
        if memory_bytes() > self.memory_limit:
            raise MemoryError("Process peak working set exceeds memory budget")
        if disk and sum(p.stat().st_size for p in self.run_dir.rglob("*") if p.is_file()) > self.output_limit:
            raise OSError("Run output-space budget exceeded")


def activate(budget: Budget | None) -> None:
    global _active
    _active = budget


def check() -> None:
    if _active is not None:
        _active.check()
