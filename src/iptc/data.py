"""Trajectory data model and CitySim import.坐标、时间单位由字段和文档确定。"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass
class Recording:
    metadata: dict
    tracks: list[str]
    track: np.ndarray
    time: np.ndarray
    xy: np.ndarray
    dt: int = 1

    def validate(self) -> None:
        n = len(self.time)
        if n == 0 or self.xy.shape != (n, 2) or len(self.track) != n:
            raise ValueError("Recording arrays must be nonempty and have matching shapes")
        if not np.isfinite(self.xy).all():
            raise ValueError("Nonfinite coordinates: no silent removal is allowed")
        if self.dt <= 0 or len(set(self.tracks)) != len(self.tracks):
            raise ValueError("Invalid time step or duplicate track names")
        if self.track.min() < 0 or self.track.max() >= len(self.tracks):
            raise ValueError("Track index out of range")
        keys = np.rec.fromarrays([self.track, self.time])
        if len(np.unique(keys)) != n:
            raise ValueError("Duplicate vehicle/timestamp keys; explicit resolution required")
        order = np.lexsort((self.time, self.track))
        if not np.array_equal(order, np.arange(n)):
            raise ValueError("Canonical arrays must be sorted by vehicle then time")


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def spreadsheet_rows(path: Path) -> list[list[str]]:
    """只读取 XLSX 的 XML 元数据；不运行宏、不加载对象。"""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            strings = ["".join(x.itertext()) for x in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in root.findall("s:sheetData/s:row", ns):
            values = []
            for cell in row.findall("s:c", ns):
                v = cell.find("s:v", ns)
                text = v.text if v is not None else ""
                if cell.get("t") == "s":
                    text = strings[int(text)]
                values.append(text)
            rows.append(values)
    return rows


def frame_rate(path: Path) -> tuple[int, Path]:
    for parent in [path.parent, *path.parents]:
        meta = parent / "metadata.xlsx"
        if meta.exists():
            rows = spreadsheet_rows(meta)
            col = rows[0].index("recordingFrameRate")
            rates = {float(r[col]) for r in rows[1:] if len(r) > col and r[col]}
            if len(rates) != 1 or next(iter(rates)) != int(next(iter(rates))):
                raise ValueError(f"Ambiguous/noninteger frame rate in {meta}")
            return int(next(iter(rates))), meta
    raise ValueError(f"No local frame-rate metadata found for {path}")


def load_citysim(path: Path, root: Path, cfg: dict) -> Recording:
    if cfg.get("position_unit") != "ft" or not cfg.get("unit_evidence"):
        raise ValueError("CitySim requires documented foot units, not magnitude inference")
    fps, meta = frame_rate(path)
    limit = int(cfg["records_per_recording"])
    policy=cfg.get("frame_order_policy","strict")
    if policy not in {"strict","sort"}:
        raise ValueError("frame_order_policy must be strict or sort")
    samples: list[tuple[str, int, float, float]] = []
    last_frame = -1
    input_disorder = 0
    seen=set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"carId", "frameNum", "carCenterXft", "carCenterYft"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Unsupported trajectory schema: {path}")
        for row in reader:
            t = int(row["frameNum"])
            if policy=="strict" and len(samples) >= limit and t != last_frame:
                break  # 只在完整帧边界结束，保留所有车辆。
            if t < last_frame:
                input_disorder += 1
                if policy=="strict":
                    raise ValueError(f"Frame order violation in {path}; streaming selection unsafe")
            x,y=float(row["carCenterXft"])*0.3048,float(row["carCenterYft"])*0.3048
            if policy=="sort":
                key=(row["carId"],t)
                if key in seen:
                    raise ValueError(f"Duplicate vehicle/timestamp key {key} in full CSV {path}; no silent removal")
                seen.add(key)
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError(f"Nonfinite coordinates in full CSV {path}; no silent removal")
                if len(samples)%20000==0:
                    from .resources import check
                    check()
            samples.append((row["carId"],t,x,y))
            last_frame = t
    rows_scanned=len(samples)
    if policy=="sort" and samples:
        # 先完整检查源文件，再排序；不按任意行截断乱序车辆。
        samples.sort(key=lambda r:r[1])
        cutoff=samples[min(limit,len(samples))-1][1]
        samples=[r for r in samples if r[1]<=cutoff]
        logging.info("Full-file frame sorting: %s rows=%d backsteps=%d selected=%d cutoff=%d",
                     path.name,rows_scanned,input_disorder,len(samples),cutoff)
    tracks = sorted({r[0] for r in samples})
    ids = {s: i for i, s in enumerate(tracks)}
    samples.sort(key=lambda r: (ids[r[0]], r[1]))
    track = np.array([ids[r[0]] for r in samples], dtype=np.int32)
    time = np.array([r[1] for r in samples], dtype=np.int64)
    xy = np.array([[r[2], r[3]] for r in samples], dtype=np.float64)
    Recording({},tracks,track,time,xy).validate()
    gaps, jumps = 0, 0
    for i in range(len(tracks)):
        loc = np.flatnonzero(track == i)
        delta = np.diff(time[loc])
        gaps += int(np.sum(delta > 1))
        if len(loc) > 1:
            speed = np.linalg.norm(np.diff(xy[loc], axis=0), axis=1) / (delta / fps)
            jumps += int(np.sum(speed > 80))
    with path.open("rb") as original:
        fingerprint = hashlib.file_digest(original,"sha256").hexdigest()
    metadata = dict(source_dataset=cfg["source_dataset"], source_version=cfg["source_version"],
                    recording_id=path.relative_to(root).as_posix(), source_file=str(path.resolve()),
                    object_type="vehicle (subtype unavailable)", position_unit="m",
                    timestamp_unit=f"1/{fps} second", ticks_per_second=fps,
                    metadata_file=str(meta), input_frame_disorder=input_disorder,
                    frame_order_policy=policy,order_scan_scope="full file" if policy=="sort" else "selected prefix",
                    source_rows_scanned=rows_scanned,
                    time_gaps=gaps, jumps_above_80_mps=jumps,
                    repair_rule=("Full CSV validation and stable frame sorting before complete-frame prefix; then sort vehicle/time; no removal, smoothing or gap filling" if policy=="sort" else "Sort vehicle/time; no point removal, smoothing or gap filling"),
                    original_sha256=fingerprint)
    rec = Recording(metadata, tracks, track, time, xy)
    rec.validate()
    logging.info("Loaded %s: %d rows, %d vehicles, frames %d..%d", path.name,
                 len(time), len(tracks), time.min(), time.max())
    return rec
