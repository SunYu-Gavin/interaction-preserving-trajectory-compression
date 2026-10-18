"""Read all vehicle rows from native InterHub Feather caches, read-only."""
import hashlib
import sys
import zipfile
from pathlib import Path
import numpy as np
from .data import Recording
from .static_metadata import vehicle_metadata


def load_interhub(member: str, cfg: dict) -> Recording:
    import pyarrow as pa
    import pyarrow.feather as feather
    archive = Path(cfg["data_root"])/cfg["cache_archive"]
    folder, filename = member.rsplit("/", 1)
    tag = filename.removeprefix("agent_data_dt").removesuffix(".feather")
    with zipfile.ZipFile(archive) as z:
        raw_metadata = z.read(folder+"/scene_metadata_dt"+tag+".dill")
        scene, types = vehicle_metadata(raw_metadata)
        raw = z.read(member)
    dt = float(scene["dt"])
    ticks = round(1/dt)
    if dt != float(tag) or abs(ticks*dt-1) > 1e-12:
        raise ValueError("Native scene time unit inconsistent")
    table = feather.read_table(pa.BufferReader(raw), columns=["agent_id", "scene_ts", "x", "y"])
    if not pa.types.is_integer(table.schema.field("scene_ts").type) or any(table[c].null_count for c in table.column_names):
        raise ValueError("Native cache requires integer steps and non-null fields")
    names = table["agent_id"].to_pylist()
    if not set(names).issubset(types):
        raise ValueError("Feather has agents missing from complete type metadata")
    keep = np.array([types[n] == 1 for n in names])
    table = table.filter(pa.array(keep))
    names = table["agent_id"].to_pylist()
    times = table["scene_ts"].to_numpy().astype(np.int64, copy=False)
    xy = np.column_stack([table[c].to_numpy() for c in ["x", "y"]]).astype(np.float64)
    if not len(times) or np.any(times < 0) or np.any(times >= scene["length_timesteps"]) or not np.isfinite(xy).all():
        raise ValueError("Invalid full vehicle table")
    # Check all source vehicle keys before selecting the complete-frame prefix.
    all_names = sorted(set(names)); lookup = {n:i for i,n in enumerate(all_names)}
    ids = np.array([lookup[n] for n in names], dtype=np.int32)
    order = np.lexsort((times, ids))
    probe = Recording({}, all_names, ids[order], times[order], xy[order]); probe.validate()
    cutoff = int(np.sort(times)[min(len(times), cfg["records_per_recording"])-1])
    selected = times <= cutoff
    names = [n for n,k in zip(names, selected, strict=True) if k]
    tracks = sorted(set(names)); lookup = {n:i for i,n in enumerate(tracks)}
    ids = np.array([lookup[n] for n in names], dtype=np.int32)
    times, xy = times[selected], xy[selected]
    order = np.lexsort((times, ids))
    source = scene["env_name"]
    group = scene["name"].split("=",1)[0] if source.startswith("nuplan") else scene["name"]
    metadata = dict(source_dataset="InterHub v7 / "+source, source_version="local-v7-native-cache",
                    recording_id=member, original_recording_group=source+"/"+group,
                    source_file=str(archive)+"::"+member, object_type="all AgentType.VEHICLE (1), including ego",
                    position_unit="m", timestamp_unit=f"1/{ticks} second", ticks_per_second=ticks,
                    original_sha256=hashlib.sha256(raw).hexdigest(), metadata_sha256=hashlib.sha256(raw_metadata).hexdigest(),
                    source_rows_scanned=len(keep), source_vehicle_rows=len(probe.time),
                    selection_rule="All vehicle rows; complete native-frame prefix; no event-participant restriction",
                    metadata_reader="Static pickle opcode syntax; inert globals/reducers; no unpickling",
                    feather_reader_version=pa.__version__, unit_evidence=cfg["unit_evidence"],
                    scene_name=scene["name"], native_dt_s=dt, full_agent_type_counts={str(k):sum(v==k for v in types.values()) for k in range(5)})
    rec = Recording(metadata, tracks, ids[order], times[order], xy[order]); rec.validate()
    return rec
