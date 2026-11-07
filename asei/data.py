from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .config import file_sha256
from .data_process import read_data
from .resources import cooperate

DATASETS = ("PSM", "SMD", "SWAT", "PUMP")


@dataclass
class Entity:
    dataset: str
    name: str
    path: Path
    train_lens: int
    test_continuous: bool = False
    validation_path: Path | None = None


@dataclass
class Record:
    z: np.ndarray
    mask: np.ndarray
    target_start: int = 0
    # Compact, fixed-model routing only; never cache trainable detector outputs.
    routing: dict | None = None


def _bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (int, float, np.number)) and value in (0, 1):
        return bool(value)
    if str(value).lower() in {"true", "1", "yes"}:
        return True
    if str(value).lower() in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Invalid boolean: {value}")


def _dataset(row: dict) -> str | None:
    for key in ("dataset", "dataset_name", "data_set_name", "data_name", "source", "file_name"):
        name = str(row.get(key, "")).upper()
        for ds in DATASETS:
            if re.search(r"(?:^|[^A-Z])" + ds + r"(?:[^A-Z]|$)", name):
                return ds
    return None


def metadata_path(data_root: str | Path, manifest=None) -> Path:
    root = Path(data_root).resolve()
    if manifest:
        path = Path(manifest).resolve()
        if not path.is_file():
            raise FileNotFoundError("Metadata manifest not found: " + str(path))
        return path
    candidates = [p for p in (root / "DETECT_META.csv", root / "anomaly_detect" / "DETECT_META.csv",
                              root / "dataset" / "anomaly_detect" / "DETECT_META.csv") if p.is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError("data.root must identify one DETECT_META.csv, directly or under anomaly_detect/.")
    return candidates[0]


def discover(data_root: str | Path, manifest=None, multivariate_only=False) -> list[Entity]:
    meta_path = metadata_path(data_root, manifest)
    base = meta_path.parent if manifest else meta_path.parent / "data"
    meta = pd.read_csv(meta_path)
    if not {"file_name", "train_lens"}.issubset(meta.columns):
        raise ValueError("Metadata requires file_name and train_lens columns.")
    entities, seen = [], set()
    for row in meta.to_dict("records"):
        ds = _dataset(row)
        if ds is None:
            continue
        if multivariate_only and _bool(row.get("if_univariate", False)):
            continue
        name = str(row.get("entity", Path(str(row["file_name"])).stem))
        # Source names may contain '@', e.g. machine-3-4.test.csv@7.csv.
        if not re.fullmatch(r"[\w.@\-]+", name) or name in {".", ".."}:
            raise ValueError(f"Unsafe entity identifier: {name}")
        if (ds, name) in seen:
            raise ValueError(f"Duplicate entity {ds}/{name}")
        seen.add((ds, name))
        length = row["train_lens"]
        # Keep ineligible entries visible to inspect instead of excluding entire datasets silently.
        length = int(length) if not pd.isna(length) and float(length).is_integer() else 0
        file_value = row.get("path")
        file_path = Path(str(row["file_name"] if file_value is None or pd.isna(file_value) else file_value))
        if not file_path.is_absolute():
            file_path = base / file_path
            if not manifest:
                # Also accept CSV files adjacent to the metadata, or file_name=data/xxx.csv.
                adjacent = meta_path.parent / str(row["file_name"] if file_value is None or pd.isna(file_value) else file_value)
                if adjacent.is_file() and file_path.is_file() and adjacent.resolve() != file_path.resolve():
                    raise ValueError("Ambiguous data file in both metadata and data directories: " + str(row["file_name"]))
                if adjacent.is_file():
                    file_path = adjacent
        val = row.get("validation_path")
        val_path = None if val is None or pd.isna(val) else Path(str(val))
        if val_path and not val_path.is_absolute():
            val_path = meta_path.parent / val_path
        entities.append(Entity(ds, name, file_path.resolve(), int(length),
                               _bool(row.get("test_continuous", False)), val_path))
    if not entities:
        raise ValueError("No " + "/".join(DATASETS) + " entries found. Supply dataset names in a manifest.")
    return entities


def read_frame(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    if "cols" in header.columns:
        frame = read_data(str(path))
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("Expected a multivariate series, not spatial-temporal data.")
    else:
        frame = pd.read_csv(path)
        frame = frame.drop(columns=[c for c in frame if c.lower() in {"date", "timestamp", "time"}
                                     or c.startswith("Unnamed:")])
    if frame.columns.duplicated().any() or "label" not in frame:
        raise ValueError(f"{path}: expected unique numeric channels and a binary label column.")
    frame = frame.apply(pd.to_numeric, errors="raise")
    labels = frame["label"].to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{path}: labels must be finite binary values 0/1.")
    return frame


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray):
        finite = np.isfinite(x)
        count = finite.sum(0)
        if (count == 0).any():
            raise ValueError("Drop channels with no finite normal-training observations before fitting.")
        mean = np.where(finite, x, 0).sum(0, dtype=np.float64) / count
        variance = np.where(finite, (x - mean) ** 2, 0).sum(0, dtype=np.float64) / count
        scale = np.sqrt(variance)
        scale[scale < 1e-12] = 1
        return cls(mean, scale)

    def transform(self, x: np.ndarray, previous=None) -> Record:
        """Causal filling: no interpolation/backfill or test-derived statistics."""
        mask = np.isfinite(x)
        filled = np.where(mask, x, np.nan)
        seed = self.mean if previous is None else previous
        # A prepended past value lets pandas implement forward fill without future access.
        filled = pd.DataFrame(np.vstack([seed, filled])).ffill().to_numpy()[1:]
        return Record(((filled - self.mean) / self.scale).astype(np.float32), mask)

    def as_dict(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


def load_entity(entity: Entity, fraction: float):
    if not 0 < fraction < 1:
        raise ValueError("train_fraction must be in (0, 1).")
    cooperate()
    frame = read_frame(entity.path)
    boundary = entity.train_lens
    if not 0 < boundary < len(frame):
        raise ValueError(f"{entity.name}: train_lens is outside the series.")
    if frame["label"].iloc[:boundary].any():
        raise ValueError(f"{entity.name}: official training prefix contains anomalies; normal training and validation data are required.")
    n_fit = boundary if entity.validation_path else int(boundary * fraction)
    if n_fit < 1 or (not entity.validation_path and n_fit == boundary):
        raise ValueError(f"{entity.name}: training prefix is too short for a train/validation split.")
    columns = frame.columns.drop("label")
    x = frame.loc[:, columns].to_numpy(np.float64)
    keep = np.isfinite(x[:n_fit]).any(0)
    dropped = columns[~keep].tolist()
    columns = columns[keep].tolist()
    if not columns:
        raise ValueError("No observed training channels.")
    x = x[:, keep]
    scaler = Standardizer.fit(x[:n_fit])
    normal = scaler.transform(x[:boundary])
    train = Record(normal.z[:n_fit], normal.mask[:n_fit])
    if entity.validation_path:
        vf = read_frame(entity.validation_path)
        if vf["label"].any():
            raise ValueError("Independent validation record must be normal.")
        if set(vf.columns.drop("label")) != set(frame.columns.drop("label")):
            raise ValueError("Validation channel schema differs from training.")
        valid = scaler.transform(vf[columns].to_numpy(np.float64))
    else:
        valid = Record(normal.z[n_fit:], normal.mask[n_fit:])
    previous = normal.z[-1].astype(np.float64) * scaler.scale + scaler.mean if entity.test_continuous else None
    test = scaler.transform(x[boundary:], previous)
    provenance = {"dataset": entity.dataset, "entity": entity.name,
                  "source": str(entity.path), "source_sha256": file_sha256(entity.path),
                  "validation_sha256": file_sha256(entity.validation_path) if entity.validation_path else None,
                  "official_train_lens": boundary, "fit_end": n_fit,
                  "test_continuous": entity.test_continuous, "channels": columns,
                  "dropped_unobserved_training_channels": dropped,
                  "scaler": scaler.as_dict()}
    return train, valid, test, frame["label"].to_numpy(np.int8)[boundary:], scaler, provenance


def validation_context(train: Record, valid: Record, window: int, independent=False) -> Record:
    if independent:
        return Record(valid.z, valid.mask, window)
    return Record(np.concatenate([train.z[-window:], valid.z]),
                  np.concatenate([train.mask[-window:], valid.mask]), window)
