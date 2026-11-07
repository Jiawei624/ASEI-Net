"""Configuration-managed inspection reports and reusable eligible-entity lists."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import file_sha256, report_path, write_json
from .data import load_entity, metadata_path


def source_stamp(cfg):
    source = metadata_path(cfg["data"]["root"], cfg["data"].get("manifest"))
    return {"data_root": str(Path(cfg["data"]["root"]).resolve()),
            "source_metadata": str(source), "source_metadata_sha256": file_sha256(source),
            "multivariate_only": cfg["data"].get("multivariate_only", True),
            "train_fraction": cfg["data"]["train_fraction"]}


def inspect_entities(entities, cfg, read_data=False):
    rows = []
    for entity in entities:
        row = {"dataset": entity.dataset, "entity": entity.name, "file": str(entity.path),
               "train_lens": entity.train_lens, "exists": entity.path.is_file()}
        if read_data:
            try:
                train, val, test, _, _, provenance = load_entity(entity, cfg["data"]["train_fraction"])
                if cfg["data"].get("multivariate_only", True) and train.z.shape[1] < 2:
                    raise ValueError("Multivariate experiments require at least two observed training channels.")
                row.update(status="eligible", fit=len(train.z), validation=len(val.z), test=len(test.z),
                           channels=train.z.shape[1], source_sha256=provenance["source_sha256"])
                selection = int(len(val.z) * cfg["data"]["selection_fraction"])
                row.update(selection=selection, calibration=len(val.z) - selection)
            except (ValueError, FileNotFoundError, KeyError) as exc:
                row.update(status="ineligible", reason=str(exc))
        rows.append(row)
    return pd.DataFrame(rows)


def save_prepared_manifest(entities, inventory, cfg, datasets, entity_filter=None):
    accepted = set(zip(inventory.loc[inventory.status.eq("eligible"), "dataset"],
                       inventory.loc[inventory.status.eq("eligible"), "entity"]))
    columns = ["dataset", "entity", "file_name", "path", "train_lens", "test_continuous", "validation_path"]
    rows = [{"dataset": e.dataset, "entity": e.name, "file_name": e.path.name, "path": str(e.path),
             "train_lens": e.train_lens, "test_continuous": e.test_continuous,
             "validation_path": str(e.validation_path) if e.validation_path else ""}
            for e in entities if (e.dataset, e.name) in accepted]
    path = report_path(cfg, "eligible_manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows, columns=columns).to_csv(temporary, index=False)
    temporary.replace(path)
    write_json(path.with_suffix(".meta.json"), {"format": 1, "source": source_stamp(cfg),
        "datasets": sorted(datasets),
        "entity_filter": sorted({value for e in entities for value in (e.name, e.path.name)}) if entity_filter else None,
        "entities": len(rows), "manifest_sha256": file_sha256(path)})
    return path


def training_manifest(cfg, datasets, entity_filter=None):
    """Explicit source manifests take precedence; prepared lists are validated before reuse."""
    if cfg["data"].get("manifest"):
        return cfg["data"]["manifest"]
    path = report_path(cfg, "eligible_manifest")
    if not cfg["data"].get("use_prepared_manifest", True) or not path.is_file():
        return None
    instruction = "Run python -m asei inspect --prepare for the requested datasets again."
    try:
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("Prepared manifest has no valid source record. " + instruction) from exc
    if meta.get("format") != 1 or meta.get("source") != source_stamp(cfg):
        raise ValueError("Prepared manifest source/configuration changed. " + instruction)
    if meta.get("manifest_sha256") != file_sha256(path):
        raise ValueError("Prepared manifest was changed after inspection. " + instruction)
    if not set(datasets).issubset(meta.get("datasets", [])):
        raise ValueError("Prepared manifest does not cover the requested datasets. " + instruction)
    inspected = meta.get("entity_filter")
    if inspected is not None and (not entity_filter or not set(entity_filter).issubset(inspected)):
        raise ValueError("Prepared manifest covers only the inspected entities. " + instruction)
    return str(path)
