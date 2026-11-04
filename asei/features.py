from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import Record, Standardizer
from .model import NormalRepresentation, DualTrajectoryDetector, segment_arrays
from .progress import periodic_progress


@dataclass
class FeatureSequence:
    """Compact frozen evidence. Descriptor differences are reconstructed on demand."""

    z: np.ndarray
    mask: np.ndarray
    prediction: np.ndarray
    marginal_index: np.ndarray
    joint_index: np.ndarray
    window: int
    labels: np.ndarray | None = None
    weights: np.ndarray | None = None

    def __len__(self):
        return len(self.z)

    def block(self, start, stop, model: NormalRepresentation):
        device = next(model.parameters()).device
        indices = np.arange(start, min(stop, len(self)))
        segment_indices = indices[:, None] + np.arange(-31, 1)
        seg = self.z[segment_indices.clip(min=0)].copy()
        seg[segment_indices < 0] = 0
        chi = model.descriptor(torch.as_tensor(seg, device=device))
        mi = torch.as_tensor(self.marginal_index[indices].astype(np.int64), device=device)
        ji = torch.as_tensor(self.joint_index[indices].astype(np.int64), device=device)
        rm, rj, mv, jv = model.memory.references(mi, ji)
        pred = torch.as_tensor(self.prediction[indices], device=device)
        current = torch.as_tensor(self.z[indices], device=device)
        ready = torch.as_tensor(indices >= self.window, device=device)[:, None].expand_as(mi)
        return {"chi": chi, "prediction": pred, "residual": current[..., None] - pred,
                "marginal": rm, "joint": rj, "m_valid": mv, "j_valid": jv,
                "observed": torch.as_tensor(self.mask[indices], device=device), "prediction_ready": ready}

    def save(self, path, key):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with open(temporary, "wb") as stream:
            np.savez_compressed(stream, z=self.z, mask=self.mask, prediction=self.prediction,
                                marginal_index=self.marginal_index, joint_index=self.joint_index,
                                window=np.int64(self.window), labels=self.labels if self.labels is not None else [],
                                weights=self.weights if self.weights is not None else [], key=np.array(key))
        temporary.replace(path)

    @classmethod
    def load(cls, path, key=None):
        with np.load(path, allow_pickle=False) as data:
            if key is not None and data["key"].item() != key:
                raise ValueError(f"Stale feature cache: {path}. Use a new output directory.")
            return cls(data["z"], data["mask"], data["prediction"], data["marginal_index"],
                       data["joint_index"], int(data["window"]),
                       data["labels"] if data["labels"].size else None,
                       data["weights"] if data["weights"].size else None)


@torch.no_grad()
def extract_features(record: Record, model: NormalRepresentation, batch_size=64,
                     log_interval_seconds=0, progress_label="feature extraction batches") -> FeatureSequence:
    model.eval()
    device = next(model.parameters()).device
    n, c = record.z.shape
    w = model.cfg["window"]
    predictions = np.zeros((n, c, 2), np.float32)
    mi, ji = np.full((n, c), -1, np.int32), np.full(n, -1, np.int32)
    for left in periodic_progress(range(w, n, batch_size), progress_label, log_interval_seconds):
        indices = np.arange(left, min(left + batch_size, n))
        hidx = indices[:, None] + np.arange(-w, 0)
        segment, smask = segment_arrays(record, indices)
        pred = model.predict(torch.as_tensor(record.z[hidx], device=device),
                             torch.as_tensor(record.mask[hidx], device=device))
        chi = model.descriptor(torch.as_tensor(segment, device=device))
        im, ij = model.memory.query(chi, torch.as_tensor(smask.all(1), device=device), model.history.identity.weight)
        predictions[indices] = pred.cpu().numpy()
        mi[indices], ji[indices] = im.cpu().numpy(), ij.cpu().numpy()
    return FeatureSequence(record.z, record.mask, predictions, mi, ji, w)


@torch.no_grad()
def score_features(sequence: FeatureSequence, representation, detector, block_length=256, diagnostics=False,
                   log_interval_seconds=0, progress_label="scoring blocks"):
    detector.eval()
    n, c = sequence.z.shape
    scores, valid = np.full(n, np.nan, np.float32), np.zeros(n, bool)
    attribution = np.full(n, -1, np.int32)
    extra = {k: np.zeros((n, c), np.float32) for k in
             ("retention", "state_difference", "marginal_difference", "joint_difference")} if diagnostics else {}
    state = None
    for start in periodic_progress(range(0, n, block_length), progress_label, log_interval_seconds):
        block = sequence.block(start, min(n, start + block_length), representation)
        for t in range(len(block["chi"])):
            out, state = detector.step({k: v[t:t + 1] for k, v in block.items()}, state, diagnostics)
            i = start + t
            if out["valid"].item():
                scores[i] = out["score"].item()
                valid[i] = True
                attribution[i] = out["attribution"].item()
            for key in extra:
                extra[key][i] = out[key][0].cpu().numpy()
    return {"scores": scores, "valid": valid, "attribution": attribution, **extra}


class StreamingDetector:
    """One arriving raw sample -> one output; history and states persist across calls."""

    def __init__(self, representation, detector, scaler: Standardizer, threshold: float):
        self.representation, self.detector = representation.eval(), detector.eval()
        self.scaler, self.threshold = scaler, float(threshold)
        self.device = next(representation.parameters()).device
        self.reset()

    def reset(self):
        self.previous = self.scaler.mean.copy()
        self.history, self.masks = [], []
        self.state = None
        self.index = 0

    @torch.no_grad()
    def push(self, raw_sample, diagnostics=False):
        sample = np.asarray(raw_sample, dtype=np.float64)
        if sample.shape != self.scaler.mean.shape:
            raise ValueError("Incoming sample has an incorrect channel shape/order.")
        mask = np.isfinite(sample)
        self.previous = np.where(mask, sample, self.previous)
        z = ((self.previous - self.scaler.mean) / self.scaler.scale).astype(np.float32)
        w = self.representation.cfg["window"]
        result = {"index": self.index, "score": float("nan"), "valid": False, "alarm": False,
                  "attribution": -1}
        if len(self.history) == w:
            history = torch.as_tensor(np.array(self.history)[None], device=self.device)
            hmask = torch.as_tensor(np.array(self.masks)[None], device=self.device)
            pred = self.representation.predict(history, hmask)
            segment = np.concatenate([np.array(self.history)[-31:], z[None]])
            smask = np.concatenate([np.array(self.masks)[-31:], mask[None]])
            chi = self.representation.descriptor(torch.as_tensor(segment[None], device=self.device))
            mi, ji = self.representation.memory.query(chi, torch.as_tensor(smask.all(0)[None], device=self.device),
                                                       self.representation.history.identity.weight)
            rm, rj, mv, jv = self.representation.memory.references(mi, ji)
            observed = torch.as_tensor(mask[None], device=self.device)
            f = {"chi": chi, "prediction": pred, "residual": torch.as_tensor(z[None, :, None], device=self.device) - pred,
                 "marginal": rm, "joint": rj, "m_valid": mv, "j_valid": jv,
                 "observed": observed, "prediction_ready": torch.ones_like(observed)}
            out, self.state = self.detector.step(f, self.state, diagnostics)
            if out["valid"].item():
                result.update(score=out["score"].item(), valid=True,
                              alarm=out["score"].item() > self.threshold,
                              attribution=out["attribution"].item())
            if diagnostics:
                result.update({k: v[0].cpu().numpy() for k, v in out.items()
                               if k not in {"score", "valid", "attribution"}})
        self.history.append(z)
        self.masks.append(mask)
        if len(self.history) > w:
            self.history.pop(0)
            self.masks.pop(0)
        self.index += 1
        return result

    def process(self, samples):
        return [self.push(sample) for sample in samples]
