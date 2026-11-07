"""Small forward/backward checks shared by environment validation tools."""
from __future__ import annotations


def model_training_probe(device, seed=42, settings=None):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from asei.config import load_config, resolve_config, DEFAULT_CONFIG
    from asei.data import Record
    from asei.features import DescriptorSchema
    from asei.normal import NormalModel
    from asei.model import DualTrajectoryDetector

    torch.manual_seed(seed)
    cfg = load_config(DEFAULT_CONFIG)
    cfg["model"].update(window=32, d_model=8, heads=2, layers=1, dropout=0., marginal_prototypes=4, joint_prototypes=4)
    if settings:
        cfg["stage2"].update(settings)
    cfg = resolve_config(cfg, 96, 3)
    z = np.random.default_rng(seed).normal(size=(96, 3)).astype(np.float32)
    z[:, 2] = (np.arange(96) // 7) % 2
    record = Record(z, np.ones_like(z, bool))
    schema = DescriptorSchema.fit(record)
    normal = NormalModel(schema.metadata, cfg).to(device)
    x = torch.as_tensor(z[:32][None], device=device)
    predicted, logits, available = normal.predictor(x, torch.ones_like(x, dtype=torch.bool))
    target = torch.as_tensor(z[32][None], device=device)
    ids = normal.schema.ids(target)
    ce = -F.log_softmax(logits, -1).gather(-1, ids[..., None, None].expand(-1, -1, 2, 1)).squeeze(-1)
    loss = torch.where(normal.schema.categorical[None, :, None], ce,
                       F.huber_loss(predicted, target[..., None].expand_as(predicted), reduction="none"))
    loss = (loss * available).sum() / available.sum().clamp(min=1)
    loss.backward()

    def gradients(module, value):
        found = [p.grad for p in module.parameters() if p.grad is not None]
        if not torch.isfinite(value) or not found or not all(torch.isfinite(g).all() for g in found):
            raise RuntimeError("Invalid loss or gradients in model probe.")
        return len(found)

    stage1_gradients = gradients(normal.predictor, loss)
    normal.memory.fit(record, seed=seed, pool_size=48)
    model = DualTrajectoryDetector(normal, cfg["stage2"]).to(device)
    normal.teacher.requires_grad_(True)
    current = normal.schema.block([record], [32], 16)
    following = normal.schema.block([record], [33], 16)
    hidden, _ = model.teacher_block(current)
    closed, _ = model.teacher_block(current, closed_loop=True)
    now, future = normal.teacher.losses(hidden, current, following)
    cn, cf = normal.teacher.losses(closed, current, following)
    teacher_loss = (now + future + cn + cf).mean() / 4
    teacher_loss.backward()
    teacher_gradients = gradients(normal.teacher, teacher_loss)
    normal.teacher.cell.project()
    normal.freeze().zero_grad(set_to_none=True)
    model.train()
    output, _ = model.forward_block(current)
    stage2_loss = F.binary_cross_entropy_with_logits(output["dynamic"], torch.ones_like(output["dynamic"]))
    stage2_loss.backward()
    stage2_gradients = gradients(model, stage2_loss)
    if any(p.grad is not None for p in normal.parameters()):
        raise RuntimeError("Detection gradients reached frozen normal parameters.")
    return dict(stage1_loss=float(loss.detach().cpu()), teacher_loss=float(teacher_loss.detach().cpu()),
                stage2_loss=float(stage2_loss.detach().cpu()), stage1_gradient_tensors=stage1_gradients,
                teacher_gradient_tensors=teacher_gradients, stage2_gradient_tensors=stage2_gradients)
