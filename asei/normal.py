"""Normal-only forecaster, real-window dictionaries and frozen state dynamics."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .compat import transformer_encoder
from .state import ContractiveCell
from .progress import Progress
from .resources import cooperate
from .features import DescriptorSchema


class TypedForecaster(nn.Module):
    def __init__(self, schema, config):
        super().__init__()
        # Schema belongs to NormalModel; avoid duplicate state_dict subtrees.
        object.__setattr__(self, "schema", schema)
        self.window, self.channels = config["window"], schema.channels
        d, q = config["d_model"], schema.q
        self.identity = nn.Embedding(schema.channels, d)
        self.symbols = nn.Embedding(schema.channels * q, 8)
        self.numeric = nn.ModuleList(nn.Linear(2 * p, d) for p in (1, 8, 32))
        self.categorical = nn.ModuleList(nn.Linear(9 * p, d) for p in (1, 8, 32))
        self.position, self.scale = nn.Embedding(self.window, d), nn.Embedding(3, d)
        endpoints = torch.cat([torch.arange(self.window - 32, self.window),
                               torch.arange(7, self.window, 8), torch.arange(31, self.window, 32)])
        self.register_buffer("endpoints", endpoints)
        self.register_buffer("causal_mask", endpoints[None] > endpoints[:, None])
        layer = nn.TransformerEncoderLayer(d, config["heads"], 4 * d, config.get("dropout", .1),
                                          activation="gelu", batch_first=True, norm_first=True)
        self.encoder = transformer_encoder(layer, config["layers"], nn.LayerNorm(d))
        self.self_num, self.rel_num = nn.Linear(d, 1), nn.Linear(2 * d, 1)
        self.self_cat, self.rel_cat = nn.Linear(d, q), nn.Linear(2 * d, q)
        initial = torch.full((schema.channels, schema.channels), .1 / max(1, schema.channels - 1))
        initial.fill_diagonal_(0)
        self.relation = nn.Parameter(initial)
        self.register_buffer("off_diagonal", ~torch.eye(schema.channels, dtype=torch.bool))

    def weights(self):
        a = self.relation * self.off_diagonal
        return a / a.abs().sum(-1, keepdim=True).clamp(min=1)

    @torch.no_grad()
    def project(self):
        self.relation.copy_(self.weights())

    def forward(self, history, observed):
        b, w, c = history.shape
        if w != self.window or c != self.channels:
            raise ValueError("History does not match the fitted forecaster.")
        values, masks = history.transpose(1, 2), observed.transpose(1, 2).float()
        ids = self.schema.ids(history).transpose(1, 2)
        offsets = torch.arange(c, device=history.device)[None, :, None] * self.schema.q
        embedded = self.symbols(ids + offsets)
        tokens = []
        for s, p in enumerate((1, 8, 32)):
            x = values[..., -32:, None] if p == 1 else values.unfold(-1, p, p)
            m = masks[..., -32:, None] if p == 1 else masks.unfold(-1, p, p)
            numeric = self.numeric[s](torch.cat([x, m], -1))
            if p == 1:
                state = embedded[..., -32:, :]
            else:
                state = embedded.reshape(b, c, w // p, p * 8)
            categorical = self.categorical[s](torch.cat([state, m], -1))
            token = torch.where(self.schema.categorical[None, :, None, None], categorical, numeric)
            tokens.append(token + self.scale.weight[s])
        x = torch.cat(tokens, 2) + self.position(self.endpoints)[None, None] + self.identity.weight[None, :, None]
        hidden = self.encoder(x.reshape(b * c, x.shape[2], -1), mask=self.causal_mask)[:, 31].reshape(b, c, -1)
        a = self.weights()
        peers = torch.einsum("ij,bjd->bid", a, hidden)
        relative = torch.cat([peers, self.identity.weight[None].expand(b, -1, -1)], -1)
        numeric = torch.stack([self.self_num(hidden).squeeze(-1), self.rel_num(relative).squeeze(-1)], -1)
        logits = torch.stack([self.self_cat(hidden), self.rel_cat(relative)], -2)
        logits = logits.masked_fill(~self.schema.allowed[None, :, None], -1e9)
        available = observed.any(1)
        peer_available = torch.einsum("ij,bj->bi", a.abs(), available.float()) > 0
        return numeric, logits, torch.stack([available, peer_available], -1)


class NormalTeacher(nn.Module):
    def __init__(self, schema, eta=.2, rho=.95):
        super().__init__()
        object.__setattr__(self, "schema", schema)
        self.identity = nn.Embedding(schema.channels, 8)
        self.num_adapter = nn.Sequential(nn.Linear(10, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh())
        self.cat_adapter = nn.Sequential(nn.Linear(2 * schema.q + 8, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh())
        self.cell = ContractiveCell(32, eta, rho)
        self.current_num = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 38))
        self.next_num = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 38))
        self.current_cat = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32 * schema.q))
        self.next_cat = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32 * schema.q))
        self.next_pair = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, schema.q ** 2))
        self._prototype_cache = None

    def prototype_inputs(self, memory):
        """Encode each possible reference once, retaining adapter gradients in training."""
        key = (self.identity.weight.device, self.identity.weight.dtype,
               tuple(p._version for p in self.parameters()), memory.m_num._version,
               memory.j_num._version, memory.m_tokens._version, memory.j_tokens._version)
        reusable = not torch.is_grad_enabled()
        if reusable and self._prototype_cache is not None and self._prototype_cache[0] == key:
            return self._prototype_cache[1]
        marginal = self.adapt(memory.m_num[..., :2].transpose(0, 1), memory.m_tokens[..., -2:].transpose(0, 1))
        joint = self.adapt(memory.j_num[..., :2], memory.j_tokens[..., -2:])
        values = (marginal, joint, self.cell.input(marginal), self.cell.input(joint))
        if reusable:
            self._prototype_cache = (key, values)
        return values

    def reference_input(self, index, tables):
        marginal, joint, marginal_projected, joint_projected = tables
        channels = torch.arange(self.schema.channels, device=index.device)
        use_joint = index >= 0
        ji = index.clamp(min=0, max=joint.shape[0] - 1)
        mi = (-index - 1).clamp(min=0, max=marginal.shape[0] - 1)
        adapted = torch.where(use_joint[..., None], joint[ji, channels], marginal[mi, channels])
        projected = torch.where(use_joint[..., None], joint_projected[ji, channels], marginal_projected[mi, channels])
        return adapted, projected

    def adapt(self, v, pair):
        identity = self.identity.weight.expand(*v.shape[:-2], -1, -1)
        if not self.schema.cat_indices:
            return self.num_adapter(torch.cat([v, identity], -1))
        if not self.schema.num_indices:
            return self.cat_adapter(torch.cat([F.one_hot(pair, self.schema.q).float().flatten(-2), identity], -1))
        num = self.num_adapter(torch.cat([v, identity], -1))
        cat = self.cat_adapter(torch.cat([F.one_hot(pair, self.schema.q).float().flatten(-2), identity], -1))
        return torch.where(self.schema.categorical[..., None], cat, num)

    def prior(self, state):
        numeric = self.next_num(state)[..., :2]
        if not self.schema.cat_indices:
            return numeric, state.new_empty(0)
        pair = self.next_pair(state).reshape(*state.shape[:-1], self.schema.q, self.schema.q)
        allowed = self.schema.allowed[:, :, None] & self.schema.allowed[:, None, :]
        return numeric, F.log_softmax(pair.masked_fill(~allowed, -1e9).flatten(-2), -1).reshape_as(pair)

    def losses(self, states, current, following):
        """Return per-target losses, with macro averaging handled by the trainer."""
        num_now = self.current_num(states)
        num_next = self.next_num(states)
        def continuous(predicted, target):
            difference = F.huber_loss(predicted, target["chi"], reduction="none")
            return .5 * (difference[..., :6].mean(-1) + difference[..., 6:].mean(-1))
        now, future = continuous(num_now, current), continuous(num_next, following)
        if self.schema.cat_indices:
            shape = (*states.shape[:-1], 32, self.schema.q)
            allowed = self.schema.allowed[None, None, :, None, :]
            cat_now = self.current_cat(states).reshape(shape).masked_fill(~allowed, -1e9)
            cat_next = self.next_cat(states).reshape(shape).masked_fill(~allowed, -1e9)
            ce_now = -F.log_softmax(cat_now, -1).gather(-1, current["tokens"][..., None]).squeeze(-1).mean(-1)
            ce_next = -F.log_softmax(cat_next, -1).gather(-1, following["tokens"][..., None]).squeeze(-1).mean(-1)
            _, pair_logits = self.prior(states)
            pair_id = following["pair"][..., 0] * self.schema.q + following["pair"][..., 1]
            pair_loss = -pair_logits.flatten(-2).gather(-1, pair_id[..., None]).squeeze(-1)
            now = torch.where(self.schema.categorical, ce_now, now)
            future = torch.where(self.schema.categorical, .5 * (ce_next + pair_loss), future)
        return now, future


class NormalMemory(nn.Module):
    def __init__(self, schema, km, kj, chunk=32):
        super().__init__()
        object.__setattr__(self, "schema", schema)
        c = schema.channels
        self.chunk = int(chunk)
        self.register_buffer("m_num", torch.zeros(c, km, 38))
        self.register_buffer("m_tokens", torch.zeros(c, km, 32, dtype=torch.long))
        self.register_buffer("m_active", torch.zeros(c, km, dtype=torch.bool))
        self.register_buffer("j_num", torch.zeros(kj, c, 38))
        self.register_buffer("j_tokens", torch.zeros(kj, c, 32, dtype=torch.long))
        self.register_buffer("j_active", torch.zeros(kj, dtype=torch.bool))
        self.register_buffer("peer_scale", torch.ones(c))
        self.register_buffer("energy_scale", torch.ones(c, 2, 2))  # short/long, marginal/joint
        self.register_buffer("reference_limit", torch.full((c,), float("inf")))

    @staticmethod
    def _representatives(x, count, seed, priority=None, rare_fraction=.25):
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import pairwise_distances_argmin_min
        count = min(count, len(x))
        if count == len(x):
            return np.arange(count)
        reserved = (np.argsort(-priority, kind="stable")[:int(count * rare_fraction)].tolist()
                    if priority is not None else [])
        estimator = MiniBatchKMeans(max(1, count - len(reserved)), batch_size=min(1024, len(x)), n_init=3, max_iter=50, random_state=seed)
        estimator.fit(x)
        indices = pairwise_distances_argmin_min(estimator.cluster_centers_, x)[0]
        selected = list(dict.fromkeys(reserved + indices.tolist()))[:count]
        remaining = np.setdiff1d(np.arange(len(x)), selected)
        selected.extend(np.random.default_rng(seed).permutation(remaining)[:count - len(selected)].tolist())
        return np.asarray(selected)

    @torch.no_grad()
    def fit(self, record, seed=42, pool_size=8192, rare_fraction=.25, log_interval_seconds=30):
        from .tasks import normal_difficulty
        rng = np.random.default_rng(seed)
        candidates = np.arange(31, len(record.z))
        priority = normal_difficulty(record, self.schema.metadata["supports"])
        capacity = min(pool_size, len(candidates))
        rare = candidates[np.argsort(-priority[candidates], kind="stable")[:int(capacity * rare_fraction)]]
        remaining = np.setdiff1d(candidates, rare)
        indices = np.r_[rare, rng.choice(remaining, size=capacity - len(rare), replace=False)]
        indices.sort()
        f = self.schema.block([record] * len(indices), indices.tolist(), 1)
        f = {k: v[:, 0] for k, v in f.items()}
        progress = Progress("memory marginal channels", self.schema.channels, log_interval_seconds)
        for i in range(self.schema.channels):
            cooperate()
            valid = torch.nonzero(f["lv"][:, i], as_tuple=False).flatten()
            if not len(valid):
                progress.update(i + 1)
                continue
            embedding = self.schema.cluster_features({k: v[valid] for k, v in f.items()}, i).cpu().numpy()
            local_priority = priority[indices[valid.cpu().numpy()]]
            if i in self.schema.cat_indices:
                pairs = f["pair"][valid, i].cpu().numpy()
                _, first, inverse, counts = np.unique(pairs, axis=0, return_index=True, return_inverse=True, return_counts=True)
                local_priority = 1 / counts[inverse]
                local_priority[first] += 2  # Prefer distinct rare transitions before duplicate windows.
            chosen = valid[torch.as_tensor(self._representatives(embedding, self.m_num.shape[1], seed + i,
                                           local_priority, rare_fraction), device=valid.device)]
            self.m_num[i, :len(chosen)] = f["chi"][chosen, i]
            self.m_tokens[i, :len(chosen)] = f["tokens"][chosen, i]
            self.m_active[i, :len(chosen)] = True
            progress.update(i + 1)
        valid = torch.nonzero(f["lv"].all(-1), as_tuple=False).flatten()
        if len(valid):
            embedding = self.schema.cluster_features({k: v[valid] for k, v in f.items()}).cpu().numpy()
            chosen = valid[torch.as_tensor(self._representatives(embedding, self.j_num.shape[0], seed,
                                           priority[indices[valid.cpu().numpy()]], rare_fraction), device=valid.device)]
            self.j_num[:len(chosen)] = f["chi"][chosen]
            self.j_tokens[:len(chosen)] = f["tokens"][chosen]
            self.j_active[:len(chosen)] = True
        for i in range(self.schema.channels):
            seen = f["sv"][:, i]
            if not seen.any():
                continue
            if i in self.schema.cat_indices:
                frequency = F.one_hot(f["pair"][seen, i], self.schema.q).float().mean(0)
                self.peer_scale[i] = (1 - frequency.square().sum(-1)).mean().clamp(min=1e-3)
            else:
                self.peer_scale[i] = f["v"][seen, i].var(0, unbiased=False).mean().clamp(min=1e-3)
        if not self.m_active.any():
            raise ValueError("No complete normal window for the dictionary.")

    def _distances(self, f, p_num, p_tokens, short=False):
        # Output [...,C,K]; prototype chunks bound the largest temporary.
        x = f["v"] if short else f["chi"]
        p = p_num[..., :2] if short else p_num
        if short:
            # Direct two-coordinate differences avoid cancellation near a large
            # prototype and keep instant energies consistent across batch sizes.
            distance = .5 * ((x[..., 0, None] - p[..., 0]).square()
                             + (x[..., 1, None] - p[..., 1]).square())
        else:
            weight = x.new_tensor([.5 / 6] * 6 + [.5 / 32] * 32)
            distance = ((x.square() * weight).sum(-1)[..., None] + (p.square() * weight).sum(-1)
                        - 2 * torch.einsum("...cd,ckd->...ck", x * weight, p)).clamp(min=0)
        if self.schema.cat_indices:
            ids = self.schema.cat_indices
            tokens = f["pair"][..., ids, :] if short else f["tokens"][..., ids, :]
            prototypes = p_tokens[ids, :, -2:] if short else p_tokens[ids]
            symbolic = (tokens[..., :, None, :] != prototypes).float().mean(-1)
            distance[..., ids, :] = symbolic
        return distance

    def long_evidence(self, f):
        if "m_index" in f:
            return self._long_from_indices(f, f["m_index"], f["j_index"][..., 0])
        shape = f["sv"].shape
        m_best = f["v"].new_full(shape, float("inf"))
        m_index = torch.zeros(shape, dtype=torch.long, device=m_best.device)
        for start in range(0, self.m_num.shape[1], self.chunk):
            end = min(start + self.chunk, self.m_num.shape[1])
            d = self._distances(f, self.m_num[:, start:end], self.m_tokens[:, start:end])
            value, index = d.masked_fill(~self.m_active[:, start:end], float("inf")).min(-1)
            improve = value < m_best
            m_index = torch.where(improve, index + start, m_index)
            m_best = torch.minimum(m_best, value)
        j_best = f["v"].new_full(shape[:-1], float("inf"))
        j_index = torch.zeros(shape[:-1], dtype=torch.long, device=m_best.device)
        for start in range(0, self.j_num.shape[0], self.chunk):
            end = min(start + self.chunk, self.j_num.shape[0])
            d = self._distances(f, self.j_num[start:end].transpose(0, 1), self.j_tokens[start:end].transpose(0, 1))
            value, index = d.mean(-2).masked_fill(~self.j_active[start:end], float("inf")).min(-1)
            improve = value < j_best
            j_index = torch.where(improve, index + start, j_index)
            j_best = torch.minimum(j_best, value)
        return self._long_from_indices(f, m_index, j_index)

    def _long_from_indices(self, f, m_index, j_index):
        channels = torch.arange(self.schema.channels, device=f["v"].device)
        mn, mt = self.m_num[channels, m_index], self.m_tokens[channels, m_index]
        jn, jt = self.j_num[j_index], self.j_tokens[j_index]
        mv = f["lv"] & self.m_active.any(-1)
        jv = f["lv"].all(-1, keepdim=True).expand_as(mv) & self.j_active.any()
        def discrepancy(num, tok):
            result = f["chi"] - num
            if self.schema.cat_indices:
                i = self.schema.cat_indices
                result[..., i, :6] = 0
                result[..., i, 6:] = (f["tokens"][..., i, :] != tok[..., i, :]).float()
            return result
        md, jd = discrepancy(mn, mt), discrepancy(jn, jt)
        weights = f["v"].new_tensor([.5 / 6] * 6 + [.5 / 32] * 32)
        dm, dj = (md.square() * weights).sum(-1), (jd.square() * weights).sum(-1)
        if self.schema.cat_indices:
            dm[..., self.schema.cat_indices] *= 2
            dj[..., self.schema.cat_indices] *= 2
        return dict(md=md.masked_fill(~mv[..., None], 0), jd=jd.masked_fill(~jv[..., None], 0),
                    long_dist=torch.stack([dm.masked_fill(~mv, 0), dj.masked_fill(~jv, 0)], -1),
                    mv=mv, jv=jv, m_index=m_index,
                    j_index=j_index[..., None].expand_as(m_index))

    @staticmethod
    def excluded_trim(cost, valid, fraction):
        """Sort peers once: O(B*K*C log C), no B*K*C*C tensor."""
        # Double precision only for reductions; recurrent and feature math stay FP32.
        x = cost.double().masked_fill(~valid[:, None, :], float("inf"))
        ordered, order = x.sort(dim=-1, stable=True)
        ordered = torch.where(torch.isfinite(ordered), ordered, torch.zeros_like(ordered))
        ranks = torch.empty_like(order).scatter_(-1, order, torch.arange(cost.shape[-1], device=cost.device).expand_as(order))
        peer_count = valid.sum(-1, keepdim=True) - valid.long()
        count = torch.ceil(peer_count * (1 - fraction)).long().clamp(min=0)
        count = count[:, None, :].expand_as(ranks)
        prefix = F.pad(ordered.cumsum(-1), (1, 0))
        base = prefix.gather(-1, count)
        # Removing a selected target uses sums on either side, not total-target.
        # q can differ for missing targets; they are reset and do not affect scoring.
        valid_q = torch.ceil((valid.sum(-1) - 1).clamp(min=0) * (1 - fraction)).long()[:, None, None]
        positions = torch.arange(cost.shape[-1], device=cost.device)[None, None, :]
        selected = torch.where(positions <= valid_q, ordered, torch.zeros_like(ordered))
        suffix = F.pad(selected.flip(-1).cumsum(-1).flip(-1), (0, 1))
        removed = prefix.gather(-1, ranks) + suffix.gather(-1, (ranks + 1).clamp(max=cost.shape[-1]))
        trimmed = torch.where(ranks < count, removed, base) / count.clamp(min=1)
        # Independent prefix/suffix sums preserve target exclusion in tie-breaking too.
        finite = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        full = F.pad(finite.cumsum(-1), (1, 0))[..., :-1] + F.pad(finite.flip(-1).cumsum(-1).flip(-1), (0, 1))[..., 1:]
        return trimmed, full / peer_count[:, None].clamp(min=1), peer_count

    def short_marginal(self, f):
        if "short_marginal" in f:
            return f["short_marginal"]
        best = f["v"].new_full(f["sv"].shape, float("inf"))
        for start in range(0, self.m_num.shape[1], self.chunk):
            end = start + self.chunk
            d = self._distances(f, self.m_num[:, start:end], self.m_tokens[:, start:end], short=True)
            best = torch.minimum(best, d.masked_fill(~self.m_active[:, start:end], float("inf")).min(-1).values)
        return best.masked_fill(~self.m_active.any(-1), 0)

    def prepare_peers(self, f, trim=.1):
        """Batch time-independent retrieval; state-dependent priors stay causal."""
        shape = f["sv"].shape
        pn, pt = self.j_num.transpose(0, 1), self.j_tokens.transpose(0, 1)
        d = self._distances(f, pn, pt, short=True) / self.peer_scale[:, None]
        peer, secondary, counts = self.excluded_trim(
            d.reshape(-1, self.schema.channels, self.j_num.shape[0]).transpose(1, 2),
            f["sv"].reshape(-1, self.schema.channels), trim)
        return dict(peer=peer.reshape(*shape[:-1], self.j_num.shape[0], shape[-1]),
                    secondary=secondary.reshape(*shape[:-1], self.j_num.shape[0], shape[-1]),
                    counts=counts.reshape(shape))

    def lookup_reference(self, index):
        channels = torch.arange(self.schema.channels, device=index.device)
        joint = index >= 0
        ji = index.clamp(min=0)
        mi = (-index - 1).clamp(min=0)
        # Both arms are gathered, so clamp their unused indices to their own table.
        ji = ji.clamp(max=self.j_num.shape[0] - 1)
        mi = mi.clamp(max=self.m_num.shape[1] - 1)
        num = torch.where(joint[..., None], self.j_num[ji, channels, :2], self.m_num[channels, mi, :2])
        pair = torch.where(joint[..., None], self.j_tokens[ji, channels, -2:], self.m_tokens[channels, mi, -2:])
        available = joint | self.m_active.any(-1)
        return num, pair, available, joint

    def reference(self, f, prior_num, prior_pair, trim=.1, prior_weight=.25, external=True,
                  prepared=None, return_index=False, return_support=False):
        # f is one time point: [B,C,...]. All routing excludes actual target input.
        prepared = self.prepare_peers(f, trim) if prepared is None else prepared
        peer, secondary, counts = prepared["peer"], prepared["secondary"], prepared["counts"]
        nprior = .5 * (prior_num[:, None] - self.j_num[None, :, :, :2]).square().sum(-1) / self.peer_scale
        if self.schema.cat_indices:
            pairs = self.j_tokens[..., -2:]
            ids = (pairs[..., 0] * self.schema.q + pairs[..., 1]).transpose(0, 1)
            pprior = -prior_pair.flatten(-2).gather(-1, ids[None].expand(len(prior_num), -1, -1)).transpose(1, 2)
            nprior = torch.where(self.schema.categorical, pprior, nprior)
        cost = peer + prior_weight * nprior.double()
        cost = cost.masked_fill(~self.j_active[None, :, None], float("inf"))
        minimum = cost.min(1, keepdim=True).values
        tied = cost == minimum
        tie_cost = secondary.masked_fill(~tied, float("inf"))
        index = tie_cost.argmin(1)
        channels = torch.arange(self.schema.channels, device=cost.device)
        joint_num = self.j_num[index, channels, :2]
        joint_pair = self.j_tokens[index, channels, -2:]
        joint_valid = (counts > 0) & self.j_active.any() & external
        md = .5 * (prior_num[:, :, None] - self.m_num[None, :, :, :2]).square().sum(-1)
        if self.schema.cat_indices:
            mid = self.m_tokens[..., -2] * self.schema.q + self.m_tokens[..., -1]
            categorical = -prior_pair.flatten(-2).gather(-1, mid[None].expand(len(prior_num), -1, -1))
            md = torch.where(self.schema.categorical[:, None], categorical, md)
        mi = md.masked_fill(~self.m_active, float("inf")).argmin(-1)
        marginal_num, marginal_pair = self.m_num[channels, mi, :2], self.m_tokens[channels, mi, -2:]
        ref_num = torch.where(joint_valid[..., None], joint_num, marginal_num)
        ref_pair = torch.where(joint_valid[..., None], joint_pair, marginal_pair)
        available = joint_valid | self.m_active.any(-1)
        result = (ref_num, ref_pair, available, joint_valid)
        if return_index:
            result = (*result, torch.where(joint_valid, index, -mi - 1))
        if return_support:
            support = peer.gather(1, index[:, None]).squeeze(1).float()
            result = (*result, support.masked_fill(~joint_valid, 0))
        return result

    def short_evidence(self, f, ref_num, ref_pair, reference_valid, joint_valid, peer_cost=None):
        marginal = self.short_marginal(f)
        relative = .5 * (f["v"] - ref_num).square().sum(-1)
        if self.schema.cat_indices:
            relative[..., self.schema.cat_indices] = (f["pair"][..., self.schema.cat_indices, :] != ref_pair[..., self.schema.cat_indices, :]).float().mean(-1)
        mv = self.m_active.any(-1) & f["sv"]
        jv = reference_valid & f["sv"]
        if peer_cost is not None:
            # Unsupported external contexts still drive the dynamic branch and
            # normal trajectory. They cannot force an unconditional instant alarm.
            jv &= ~joint_valid | (peer_cost <= self.reference_limit)
        distance = torch.stack([marginal.masked_fill(~mv, 0), relative.masked_fill(~jv, 0)], -1)
        available = torch.stack([mv, jv], -1)
        return distance, available


class NormalModel(nn.Module):
    def __init__(self, metadata, config):
        super().__init__()
        self.metadata, self.config = metadata, config
        self.schema = DescriptorSchema(metadata)
        self.predictor = TypedForecaster(self.schema, config["model"])
        self.memory = NormalMemory(self.schema, config["model"]["marginal_prototypes"], config["model"]["joint_prototypes"],
                                      config.get("memory", {}).get("prototype_chunk", 32))
        self.teacher = NormalTeacher(self.schema, config["stage2"].get("eta", .2), config["stage2"].get("rho", .95))

    def freeze(self):
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self
