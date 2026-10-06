"""
Fusion heads (spec 05)
-----------------------
Every head sees the same encoder outputs and returns
    {"logit": [B], "aux_loss": scalar, "stats": dict}
from forward(e_G, e_S, ctx, evidence):

    e_G       graph part   [u ‖ v ‖ u⊙v]            [B, 3d]
    e_S       sequence part s_seq                    [B, 2d]   (None with --no-seq)
    ctx       context categorical embeddings ‖ numeric
    evidence  per-view causal evidence (n_G, n_S)    [B, 2]

Heads: concat (today's MLP, bitwise identical), gate, evgate, moe, misa,
sparse (shared/private top-k dictionaries with a per-view evidence budget).
Dataset-agnostic: evidence comes precomputed from features.view_evidence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

FUSIONS = ("concat", "gate", "evgate", "moe", "misa", "sparse")


def mlp_head(in_dim: int, dropout: float) -> list[nn.Module]:
    """[256, 128] LayerNorm + ReLU + dropout MLP → 1 logit (the spec 03 head)."""
    layers, prev = [], in_dim
    for h in (256, 128):
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return layers


def _zero(ref: Tensor) -> Tensor:
    return torch.zeros((), device=ref.device, dtype=ref.dtype)


def info_nce(z1: Tensor, z2: Tensor, temperature: float) -> Tensor:
    z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / temperature
    return F.cross_entropy(logits, torch.arange(len(z1), device=z1.device))


def cross_cov_penalty(a: Tensor, b: Tensor) -> Tensor:
    """Linear-kernel HSIC: squared Frobenius norm of the batch cross-covariance / (m_a·m_b)."""
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    c = a.t() @ b / max(len(a) - 1, 1)
    return c.pow(2).sum() / (a.shape[1] * b.shape[1])


# ── concat (= arm N4's head) ──────────────────────────────────────────────────

class ConcatFusion(nn.Sequential):
    """
    Today's head, unchanged: MLP([e_G ‖ e_S ‖ ctx]).  Subclasses nn.Sequential so its
    parameters keep the pre-spec-05 names (head.0.weight, …): older checkpoints load.
    """

    def __init__(self, in_dim: int, dropout: float) -> None:
        super().__init__(*mlp_head(in_dim, dropout))

    def fuse(self, e_G, e_S, ctx, evidence) -> dict:
        parts = [e for e in (e_G, e_S) if e is not None] + [ctx]
        logit = super().forward(torch.cat(parts, dim=-1)).squeeze(-1)
        return {"logit": logit, "aux_loss": _zero(logit), "stats": {}}


# ── shared projection helper ──────────────────────────────────────────────────

class _Projected(nn.Module):
    """p_v = LayerNorm(W_v e_v) for both views at fusion_dim."""

    def __init__(self, g_dim: int, s_dim: int, fdim: int) -> None:
        super().__init__()
        self.proj_G = nn.Sequential(nn.Linear(g_dim, fdim), nn.LayerNorm(fdim))
        self.proj_S = nn.Sequential(nn.Linear(s_dim, fdim), nn.LayerNorm(fdim))

    def project(self, e_G: Tensor, e_S: Tensor) -> tuple[Tensor, Tensor]:
        return self.proj_G(e_G), self.proj_S(e_S)


class _EvidenceNorm:
    """
    Normalised per-view evidence s_v = clip(n_v / n_ref,v, 0, 1) (spec 05 B4), shared by
    sparse and evgate.  n_ref (95th percentile over training rows) is a checkpointed
    buffer; training-mode forward refuses to run until set_n_ref has been called.
    """

    def _init_n_ref(self) -> None:
        self.register_buffer("n_ref", torch.ones(2))
        self.register_buffer("n_ref_set", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def set_n_ref(self, ref) -> None:
        ref = torch.as_tensor(ref, dtype=torch.float32).clamp_min(1e-6)
        self.n_ref.copy_(ref.to(self.n_ref.device))
        self.n_ref_set.fill_(True)

    def scaled_evidence(self, evidence: Tensor) -> Tensor:
        if self.training and not bool(self.n_ref_set):
            raise RuntimeError(f"{type(self).__name__}: n_ref was never set; call set_n_ref() "
                               "with the training-row 95th percentiles before training")
        return (evidence / self.n_ref.clamp_min(1e-6)).clamp(0, 1)


# ── gate / evgate ─────────────────────────────────────────────────────────────

class GateFusion(_EvidenceNorm, _Projected):
    """
    g = σ(W[p_G ‖ p_S]); z = [g⊙p_G + (1−g)⊙p_S ‖ ctx].
    evidence=True (evgate): g = σ(MLP([p_G ‖ p_S ‖ s_G ‖ s_S])) with the same normalised
    evidence s_v as sparse.
    """

    def __init__(self, g_dim, s_dim, ctx_dim, fdim, dropout, evidence: bool = False,
                 gate_hidden: int = 64) -> None:
        super().__init__(g_dim, s_dim, fdim)
        self.use_evidence = evidence
        if evidence:
            self.gate = nn.Sequential(nn.Linear(2 * fdim + 2, gate_hidden), nn.ReLU(),
                                      nn.Linear(gate_hidden, fdim))
        else:
            self.gate = nn.Linear(2 * fdim, fdim)
        self.head = nn.Sequential(*mlp_head(fdim + ctx_dim, dropout))
        if evidence:
            self._init_n_ref()

    def fuse(self, e_G, e_S, ctx, evidence) -> dict:
        p_G, p_S = self.project(e_G, e_S)
        gin = [p_G, p_S] + ([self.scaled_evidence(evidence)] if self.use_evidence else [])
        g = torch.sigmoid(self.gate(torch.cat(gin, dim=-1)))
        z = torch.cat([g * p_G + (1 - g) * p_S, ctx], dim=-1)
        logit = self.head(z).squeeze(-1)
        return {"logit": logit, "aux_loss": _zero(logit),
                "stats": {"gate_mean": g.mean().detach()}}


# ── moe ───────────────────────────────────────────────────────────────────────

class MoEFusion(_Projected):
    """Softmax router over expert MLPs on [p_G ‖ p_S ‖ ctx]; load-balancing loss."""

    def __init__(self, g_dim, s_dim, ctx_dim, fdim, dropout, experts: int = 4,
                 balance_weight: float = 0.01) -> None:
        super().__init__(g_dim, s_dim, fdim)
        in_dim = 2 * fdim + ctx_dim
        self.router = nn.Linear(in_dim, experts)
        self.experts = nn.ModuleList([nn.Sequential(*mlp_head(in_dim, dropout))
                                      for _ in range(experts)])
        self.balance_weight = balance_weight

    def fuse(self, e_G, e_S, ctx, evidence) -> dict:
        p_G, p_S = self.project(e_G, e_S)
        x = torch.cat([p_G, p_S, ctx], dim=-1)
        r = torch.softmax(self.router(x), dim=-1)                       # [B, E]
        out = torch.cat([e(x) for e in self.experts], dim=-1)           # [B, E]
        logit = (r * out).sum(-1)
        load = r.mean(0)
        balance = len(self.experts) * (load * load).sum()               # 1 when balanced
        return {"logit": logit, "aux_loss": self.balance_weight * balance,
                "stats": {"router_entropy": -(r * r.clamp_min(1e-9).log()).sum(-1).mean().detach()}}


# ── misa (dense shared/private) ───────────────────────────────────────────────

class MISAFusion(_Projected):
    """
    Dense shared/private decomposition: one shared encoder applied to both p_v,
    a private encoder per view; InfoNCE on shared, orthogonality shared⊥private
    per view, reconstruction of the detached p_v.
    """

    def __init__(self, g_dim, s_dim, ctx_dim, fdim, dropout, hdim: int = 128,
                 w_rec: float = 1.0, w_align: float = 0.1, w_dec: float = 0.1,
                 temperature: float = 0.2) -> None:
        super().__init__(g_dim, s_dim, fdim)
        self.shared = nn.Sequential(nn.Linear(fdim, hdim), nn.ReLU())
        self.private = nn.ModuleDict({v: nn.Sequential(nn.Linear(fdim, hdim), nn.ReLU())
                                      for v in ("G", "S")})
        self.decode = nn.ModuleDict({v: nn.Linear(2 * hdim, fdim) for v in ("G", "S")})
        self.head = nn.Sequential(*mlp_head(3 * hdim + ctx_dim, dropout))
        self.w_rec, self.w_align, self.w_dec, self.temperature = w_rec, w_align, w_dec, temperature

    def fuse(self, e_G, e_S, ctx, evidence) -> dict:
        p = dict(zip(("G", "S"), self.project(e_G, e_S)))
        sh = {v: self.shared(p[v]) for v in p}
        pr = {v: self.private[v](p[v]) for v in p}
        z = torch.cat([0.5 * (sh["G"] + sh["S"]), pr["G"], pr["S"], ctx], dim=-1)
        logit = self.head(z).squeeze(-1)
        aux = _zero(logit)
        if self.w_rec:
            aux = aux + self.w_rec * sum(
                (p[v].detach() - self.decode[v](torch.cat([sh[v], pr[v]], -1))).pow(2).sum(-1).mean()
                for v in p)
        if self.w_align:
            aux = aux + self.w_align * info_nce(sh["G"], sh["S"], self.temperature)
        if self.w_dec:   # orthogonality ‖H_shᵀ H_pr‖_F² per view (normalised by size)
            aux = aux + self.w_dec * sum(
                (sh[v].t() @ pr[v]).pow(2).sum() / (len(sh[v]) ** 2 * sh[v].shape[1]) for v in p)
        return {"logit": logit, "aux_loss": aux, "stats": {}}


# ── sparse shared/private (proposed) ──────────────────────────────────────────

class SparseSPFusion(_EvidenceNorm, _Projected):
    """
    Shared + view-private sparse codes over unit-norm dictionaries (spec 05 B).

    a_v = ReLU(E_v p_v + b_v) split into a_v^sh (m_shared) and a_v^pr (m_private).
    Top-k: k_sh fixed; k_pr,v = k_min + round((k_max − k_min)·clip(n_v / n_ref,v, 0, 1))
    (or a fixed k).  Head on [½(a_G^sh + a_S^sh) ‖ a_G^pr ‖ a_S^pr ‖ ctx].
    """

    def __init__(self, g_dim, s_dim, ctx_dim, fdim, dropout, m_shared: int = 256,
                 m_private: int = 256, k_shared: int = 16, k_private_min: int = 4,
                 k_private_max: int = 48, k_schedule: str = "adaptive", k_private_fixed: int = 24,
                 w_rec: float = 1.0, w_align: float = 0.1, w_dec: float = 0.1,
                 temperature: float = 0.2) -> None:
        super().__init__(g_dim, s_dim, fdim)
        self.m_sh, self.m_pr = m_shared, m_private
        self.k_sh, self.k_min, self.k_max = k_shared, k_private_min, k_private_max
        self.k_schedule, self.k_fixed = k_schedule, k_private_fixed
        self.w_rec, self.w_align, self.w_dec, self.temperature = w_rec, w_align, w_dec, temperature

        def unit(m):
            return nn.Parameter(F.normalize(torch.randn(m, fdim), dim=-1))
        self.D_sh = unit(m_shared)
        self.D_pr = nn.ParameterDict({"G": unit(m_private), "S": unit(m_private)})
        self.enc = nn.ModuleDict({v: nn.Linear(fdim, m_shared + m_private) for v in ("G", "S")})
        self.head = nn.Sequential(*mlp_head(m_shared + 2 * m_private + ctx_dim, dropout))
        # n_ref per view: 95th percentile of n_v over training rows (set by the trainer)
        self._init_n_ref()
        # firing bookkeeping for dead-atom resampling (steps since last fired)
        for name, m in (("idle_sh", m_shared), ("idle_G", m_private), ("idle_S", m_private)):
            self.register_buffer(name, torch.zeros(m, dtype=torch.long), persistent=False)
        self._last: dict = {}

    # schedule ---------------------------------------------------------------
    def k_private(self, evidence: Tensor) -> Tensor:
        """[B, 2] per-row private budgets for (G, S)."""
        if self.k_schedule == "fixed":
            return torch.full_like(evidence, self.k_fixed, dtype=torch.long)
        s = self.scaled_evidence(evidence)
        return (self.k_min + torch.round((self.k_max - self.k_min) * s)).long()

    @staticmethod
    def topk_mask(a: Tensor, k: Tensor | int) -> Tensor:
        """Keep the top-k entries per row (k scalar or [B]); gradients flow through kept entries."""
        if isinstance(k, int):
            idx = a.topk(k, dim=-1).indices
            return torch.zeros_like(a).scatter(-1, idx, 1.0) * a
        order = a.argsort(dim=-1, descending=True)
        rank = torch.empty_like(order).scatter_(-1, order, torch.arange(a.shape[-1], device=a.device)
                                                .expand_as(order))
        return a * (rank < k.unsqueeze(-1)).to(a.dtype)

    def codes(self, p: dict[str, Tensor], evidence: Tensor) -> tuple[dict, dict]:
        kp = self.k_private(evidence)
        sh, pr = {}, {}
        for i, v in enumerate(("G", "S")):
            a = F.relu(self.enc[v](p[v]))
            sh[v] = self.topk_mask(a[:, :self.m_sh], self.k_sh)
            pr[v] = self.topk_mask(a[:, self.m_sh:], kp[:, i])
        return sh, pr

    def fuse(self, e_G, e_S, ctx, evidence) -> dict:
        p = dict(zip(("G", "S"), self.project(e_G, e_S)))
        sh, pr = self.codes(p, evidence)
        z = torch.cat([0.5 * (sh["G"] + sh["S"]), pr["G"], pr["S"], ctx], dim=-1)
        logit = self.head(z).squeeze(-1)

        aux = _zero(logit)
        # Reconstruction: codes from the detached projection, so L_rec trains only the
        # encoders and dictionaries, never W_v or anything upstream of p_v
        pd_ = {v: p[v].detach() for v in p}
        sh_d, pr_d = self.codes(pd_, evidence)
        rec_err = {}
        for v in ("G", "S"):
            recon = sh_d[v] @ self.D_sh + pr_d[v] @ self.D_pr[v]
            rec_err[v] = (pd_[v] - recon).pow(2).sum(-1)
        if self.w_rec:
            aux = aux + self.w_rec * sum(e.mean() for e in rec_err.values())
        if self.w_align:
            aux = aux + self.w_align * info_nce(sh["G"], sh["S"], self.temperature)
        if self.w_dec:
            aux = aux + self.w_dec * cross_cov_penalty(pr["G"], pr["S"])

        l1 = lambda t: t.abs().sum(-1)
        rho = (l1(sh["G"]) + l1(sh["S"])) / (l1(sh["G"]) + l1(sh["S"]) + l1(pr["G"]) + l1(pr["S"])).clamp_min(1e-12)
        stats = {"rho": rho.detach(),
                 "active_sh_G": (sh["G"] > 0).sum(-1).detach(), "active_sh_S": (sh["S"] > 0).sum(-1).detach(),
                 "active_pr_G": (pr["G"] > 0).sum(-1).detach(), "active_pr_S": (pr["S"] > 0).sum(-1).detach()}
        self._last = {"p": {v: p[v].detach() for v in p}, "rec_err": {v: rec_err[v].detach() for v in p},
                      "fired": {"sh": ((sh["G"] > 0) | (sh["S"] > 0)).any(0),
                                "G": (pr["G"] > 0).any(0), "S": (pr["S"] > 0).any(0)},
                      "codes": {"sh_G": sh["G"].detach(), "sh_S": sh["S"].detach(),
                                "pr_G": pr["G"].detach(), "pr_S": pr["S"].detach()}}
        return {"logit": logit, "aux_loss": aux, "stats": stats}

    # training hooks ---------------------------------------------------------
    @torch.no_grad()
    def renormalize(self) -> None:
        """Unit-norm atoms (called after every optimiser step)."""
        for D in (self.D_sh, self.D_pr["G"], self.D_pr["S"]):
            D.data = F.normalize(D.data, dim=-1)

    @torch.no_grad()
    def track_and_resample(self, step: int, every: int = 1000, dead_after: int = 10_000,
                           optimizer: torch.optim.Optimizer | None = None) -> dict:
        """
        Update steps-since-fired per atom; every `every` steps re-initialise atoms idle for
        `dead_after` steps towards the residual of a random high-reconstruction-error row
        (encoder row reset to match, and the optimiser's moments for those rows zeroed).
        Returns dead fractions per dictionary.
        """
        if not self._last:
            return {}
        fired = self._last["fired"]
        for name, key in (("idle_sh", "sh"), ("idle_G", "G"), ("idle_S", "S")):
            idle = getattr(self, name)
            idle += 1
            idle[fired[key]] = 0
        dead = {k: float((getattr(self, n) >= dead_after).float().mean())
                for k, n in (("shared", "idle_sh"), ("G", "idle_G"), ("S", "idle_S"))}
        if step % every == 0:
            self.resample_dead(dead_after, optimizer)
        return dead

    @torch.no_grad()
    def resample_dead(self, dead_after: int, optimizer: torch.optim.Optimizer | None = None) -> int:
        n = 0

        def reset_state(param: Tensor, row: int) -> None:
            st = optimizer.state.get(param, {}) if optimizer is not None else {}
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key in st and st[key].dim() > 0:
                    st[key][row] = 0

        p, err = self._last["p"], self._last["rec_err"]
        for v in ("G", "S"):
            for block, idle_name in (("sh", "idle_sh"), (v, f"idle_{v}")):
                idle = getattr(self, idle_name)
                dead = torch.nonzero(idle >= dead_after).flatten()
                if block == "sh" and v == "S":
                    continue                     # shared atoms resampled once, from the G view
                for atom in dead.tolist():
                    hi = torch.topk(err[v], max(1, len(err[v]) // 10)).indices
                    row = hi[torch.randint(len(hi), (1,), device=hi.device)]
                    target = F.normalize(p[v][row].flatten(), dim=0)
                    D = self.D_sh if block == "sh" else self.D_pr[v]
                    D.data[atom] = target
                    reset_state(D, atom)
                    offset = 0 if block == "sh" else self.m_sh
                    for vv in (("G", "S") if block == "sh" else (v,)):
                        self.enc[vv].weight.data[offset + atom] = target * 0.1
                        self.enc[vv].bias.data[offset + atom] = 0.0
                        reset_state(self.enc[vv].weight, offset + atom)
                        reset_state(self.enc[vv].bias, offset + atom)
                    idle[atom] = 0
                    n += 1
        return n


def build_fusion(name: str, g_dim: int | None, s_dim: int | None, ctx_dim: int, args) -> nn.Module:
    """Construct the head selected by --fusion (and its variant flags)."""
    if name == "concat":
        return ConcatFusion((g_dim or 0) + (s_dim or 0) + ctx_dim, args.dropout)
    if g_dim is None or s_dim is None:
        raise ValueError(f"--fusion {name} needs both views; it cannot be combined with --no-graph/--no-seq")
    fd = args.fusion_dim
    if name == "gate":
        return GateFusion(g_dim, s_dim, ctx_dim, fd, args.dropout)
    if name == "evgate":
        return GateFusion(g_dim, s_dim, ctx_dim, fd, args.dropout, evidence=True,
                          gate_hidden=args.gate_hidden)
    if name == "moe":
        return MoEFusion(g_dim, s_dim, ctx_dim, fd, args.dropout, experts=args.moe_experts)
    if name == "misa":
        return MISAFusion(g_dim, s_dim, ctx_dim, fd, args.dropout, w_rec=args.w_rec,
                          w_align=args.w_align, w_dec=args.w_dec)
    if name == "sparse":
        return SparseSPFusion(g_dim, s_dim, ctx_dim, fd, args.dropout, m_shared=args.m_shared,
                              m_private=args.m_private, k_shared=args.k_shared,
                              k_private_min=args.k_private_min, k_private_max=args.k_private_max,
                              k_schedule=args.k_schedule, k_private_fixed=args.k_private_fixed,
                              w_rec=args.w_rec, w_align=args.w_align, w_dec=args.w_dec)
    raise ValueError(f"unknown fusion {name!r}; choose from {FUSIONS}")
