"""
HUG — graph view + sequence view (spec 03)
-------------------------------------------
InputEncoder        shared h0 for every node type (IDs + features)
RelLightGCN         relation-aware LightGCN over a graph snapshot, with an
                    XSimGCL-style contrastive regulariser
HistoryTransformer  causal-by-construction encoder over the user's click
                    history with target attention
HUGModel            fusion head over [g_user, g_video, g_user*g_video, s_seq, ctx]

This module is dataset-agnostic: it sees only index tensors and feature
matrices prepared by hug_train.py.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Node types: "user", "video" (the item type of any dataset) and the dataset's
# metadata types (ID embeddings), declared by its GraphSpec.


def _emb(n: int, d: int, std: float = 0.1) -> nn.Embedding:
    e = nn.Embedding(n, d)
    nn.init.normal_(e.weight, std=std)
    return e


# ── A. Shared input layer ─────────────────────────────────────────────────────

class InputEncoder(nn.Module):
    """
    h0 for every node:
        user     : user_id_emb[vocab] + Linear(cont features ‖ onehot embeddings)
        video    : video_id_emb[vocab] + Linear(snapshot video features) + Σ cat_emb
        metadata : ID embedding per metadata node type (e.g. author, category)

    Vocabulary row 0 is the shared OOV row.  Node → vocabulary maps and static
    node features are buffers set by `set_node_data` (not saved in checkpoints;
    they are rebuilt deterministically from the data).
    """

    def __init__(
        self,
        d:                 int,
        n_user_vocab:      int,
        user_cont_dim:     int,
        user_onehot_vocab: list[int],
        n_video_vocab:     int,
        video_feat_dim:    int,
        cat_vocab_sizes:   list[int],
        meta_counts:       dict[str, int],
    ) -> None:
        super().__init__()
        self.d = d
        self.user_id   = _emb(n_user_vocab, d)
        self.onehot    = nn.ModuleList([_emb(v + 1, 8) for v in user_onehot_vocab])
        self.user_proj = nn.Linear(user_cont_dim + 8 * len(user_onehot_vocab), d)
        self.video_id   = _emb(n_video_vocab, d)
        self.video_proj = nn.Linear(video_feat_dim, d)
        self.cat        = nn.ModuleList([_emb(v, d) for v in cat_vocab_sizes])
        # one ID table per metadata type, registered under the type's name (in order)
        self.meta_types = list(meta_counts)
        for t, n in meta_counts.items():
            if t in ("user", "video") or hasattr(self, t):
                raise ValueError(f"metadata node type {t!r} clashes with an InputEncoder attribute")
            setattr(self, t, _emb(n, d))

    def set_node_data(self, user_vocab: Tensor, user_x: Tensor, user_onehot: Tensor,
                      video_vocab: Tensor, video_cat: Tensor) -> None:
        for name, t in (("user_vocab", user_vocab), ("user_x", user_x),
                        ("user_onehot", user_onehot), ("video_vocab", video_vocab),
                        ("video_cat", video_cat)):
            self.register_buffer(name, t, persistent=False)

    def users(self, idx: Tensor | None = None) -> Tensor:
        sel = (lambda t: t) if idx is None else (lambda t: t[idx])
        oh  = sel(self.user_onehot)
        x   = torch.cat([sel(self.user_x)] + [e(oh[..., i]) for i, e in enumerate(self.onehot)], -1)
        return self.user_id(sel(self.user_vocab)) + self.user_proj(x)

    def videos(self, video_x: Tensor, idx: Tensor | None = None) -> Tensor:
        sel = (lambda t: t) if idx is None else (lambda t: t[idx])
        h = self.video_id(sel(self.video_vocab)) + self.video_proj(sel(video_x))
        cats = sel(self.video_cat)
        for i, e in enumerate(self.cat):
            h = h + e(cats[..., i])
        return h

    def tables(self, video_x: Tensor) -> dict[str, Tensor]:
        return {"user": self.users(), "video": self.videos(video_x),
                **{t: getattr(self, t).weight for t in self.meta_types}}

    def l2_touched(self, user_idx: Tensor, video_idx: Tensor) -> Tensor:
        """Squared norm of the ID-embedding rows a batch touches (users, videos incl. history)."""
        u = self.user_id(self.user_vocab[user_idx])
        v = self.video_id(self.video_vocab[video_idx])
        return u.pow(2).sum() + v.pow(2).sum()


# ── B. Graph view ─────────────────────────────────────────────────────────────

class RelLightGCN(nn.Module):
    """
    h{l+1}[i] = Σ_r softmax(a_l)[r] · Σ_{j∈N_r(i)} h_l[j] / sqrt(deg_r(i)·deg_r(j))
    g[i]      = mean_{l=0..L} h_l[i]

    Each relation is a sparse [n_dst, n_src] matrix sized to its destination
    node type.  Matrices are rebuilt by `set_graph` whenever a snapshot is
    entered, from that snapshot's (masked) edges only, so degrees are
    snapshot degrees.
    """

    def __init__(self, relations: list[tuple[str, str]], num_layers: int,
                 eps: float = 0.1, cl_layer: int = 1) -> None:
        super().__init__()
        self.relations  = relations                 # (src_type, dst_type) per relation id
        self.num_layers = num_layers
        self.eps        = eps
        self.cl_layer   = cl_layer
        self.a = nn.Parameter(torch.zeros(num_layers, len(relations)))
        self._adj: list[Tensor | None] = [None] * len(relations)

    @staticmethod
    def normalized_adjacency(src: Tensor, dst: Tensor, n_src: int, n_dst: int) -> Tensor:
        """Sparse [n_dst, n_src] with entries 1/sqrt(deg_dst(i)·deg_src(j)) (multi-edges summed)."""
        if src.numel() == 0:
            return torch.sparse_coo_tensor(torch.zeros(2, 0, dtype=torch.long, device=src.device),
                                           torch.zeros(0, device=src.device), (n_dst, n_src)).coalesce()
        deg_dst = torch.bincount(dst, minlength=n_dst).float()
        deg_src = torch.bincount(src, minlength=n_src).float()
        w = (deg_dst[dst] * deg_src[src]).rsqrt()
        return torch.sparse_coo_tensor(torch.stack([dst, src]), w, (n_dst, n_src)).coalesce()

    def set_graph(self, edges: list[tuple[Tensor, Tensor]], counts: dict[str, int]) -> None:
        """edges[r] = (src_local, dst_local) for relation r in the current snapshot."""
        self._adj = [
            self.normalized_adjacency(s, t, counts[st], counts[dt])
            for (s, t), (st, dt) in zip(edges, self.relations)
        ]

    def _perturb(self, h: Tensor) -> Tensor:
        noise = F.normalize(torch.rand_like(h), dim=-1)
        return h + self.eps * torch.sign(h) * noise

    def forward(self, h0: dict[str, Tensor], perturb: bool = False,
                uniform: bool = False) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Returns (g = layer mean, h at layer cl_layer)."""
        h = dict(h0)
        acc = {t: v for t, v in h0.items()}
        h_cl = dict(h0) if self.cl_layer == 0 else {}
        for l in range(self.num_layers):
            w = (torch.full_like(self.a[l], 1.0 / len(self.relations)) if uniform
                 else torch.softmax(self.a[l], dim=0))
            out = {t: torch.zeros_like(v) for t, v in h.items()}
            for r, (st, dt) in enumerate(self.relations):
                adj = self._adj[r]
                if adj is None or adj._nnz() == 0:
                    continue
                out[dt] = out[dt] + w[r] * torch.sparse.mm(adj, h[st])
            if perturb:
                out = {t: self._perturb(v) for t, v in out.items()}
            h = out
            acc = {t: acc[t] + h[t] for t in acc}
            if l + 1 == self.cl_layer:
                h_cl = dict(h)
        g = {t: v / (self.num_layers + 1) for t, v in acc.items()}
        return g, h_cl

    def relation_weights(self) -> Tensor:
        """softmax(a_l) table [layers, relations]."""
        return torch.softmax(self.a.detach(), dim=1)


def info_nce(z1: Tensor, z2: Tensor, temperature: float) -> Tensor:
    z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / temperature
    return F.cross_entropy(logits, torch.arange(len(z1), device=z1.device))


# ── C. Sequence view ──────────────────────────────────────────────────────────

class HistoryTransformer(nn.Module):
    """
    Tokens: h0(video) + pos_emb(recency rank) + emb(log-bucketed gap) + emb(same-session).
    Pre-LN transformer with a padding mask (every token is already in the past),
    then target attention with the candidate's h0 as query, concatenated with
    the mean over valid tokens.  Rows with no history get a learned vector.
    """

    N_GAP_BUCKETS = 32

    def __init__(self, d: int, num_layers: int, max_len: int, heads: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.pos  = _emb(max_len, d)
        self.gap  = _emb(self.N_GAP_BUCKETS, d)
        self.sess = _emb(2, d)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.empty = nn.Parameter(torch.zeros(2 * d))

    @classmethod
    def gap_bucket(cls, gap_ms: Tensor) -> Tensor:
        """floor(log2(seconds + 1)), clipped to the bucket range."""
        sec = gap_ms.clamp(min=0).double() / 1000.0
        return torch.log2(sec + 1).floor().long().clamp(0, cls.N_GAP_BUCKETS - 1)

    def forward(self, tok: Tensor, mask: Tensor, rank: Tensor, gap: Tensor,
                same_sess: Tensor, query: Tensor) -> Tensor:
        """
        tok [B,L,d] item h0, mask [B,L] True = real token, rank/gap/same_sess [B,L] long,
        query [B,d] candidate h0.  Returns s_seq [B, 2d].
        """
        has = mask.any(dim=1)
        safe = mask.clone()
        safe[~has, 0] = True                         # avoid all-masked rows in attention
        x = tok + self.pos(rank) + self.gap(gap) + self.sess(same_sess)
        o = self.norm(self.encoder(x, src_key_padding_mask=~safe))

        score = (self.q(query).unsqueeze(1) * self.k(o)).sum(-1) / math.sqrt(o.shape[-1])
        alpha = torch.softmax(score.masked_fill(~safe, float("-inf")), dim=1)
        att   = (alpha.unsqueeze(-1) * o).sum(1)
        m     = safe.unsqueeze(-1).float()
        mean  = (o * m).sum(1) / m.sum(1)
        s = torch.cat([att, mean], dim=-1)
        return torch.where(has.unsqueeze(-1), s, self.empty.expand_as(s))


# ── D. Fusion head + full model ───────────────────────────────────────────────

class HUGModel(nn.Module):
    """
    z = [ u ‖ v ‖ u⊙v ‖ s_seq ‖ ctx ],  logit = MLP(z)

    u, v are graph-view outputs g (or h0 with use_graph=False).
    ctx = emb(context categoricals) ‖ per-row numeric context.
    """

    def __init__(
        self,
        inp:          InputEncoder,
        gcn:          RelLightGCN | None,
        seq:          HistoryTransformer | None,
        ctx_vocab:    list[int],
        ctx_num_dim:  int,
        d:            int,
        cl_weight:    float = 0.1,
        cl_temp:      float = 0.2,
        emb_l2:       float = 1e-6,
        freeze_graph: bool = False,
        graph_tokens: bool = False,
        dropout:      float = 0.1,
        fusion:       str = "concat",
        fusion_args=None,
    ) -> None:
        super().__init__()
        self.inp, self.gcn, self.seq = inp, gcn, seq
        self.cl_weight    = 0.0 if (freeze_graph or gcn is None) else cl_weight
        self.cl_temp      = cl_temp
        self.emb_l2       = emb_l2
        self.freeze_graph = freeze_graph
        self.graph_tokens = graph_tokens and gcn is not None
        self.ctx_emb = nn.ModuleList([_emb(v, 16) for v in ctx_vocab])
        # Fusion head (spec 05).  'concat' is the spec 03 MLP, built at the same point so
        # parameter init and names are unchanged.
        from fusion import build_fusion
        if fusion != "concat" and (gcn is None or seq is None):
            raise ValueError(f"--fusion {fusion} needs both views; drop --no-graph/--no-seq")
        self.fusion = fusion
        ctx_dim = 16 * len(ctx_vocab) + ctx_num_dim
        self.head = build_fusion(fusion, 3 * d, 2 * d if seq is not None else None, ctx_dim,
                                 fusion_args if fusion_args is not None else _ConcatOnly(dropout))

    # graph tables ------------------------------------------------------------
    def graph_tables(self, video_x: Tensor, train: bool) -> dict:
        """
        Full-graph forward.  Training: with gradients, perturbed (XSimGCL).
        Frozen graph: no gradients, uniform relation weights, no perturbation.
        """
        h0 = self.inp.tables(video_x)
        if self.gcn is None:
            return {"g": h0, "h0": h0, "h_cl": None}
        if self.freeze_graph:
            with torch.no_grad():
                g, _ = self.gcn({t: v.detach() for t, v in h0.items()}, uniform=True)
            return {"g": g, "h0": h0, "h_cl": None}
        perturb = train and self.cl_weight > 0
        g, h_cl = self.gcn(h0, perturb=perturb)
        return {"g": g, "h0": h0, "h_cl": h_cl if perturb else None}

    # per-row forward ---------------------------------------------------------
    def forward(self, batch: dict[str, Tensor], video_x: Tensor,
                tables: dict | None = None, train: bool = False) -> dict[str, Tensor]:
        """
        batch: user, video [B]; hist [B,L] (-1 pad), hist_rank/gap/sess [B,L];
               ctx_cat [B,C] long; ctx_num [B,K]; label [B] (optional).
        `tables` (from graph_tables) is required when the graph view is on; with
        use_graph off, h0 is computed for the needed rows only.
        """
        user, video, hist = batch["user"], batch["video"], batch["hist"]
        hmask = hist >= 0
        hidx  = hist.clamp(min=0)

        if self.gcn is not None:
            u, v = tables["g"]["user"][user], tables["g"]["video"][video]
            tok_table = tables["g"]["video"] if self.graph_tokens else tables["h0"]["video"]
            tok, q = tok_table[hidx], tables["h0"]["video"][video]
        else:
            u = self.inp.users(user)
            v = self.inp.videos(video_x, video)
            tok = self.inp.videos(video_x, hidx) if self.seq is not None else None
            q = v

        e_G = torch.cat([u, v, u * v], dim=-1)
        e_S = (self.seq(tok, hmask, batch["hist_rank"], batch["hist_gap"], batch["hist_sess"], q)
               if self.seq is not None else None)
        ctx = torch.cat([e(batch["ctx_cat"][:, i]) for i, e in enumerate(self.ctx_emb)]
                        + [batch["ctx_num"]], dim=-1)
        fused = self.head.fuse(e_G, e_S, ctx, batch.get("evidence"))
        logit = fused["logit"]
        out = {"logit": logit, "proba": torch.sigmoid(logit), "stats": fused["stats"]}

        if "label" in batch:
            bce = F.binary_cross_entropy_with_logits(logit, batch["label"].float())
            loss = bce
            if self.emb_l2 > 0:
                touched_v = torch.cat([video, hist[hmask]])
                loss = loss + self.emb_l2 * self.inp.l2_touched(user, touched_v) / len(user)
            aux = fused["aux_loss"].detach()
            if self.fusion != "concat":
                loss = loss + fused["aux_loss"]
            cl = torch.zeros((), device=logit.device)
            if train and self.cl_weight > 0 and tables is not None and tables["h_cl"]:
                for t, idx in (("user", user), ("video", video)):
                    nodes = torch.unique(idx)
                    cl = cl + info_nce(tables["g"][t][nodes], tables["h_cl"][t][nodes], self.cl_temp)
                loss = loss + self.cl_weight * cl
            out.update(loss=loss, bce=bce.detach(), cl=cl.detach(), aux=aux)
        return out


class _ConcatOnly:
    """Minimal argument holder for the default concat head."""

    def __init__(self, dropout: float) -> None:
        self.dropout = dropout
