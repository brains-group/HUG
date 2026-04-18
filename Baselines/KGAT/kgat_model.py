"""
kgat_model.py
=============
Knowledge Graph Attention Network (KGAT) for CVR prediction.

Based on: Wang et al., "KGAT: Knowledge Graph Attention Network for
Recommendation", KDD 2019.

Adapted for KuaiRand using the structural subgraph (user, video, author,
category nodes, 7 relation types) from the HUG framework.

Architecture
------------
1. Project each node type to a shared hidden_dim via learned linear layers.
2. L KGATLayer message-passing steps using TransR-style relational attention:
       score(h, r, t) = (W_r @ h_src)^T h_dst
       alpha = per-source-node softmax over scores
       h_new = LayerNorm(h + dropout( Σ alpha * h_dst ))
3. Project all nodes from hidden_dim to out_dim.
4. CVR head: concat(user_emb, video_emb) → 3-layer MLP → P(click).

API compatibility
-----------------
The model exposes the same two-method API as SingleGNNModel in the HUG
framework (encode_graph / forward_from_embeddings), so the shared
run_epoch training loop in Framework/main.py works without modification.

Dimension annotations
---------------------
  N_all = N_u + N_v + N_a + N_c   (all entity nodes, global-offset layout)
  H     = hidden_dim
  D     = out_dim
  E     = number of edges in the structural subgraph
  B     = mini-batch size
"""

from __future__ import annotations

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax as pyg_softmax   # per-node group softmax


# ── KGAT Message-Passing Layer ────────────────────────────────────────────────

class KGATLayer(nn.Module):
    """
    One KGAT message-passing layer with TransR-style relational attention.

    For each entity node h, the layer aggregates messages from its KG
    neighbours t via relation r:

        h_proj[e]     = W_r[rel[e]] @ h_all[src[e]]     [E, H]
        score[e]      = (h_proj[e] · h_all[dst[e]])      [E]
        alpha[e]      = softmax over edges with same src  [E]
        h_agg[src[e]] += alpha[e] * h_all[dst[e]]         scatter [N_all, H]
        h_out         = LayerNorm(h_all + dropout(h_agg)) [N_all, H]

    Shapes
    ------
    h          : [N_all, H]   — all entity embeddings in global-offset order
    edge_index : [2, E]       — (src, dst) using global node indices
    rel_types  : [E]          — integer relation id per edge, 0 … R-1
    """

    def __init__(self, hidden_dim: int, n_relations: int, dropout: float = 0.2):
        super().__init__()
        self.H = hidden_dim

        # Per-relation projection matrices  W_r : [R, H, H]
        # Initialised per-slice with Xavier uniform to keep activation variance
        # stable regardless of how many relations R is.
        self.W_r = nn.Parameter(torch.empty(n_relations, hidden_dim, hidden_dim))
        for i in range(n_relations):
            nn.init.xavier_uniform_(self.W_r[i])

        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        h:          Tensor,   # [N_all, H]
        edge_index: Tensor,   # [2, E]
        rel_types:  Tensor,   # [E]       relation id per edge
    ) -> Tensor:              # [N_all, H]
        src, dst = edge_index[0], edge_index[1]   # [E] each
        N = h.shape[0]
        H = self.H

        # ── Relational projection of source embeddings ────────────────────────
        # W_r[rel_types] : [E, H, H]   (one matrix per edge, indexed by rel id)
        # h[src]         : [E, H]
        # einsum 'eij,ej->ei': for edge e, output[e,i] = Σ_j W_r[e,i,j]*h[src[e],j]
        # → each row is W_r[rel_e] @ h_src_e                             [E, H]
        h_proj = torch.einsum('eij,ej->ei', self.W_r[rel_types], h[src])   # [E, H]

        # ── Attention scores ──────────────────────────────────────────────────
        # Inner product of projected source with destination embedding
        scores = (h_proj * h[dst]).sum(dim=-1)   # [E]

        # ── Per-source-node softmax ───────────────────────────────────────────
        # pyg_softmax groups by `src` and normalises within each group.
        # Nodes with no outgoing edges receive no messages (zero h_agg row).
        alpha = pyg_softmax(scores, src, num_nodes=N)   # [E]

        # ── Aggregate weighted destination embeddings → source nodes ──────────
        # weighted[e] = alpha[e] * h[dst[e]]                             [E, H]
        weighted = alpha.unsqueeze(-1) * h[dst]                          # [E, H]

        # scatter_add: h_agg[src[e]] += weighted[e]
        h_agg = torch.zeros(N, H, dtype=h.dtype, device=h.device)
        h_agg.scatter_add_(
            0,
            src.unsqueeze(-1).expand(-1, H),   # [E, H] index
            weighted,                           # [E, H] src
        )   # h_agg : [N_all, H]

        # ── Residual connection + LayerNorm ───────────────────────────────────
        return self.norm(h + self.drop(h_agg))   # [N_all, H]


# ── Full KGAT Model ───────────────────────────────────────────────────────────

class KGATModel(nn.Module):
    """
    KGAT-based CVR prediction model for KuaiRand.

    Parameters
    ----------
    user_cont_dim : int
        Continuous feature dimension for user nodes (from HKG user.x).
    user_onehot_vocab : list[int]
        Vocabulary size of each encrypted onehot user feature.
        One nn.Embedding table of size 8 is created per entry.
    video_feat_dim : int
        Feature dimension for video nodes (from HKG video.x, typically 62).
    author_feat_dim : int
        Feature dimension for author nodes (typically 1).
    category_feat_dim : int
        Feature dimension for category nodes (typically 1).
    hidden_dim : int
        Width of all intermediate KGAT representations.
    out_dim : int
        Output embedding dimension for user and video nodes.
    n_relations : int
        Number of distinct relation types in the structural subgraph (7 in HUG).
    n_layers : int
        Number of KGAT message-passing layers.
    dropout : float
        Dropout applied inside each KGATLayer and in the CVR head.
    """

    def __init__(
        self,
        user_cont_dim:      int,
        user_onehot_vocab:  list[int],
        video_feat_dim:     int,
        author_feat_dim:    int,
        category_feat_dim:  int,
        hidden_dim:         int   = 128,
        out_dim:            int   = 64,
        n_relations:        int   = 7,
        n_layers:           int   = 2,
        dropout:            float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim    = out_dim

        # ── Input projections (one per node type) ─────────────────────────────
        # Onehot features: each encrypted column → embedding of size 8
        self.onehot_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, 8) for vocab_size in user_onehot_vocab
        ])
        user_input_dim = user_cont_dim + 8 * len(user_onehot_vocab)

        self.user_proj     = nn.Linear(user_input_dim,              hidden_dim)
        self.video_proj    = nn.Linear(video_feat_dim,              hidden_dim)
        self.author_proj   = nn.Linear(max(author_feat_dim,   1),   hidden_dim)
        self.category_proj = nn.Linear(max(category_feat_dim, 1),   hidden_dim)

        # ── KGAT message-passing layers ───────────────────────────────────────
        self.kgat_layers = nn.ModuleList([
            KGATLayer(hidden_dim, n_relations, dropout)
            for _ in range(n_layers)
        ])

        # ── Output projection: hidden_dim → out_dim ───────────────────────────
        # Applied after all KGAT layers to compress to the shared embedding dim.
        self.out_proj = nn.Linear(hidden_dim, out_dim)

        # ── CVR head ──────────────────────────────────────────────────────────
        # Concat(user_emb, video_emb) → fused_dim = out_dim * 2
        # Must match SingleGNNModel convention (no alignment module).
        #
        # Imported here (not at module level) so this file can be used
        # independently of the PYTHONPATH setup in train_kgat.py.
        _framework = os.path.join(os.path.dirname(__file__), "..", "..", "Framework")
        if _framework not in sys.path:
            sys.path.insert(0, _framework)
        from models import CVRHead

        self.cvr_head = CVRHead(
            fused_dim   = out_dim * 2,
            hidden_dims = [hidden_dim, hidden_dim // 2],
            dropout     = dropout,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _encode_user(self, x: Tensor, onehot: Tensor) -> Tensor:
        """Project continuous + onehot user features to hidden_dim."""
        embeds = [emb(onehot[:, i]) for i, emb in enumerate(self.onehot_embeddings)]
        return F.relu(self.user_proj(torch.cat([x] + embeds, dim=-1)))   # [N_u, H]

    @property
    def head_device(self) -> torch.device:
        return next(self.cvr_head.parameters()).device

    # ── Graph encoding (called once per epoch, no gradients) ─────────────────

    @torch.no_grad()
    def encode_graph(
        self,
        graph:     HeteroData,
        rel_ei:    Tensor,     # [2, E]  from build_relation_edge_index
        rel_types: Tensor,     # [E]     relation type per edge
    ) -> dict[str, Tensor]:
        """
        Run KGAT propagation over the full structural subgraph.

        Node layout in h_all: [user | video | author | category]
        (identical to StructuralGNN so build_relation_edge_index offsets match).

        Returns
        -------
        dict with keys 's_user' and 's_video' (same as SingleGNNModel) so
        the shared run_epoch loop in main.py works without modification.

            's_user'  : [N_u, out_dim]
            's_video' : [N_v, out_dim]
        """
        dev = next(self.parameters()).device

        # ── Project each node type to hidden_dim ──────────────────────────────
        h_user     = self._encode_user(
            graph["user"].x.to(dev), graph["user"].onehot.to(dev)
        )                                                          # [N_u, H]
        h_video    = F.relu(self.video_proj(   graph["video"].x.to(dev)))    # [N_v, H]
        h_author   = F.relu(self.author_proj(  graph["author"].x.to(dev)))   # [N_a, H]
        h_category = F.relu(self.category_proj(graph["category"].x.to(dev))) # [N_c, H]

        n_user  = h_user.shape[0]
        n_video = h_video.shape[0]

        # ── Build h_all in global-offset order ────────────────────────────────
        # Offsets match build_relation_edge_index: user=0, video=n_u, author=n_u+n_v …
        h_all = torch.cat([h_user, h_video, h_author, h_category], dim=0)  # [N_all, H]

        # ── KGAT message passing ──────────────────────────────────────────────
        rei = rel_ei.to(dev)     # [2, E]
        rlt = rel_types.to(dev)  # [E]

        for layer in self.kgat_layers:
            h_all = layer(h_all, rei, rlt)   # [N_all, H]

        # ── Project to out_dim ────────────────────────────────────────────────
        # ReLU before projection keeps activations non-negative (KGAT paper uses
        # LeakyReLU; we use ReLU to match the rest of the HUG codebase).
        h_out = F.relu(self.out_proj(h_all))   # [N_all, D]

        return {
            "s_user":  h_out[:n_user],                    # [N_u, D]
            "s_video": h_out[n_user : n_user + n_video],  # [N_v, D]
        }

    # ── Batch-level prediction (called every mini-batch, with gradients) ──────

    def forward_from_embeddings(
        self,
        emb:         dict[str, Tensor],
        user_idx:    Tensor,              # [B]
        video_idx:   Tensor,              # [B]
        session_idx: Tensor,              # [B]  accepted for API compat, unused
        ips_weights: Tensor | None = None,
        labels:      Tensor | None = None,
        kg_relation: Tensor | None = None,  # unused
    ) -> dict[str, Tensor]:
        """
        Gather per-sample embeddings and run the CVR head.

        No alignment module — user and video embeddings are concatenated
        (same as SingleGNNModel).

        Returns
        -------
        dict with keys: 'logits' [B], 'proba' [B], 'fused' [B, out_dim*2],
        and 'loss' (scalar) when labels are provided.
        """
        h_user  = emb["s_user"][user_idx]    # [B, D]
        h_video = emb["s_video"][video_idx]  # [B, D]

        fused  = torch.cat([h_user, h_video], dim=-1)   # [B, D*2]
        logits = self.cvr_head(fused)                    # [B]
        proba  = torch.sigmoid(logits)

        out: dict[str, Tensor] = {"logits": logits, "proba": proba, "fused": fused}

        if labels is not None:
            from models import CVRHead
            w = ips_weights if ips_weights is not None else torch.ones_like(logits)
            out["loss"] = CVRHead.ips_bce_loss(logits, labels, w)

        return out
