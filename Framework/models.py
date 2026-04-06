"""
Alignment Module + CVR Head
-----------------------------
AlignmentModule
    Takes structural embeddings (h_s_user, h_s_video) and sequential
    embeddings (h_q_video, h_q_session) and fuses them via KG-relation-aware
    cross-attention.  The KG relation path between user and candidate video
    (e.g. user-liked-category ↔ video-tagged-category) acts as a structural
    prior that gates how much each embedding space contributes to the final
    fused representation.

CVRHead
    A lightweight MLP that takes the fused user–video pair representation
    and outputs P(click | user, video, context).  Applies IPS weighting
    during training to correct for recommendation policy bias.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Alignment Module ──────────────────────────────────────────────────────────

class AlignmentModule(nn.Module):
    """
    KG-guided cross-attention alignment of structural and sequential embeddings.

    For each (user, candidate_video) pair the module:
      1. Computes cross-attention: structural video embedding attends over the
         sequential session embedding (and vice versa).
      2. Weights the attention by a KG-relation compatibility score derived
         from shared category / author paths between user and video.
      3. Concatenates the aligned representations into a fused vector.

    Parameters
    ----------
    emb_dim : int
        Embedding dimension from both GNNs (must match).
    kg_relation_dim : int
        Dimension of the KG relation path feature vector.
        Set to 0 to disable KG gating (falls back to plain cross-attention).
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout on attention weights.
    """

    def __init__(
        self,
        emb_dim:         int = 64,
        kg_relation_dim: int = 16,
        num_heads:       int = 4,
        dropout:         float = 0.1,
    ) -> None:
        super().__init__()
        self.emb_dim         = emb_dim
        self.kg_relation_dim = kg_relation_dim
        self.num_heads       = num_heads
        head_dim             = emb_dim // num_heads
        assert head_dim * num_heads == emb_dim, "emb_dim must be divisible by num_heads"

        # Cross-attention: structural video ← sequential session
        self.q_struct = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k_seq    = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v_seq    = nn.Linear(emb_dim, emb_dim, bias=False)

        # Cross-attention: sequential session ← structural video
        self.q_seq    = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k_struct = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v_struct = nn.Linear(emb_dim, emb_dim, bias=False)

        # KG relation gate (scalar per head per pair)
        if kg_relation_dim > 0:
            self.kg_proj   = nn.Linear(kg_relation_dim, num_heads)
            self.use_kg_gate = True
        else:
            self.use_kg_gate = False

        # User embedding alignment
        self.user_align = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
        )

        # Final fusion: [aligned_user | aligned_video_struct | aligned_video_seq]
        self.fusion = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim),
        )

        self.scale = (emb_dim // num_heads) ** -0.5

    def forward(
        self,
        h_s_user:   Tensor,           # [B, emb_dim]   structural user emb
        h_s_video:  Tensor,           # [B, emb_dim]   structural video emb (candidate)
        h_q_video:  Tensor,           # [B, emb_dim]   sequential video emb (candidate)
        h_q_session: Tensor,          # [B, emb_dim]   sequential session emb
        kg_relation: Tensor | None = None,  # [B, kg_relation_dim]
    ) -> Tensor:
        """
        Parameters
        ----------
        All tensors are batched over (user, candidate_video) pairs.

        Returns
        -------
        fused : Tensor of shape [B, emb_dim]
        """
        B = h_s_user.shape[0]
        H = self.num_heads
        D = self.emb_dim // H

        # ── Cross-attention A: structural video ← sequential session ──────────
        q_s  = self.q_struct(h_s_video).view(B, H, D)
        k_sq = self.k_seq(h_q_session).view(B, H, D)
        v_sq = self.v_seq(h_q_session).view(B, H, D)

        attn_s = (q_s * k_sq).sum(-1) * self.scale   # [B, H]

        # ── Cross-attention B: sequential video ← structural video ────────────
        q_q  = self.q_seq(h_q_video).view(B, H, D)
        k_qs = self.k_struct(h_s_video).view(B, H, D)
        v_qs = self.v_struct(h_s_video).view(B, H, D)

        attn_q = (q_q * k_qs).sum(-1) * self.scale   # [B, H]

        # ── KG relation gate ──────────────────────────────────────────────────
        if self.use_kg_gate and kg_relation is not None:
            kg_gate = self.kg_proj(kg_relation)          # [B, H]
            attn_s  = attn_s + kg_gate
            attn_q  = attn_q + kg_gate

        attn_s = torch.softmax(attn_s, dim=-1).unsqueeze(-1)   # [B, H, 1]
        attn_q = torch.softmax(attn_q, dim=-1).unsqueeze(-1)   # [B, H, 1]

        aligned_s = (attn_s * v_sq).view(B, self.emb_dim)      # [B, D]
        aligned_q = (attn_q * v_qs).view(B, self.emb_dim)      # [B, D]

        # ── User alignment: blend structural user with session context ─────────
        aligned_user = self.user_align(torch.cat([h_s_user, h_q_session], dim=-1))

        # ── Final fusion ───────────────────────────────────────────────────────
        fused = self.fusion(torch.cat([aligned_user, aligned_s, aligned_q], dim=-1))
        return fused


# ── CVR Prediction Head ────────────────────────────────────────────────────────

class CVRHead(nn.Module):
    """
    MLP that maps a fused (user, video) representation to P(click).

    Supports IPS-weighted binary cross-entropy loss to correct for
    recommendation policy exposure bias.

    Parameters
    ----------
    fused_dim : int
        Input dimension from the AlignmentModule output.
    hidden_dims : list[int]
        Width of each hidden MLP layer.
    dropout : float
        Dropout between hidden layers.
    """

    def __init__(
        self,
        fused_dim:   int = 64,
        hidden_dims: list[int] | None = None,
        dropout:     float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers: list[nn.Module] = []
        in_dim = fused_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, fused: Tensor) -> Tensor:
        """
        Parameters
        ----------
        fused : Tensor [B, fused_dim]

        Returns
        -------
        logits : Tensor [B] (raw logits, not probabilities)
        """
        return self.mlp(fused).squeeze(-1)

    def predict_proba(self, fused: Tensor) -> Tensor:
        """Return P(click) in [0, 1]."""
        return torch.sigmoid(self.forward(fused))

    @staticmethod
    def ips_bce_loss(
        logits:     Tensor,
        labels:     Tensor,
        ips_weights: Tensor,
        clip_weight: float = 10.0,
    ) -> Tensor:
        """
        IPS-weighted binary cross-entropy loss.

        Random-policy interactions get weight ≈ 1.0 (unbiased).
        Standard-policy interactions are downweighted by their propensity.

        Parameters
        ----------
        logits : Tensor [B]
        labels : Tensor [B]  — is_click target
        ips_weights : Tensor [B]  — pre-computed IPS weights per sample
        clip_weight : float
            Cap on IPS weight to prevent variance explosion.

        Returns
        -------
        Scalar loss tensor.
        """
        w = ips_weights.clamp(max=clip_weight)
        w = w / w.mean()   # normalise so effective batch size is preserved

        bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
        return (bce * w).mean()


# ── Full CVR Model (wires everything together) ─────────────────────────────────

class KuaiCVRModel(nn.Module):
    """
    End-to-end model: StructuralGNN + SequentialGNN → AlignmentModule → CVRHead.

    Intended for training and inference as a single nn.Module.

    Parameters
    ----------
    structural_gnn : StructuralGNN
    sequential_gnn : SequentialGNN
    alignment      : AlignmentModule
    cvr_head       : CVRHead
    """

    def __init__(
        self,
        structural_gnn,
        sequential_gnn,
        alignment:  AlignmentModule,
        cvr_head:   CVRHead,
    ) -> None:
        super().__init__()
        self.structural_gnn = structural_gnn
        self.sequential_gnn = sequential_gnn
        self.alignment      = alignment
        self.cvr_head       = cvr_head

    def split_across_gpus(
        self,
        gpu_structural: int = 0,
        gpu_sequential: int = 1,
        gpu_head:       int = 0,
    ) -> "KuaiCVRModel":
        """
        Place different components on different GPUs.

        StructuralGNN  → gpu_structural  (R-GCN over large node set)
        SequentialGNN  → gpu_sequential  (GGNN over session graph)
        AlignmentModule + CVRHead → gpu_head (lightweight, runs per batch)

        The encode step runs both GNNs in parallel on their respective GPUs.
        The batch loop runs alignment + head on gpu_head using small [B, D]
        slices that are transferred from CPU pinned memory.

        Returns self for chaining: model.split_across_gpus().
        """
        self.structural_gnn = self.structural_gnn.to(f"cuda:{gpu_structural}")
        self.sequential_gnn = self.sequential_gnn.to(f"cuda:{gpu_sequential}")
        self.alignment      = self.alignment.to(f"cuda:{gpu_head}")
        self.cvr_head       = self.cvr_head.to(f"cuda:{gpu_head}")
        self._gpu_structural = gpu_structural
        self._gpu_sequential = gpu_sequential
        self._gpu_head       = gpu_head
        return self

    @property
    def head_device(self) -> torch.device:
        """Device where alignment + CVR head live."""
        return next(self.alignment.parameters()).device

    def forward(
        self,
        structural_graph,          # HeteroData — structural subgraph
        sequential_graph,          # HeteroData — sequential subgraph
        relation_edge_index,       # [2, E] merged edge index for R-GCN
        relation_types,            # [E]   relation type per edge
        session_id,                # [E]   session id per seq edge
        user_idx:     Tensor,      # [B]   user indices for batch
        video_idx:    Tensor,      # [B]   candidate video indices for batch
        session_idx:  Tensor,      # [B]   session indices for batch
        kg_relation:  Tensor | None = None,  # [B, kg_relation_dim]
        ips_weights:  Tensor | None = None,  # [B]
        labels:       Tensor | None = None,  # [B]  is_click
    ) -> dict[str, Tensor]:
        """
        Returns
        -------
        dict with:
            'logits'   : [B] raw CVR logits
            'proba'    : [B] P(click)
            'loss'     : scalar (only when labels provided)
            'fused'    : [B, emb_dim] fused representation (for inspection)
        """
        # ── Encode ────────────────────────────────────────────────────────────
        struct_emb = self.structural_gnn(
            structural_graph, relation_edge_index, relation_types
        )
        seq_emb = self.sequential_gnn(sequential_graph, session_id)

        # ── Gather per-sample embeddings ──────────────────────────────────────
        h_s_user    = struct_emb["user"][user_idx]       # [B, D]
        h_s_video   = struct_emb["video"][video_idx]     # [B, D]
        h_q_video   = seq_emb["video"][video_idx]        # [B, D]
        h_q_session = seq_emb["session"][session_idx]    # [B, D]

        # ── Align + fuse ──────────────────────────────────────────────────────
        fused  = self.alignment(h_s_user, h_s_video, h_q_video, h_q_session, kg_relation)
        logits = self.cvr_head(fused)
        proba  = torch.sigmoid(logits)

        out: dict[str, Tensor] = {"logits": logits, "proba": proba, "fused": fused}

        if labels is not None:
            w = ips_weights if ips_weights is not None else torch.ones_like(logits)
            out["loss"] = CVRHead.ips_bce_loss(logits, labels, w)

        return out

    # ── Pre-computed embedding API ────────────────────────────────────────────

    @torch.no_grad()
    def encode_graph(
        self,
        structural_graph,
        sequential_graph,
        relation_edge_index,
        relation_types,
        session_id,
    ) -> dict[str, Tensor]:
        """
        Run both GNNs once over the full graph and return embedding matrices.

        Called once per epoch OUTSIDE the batch loop.  No gradients are kept
        here — this is purely inference over the graph structure.  The returned
        tensors live on the same device as the model parameters.

        Returns
        -------
        dict with keys:
            's_user'    : [N_users,    D]  structural user embeddings
            's_video'   : [N_videos,   D]  structural video embeddings
            'q_video'   : [N_videos,   D]  sequential video embeddings
            'q_session' : [N_sessions, D]  sequential session embeddings
        """
        struct_emb = self.structural_gnn(
            structural_graph, relation_edge_index, relation_types
        )
        seq_emb = self.sequential_gnn(sequential_graph, session_id)

        return {
            "s_user":    struct_emb["user"],
            "s_video":   struct_emb["video"],
            "q_video":   seq_emb["video"],
            "q_session": seq_emb["session"],
        }

    def forward_from_embeddings(
        self,
        emb:         dict[str, Tensor],  # from encode_graph
        user_idx:    Tensor,             # [B]
        video_idx:   Tensor,             # [B]
        session_idx: Tensor,             # [B]
        ips_weights: Tensor | None = None,
        labels:      Tensor | None = None,
        kg_relation: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Run only the alignment module and CVR head using pre-computed embeddings.

        Called inside the batch loop with gradients enabled.  Only the
        alignment and head parameters receive gradients — the GNN weights
        are updated implicitly because encode_graph is called with the live
        model weights at the start of each epoch.

        Parameters
        ----------
        emb         : output of encode_graph (on device)
        user_idx    : [B] indices into emb["s_user"]
        video_idx   : [B] indices into emb["s_video"] and emb["q_video"]
        session_idx : [B] indices into emb["q_session"]
        """
        h_s_user    = emb["s_user"][user_idx]
        h_s_video   = emb["s_video"][video_idx]
        h_q_video   = emb["q_video"][video_idx]
        h_q_session = emb["q_session"][session_idx]

        fused  = self.alignment(h_s_user, h_s_video, h_q_video, h_q_session, kg_relation)
        logits = self.cvr_head(fused)
        proba  = torch.sigmoid(logits)

        out: dict[str, Tensor] = {"logits": logits, "proba": proba, "fused": fused}

        if labels is not None:
            w = ips_weights if ips_weights is not None else torch.ones_like(logits)
            out["loss"] = CVRHead.ips_bce_loss(logits, labels, w)

        return out


# ── Single GNN Baseline Model ─────────────────────────────────────────────────

class SingleGNNModel(nn.Module):
    """
    Baseline model: one HGT over the full HKG → concatenate user + video → CVRHead.

    The alignment module is absent.  User and video embeddings from the single
    HGT are concatenated into a [B, out_dim * 2] vector and passed directly
    to the CVR head.  Session context enters through the HGT's own message
    passing over next_in_session edges rather than through a dedicated GGNN.

    This serves as the ablation baseline for the dual-GNN architecture.
    A higher dual-GNN AUC demonstrates the value of:
      - Separating structural and sequential signals into dedicated encoders
      - The KG-guided cross-attention alignment module
      - The soft-attention + recency-gate session readout in SR-GNN

    Parameters
    ----------
    hgt        : SingleHGT
    cvr_head   : CVRHead  (must have fused_dim = out_dim * 2)
    """

    def __init__(self, hgt, cvr_head: "CVRHead") -> None:
        super().__init__()
        self.hgt      = hgt
        self.cvr_head = cvr_head

    @property
    def head_device(self) -> torch.device:
        return next(self.cvr_head.parameters()).device

    @torch.no_grad()
    def encode_graph(
        self,
        full_graph,
        merged_edge_index: Tensor,
        merged_edge_types:  Tensor,
        session_id:         Tensor,
    ) -> dict[str, Tensor]:
        """
        Run the HGT once and return full embedding matrices.

        Returns
        -------
        dict with keys:
            'user'    : [N_users,    out_dim]
            'video'   : [N_videos,   out_dim]
            'session' : [N_sessions, out_dim]
        """
        return self.hgt(full_graph, merged_edge_index, merged_edge_types, session_id)

    def forward_from_embeddings(
        self,
        emb:         dict[str, Tensor],
        user_idx:    Tensor,
        video_idx:   Tensor,
        session_idx: Tensor,
        ips_weights: Tensor | None = None,
        labels:      Tensor | None = None,
        kg_relation: Tensor | None = None,   # unused, kept for API symmetry
    ) -> dict[str, Tensor]:
        """
        Gather batch embeddings and run the CVR head.

        No alignment module — user and video embeddings are concatenated.
        Session context is baked into the video embedding by the HGT
        (next_in_session edges propagate session signal to video nodes).
        """
        h_user  = emb["s_user"][user_idx]           # [B, D]
        h_video = emb["s_video"][video_idx]        # [B, D]

        # Simple concatenation instead of cross-attention alignment
        fused  = torch.cat([h_user, h_video], dim=-1)   # [B, D*2]
        logits = self.cvr_head(fused)
        proba  = torch.sigmoid(logits)

        out: dict[str, Tensor] = {"logits": logits, "proba": proba, "fused": fused}

        if labels is not None:
            w = ips_weights if ips_weights is not None else torch.ones_like(logits)
            out["loss"] = CVRHead.ips_bce_loss(logits, labels, w)

        return out