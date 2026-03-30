"""
GNN Encoders
-------------
Two separate GNN encoders, each consuming a different subgraph view.

StructuralGNN  (R-GCN style)
    Operates on the structural subgraph: User / Video / Author / Category nodes
    with typed relation edges.  Produces static embeddings capturing global
    preference patterns.

SequentialGNN  (SR-GNN style)
    Operates on the sequential subgraph: Video nodes connected by directed
    next_in_session edges.  Uses a gated graph neural network to capture
    session-level transition dynamics, then pools session representations
    via a soft-attention readout.

Both encoders produce a Video embedding of the same dimensionality so the
downstream alignment module can compare them directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, RGCNConv, GatedGraphConv
from torch_geometric.utils import softmax as pyg_softmax


# ── Structural GNN (R-GCN with optional GAT refinement) ───────────────────────

class StructuralGNN(nn.Module):
    """
    Relational GCN over the structural subgraph.

    Produces embeddings for User and Video nodes by propagating messages
    across typed relation edges (clicked, liked, hated, tagged_as, made_by …).

    Parameters
    ----------
    user_cont_dim : int
        Dimension of continuous user features from HKG (graph["user"].x).
    user_onehot_vocab : list[int]
        Cardinality of each encrypted onehot feature for User nodes.
        One embedding table is created per feature.
    video_feat_dim : int
        Dimension of video feature vector from HKG (graph["video"].x).
    author_feat_dim : int
        Dimension of author node features (typically 1 — learned embedding).
    category_feat_dim : int
        Dimension of category node features (typically 1 — learned embedding).
    hidden_dim : int
        Width of all hidden layers.
    out_dim : int
        Output embedding dimension (same for user and video).
    num_relations : int
        Number of distinct edge relation types in the structural subgraph.
    num_layers : int
        Number of R-GCN message passing layers.
    dropout : float
        Dropout applied after each layer.
    """

    def __init__(
        self,
        user_cont_dim:      int,
        user_onehot_vocab:  list[int],
        video_feat_dim:     int,
        author_feat_dim:    int,
        category_feat_dim:  int,
        hidden_dim:         int = 128,
        out_dim:            int = 64,
        num_relations:      int = 7,
        num_layers:         int = 2,
        dropout:            float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.out_dim     = out_dim
        self.num_layers  = num_layers
        self.dropout     = dropout

        # ── User input projection ──────────────────────────────────────────────
        # Encrypted onehot features get their own embedding tables
        self.onehot_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, 8) for vocab_size in user_onehot_vocab
        ])
        user_input_dim = user_cont_dim + 8 * len(user_onehot_vocab)
        self.user_proj = nn.Linear(user_input_dim, hidden_dim)

        # ── Video / Author / Category input projections ────────────────────────
        self.video_proj    = nn.Linear(video_feat_dim,    hidden_dim)
        self.author_proj   = nn.Linear(max(author_feat_dim, 1),   hidden_dim)
        self.category_proj = nn.Linear(max(category_feat_dim, 1), hidden_dim)

        # ── R-GCN layers ───────────────────────────────────────────────────────
        # PyG RGCNConv operates on a homogeneous edge_index with a relation
        # tensor.  We project all node types to hidden_dim first so they share
        # the same message space.
        self.rgcn_layers = nn.ModuleList([
            RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
            for _ in range(num_layers)
        ])

        # ── GAT refinement (video-only, post R-GCN) ───────────────────────────
        self.gat = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)

        # ── Output projections ─────────────────────────────────────────────────
        self.user_out  = nn.Linear(hidden_dim, out_dim)
        self.video_out = nn.Linear(hidden_dim, out_dim)

        self.norm = nn.LayerNorm(hidden_dim)

    def _encode_user(self, x: Tensor, onehot: Tensor) -> Tensor:
        embeds = [emb(onehot[:, i]) for i, emb in enumerate(self.onehot_embeddings)]
        return F.relu(self.user_proj(torch.cat([x] + embeds, dim=-1)))

    def forward(
        self,
        graph: HeteroData,
        relation_edge_index: Tensor,   # [2, E] merged over all relation types
        relation_types:      Tensor,   # [E]    integer relation id per edge
        video_video_edge_index: Tensor | None = None,  # for GAT pass
    ) -> dict[str, Tensor]:
        """
        Parameters
        ----------
        graph : HeteroData
            Structural subgraph.
        relation_edge_index : Tensor
            All structural edges concatenated, shape [2, E].
        relation_types : Tensor
            Relation type index per edge, shape [E].
        video_video_edge_index : Tensor | None
            Optional video co-occurrence edges for the GAT refinement pass.

        Returns
        -------
        dict with keys 'user' and 'video', each a Tensor of shape [N, out_dim].
        """
        # Infer device from model parameters and move graph tensors there
        dev = next(self.parameters()).device
        h_user     = self._encode_user(
            graph["user"].x.to(dev), graph["user"].onehot.to(dev)
        )
        h_video    = F.relu(self.video_proj(graph["video"].x.to(dev)))
        h_author   = F.relu(self.author_proj(graph["author"].x.to(dev)))
        h_category = F.relu(self.category_proj(graph["category"].x.to(dev)))

        # Concatenate all node embeddings into one big tensor
        # Order: [user | video | author | category]
        n_user     = h_user.shape[0]
        n_video    = h_video.shape[0]
        n_author   = h_author.shape[0]

        h_all = torch.cat([h_user, h_video, h_author, h_category], dim=0)

        # R-GCN message passing over merged edge index
        for layer in self.rgcn_layers:
            h_all = F.relu(self.norm(layer(h_all, relation_edge_index, relation_types)))
            h_all = F.dropout(h_all, p=self.dropout, training=self.training)

        # Split back into per-type tensors
        h_user_out  = h_all[:n_user]
        h_video_out = h_all[n_user : n_user + n_video]

        # Optional GAT refinement on video nodes only (video–video co-engagement)
        if video_video_edge_index is not None and video_video_edge_index.shape[1] > 0:
            h_video_out = F.relu(self.gat(h_video_out, video_video_edge_index))

        return {
            "user":  self.user_out(h_user_out),
            "video": self.video_out(h_video_out),
        }


# ── Sequential GNN (SR-GNN style gated GNN + attention readout) ───────────────

class SequentialGNN(nn.Module):
    """
    Gated Graph Neural Network over session graphs.

    Follows the SR-GNN architecture:
      1. Run a GatedGraphConv over directed next_in_session edges.
      2. Aggregate per-session node embeddings via a soft-attention readout
         that weights recent items more (recency gate).
      3. Return both item-level embeddings and session-level embeddings.

    Parameters
    ----------
    video_feat_dim : int
        Input feature dimension from HKG video node features.
    hidden_dim : int
        GNN hidden width.
    out_dim : int
        Output embedding dimension — must match StructuralGNN.out_dim for
        the alignment module.
    num_layers : int
        Number of gated graph conv steps (time unrolls).
    dropout : float
        Dropout applied to session readout.
    """

    def __init__(
        self,
        video_feat_dim: int,
        hidden_dim:     int = 128,
        out_dim:        int = 64,
        num_layers:     int = 3,
        dropout:        float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim    = out_dim
        self.dropout    = dropout

        # Input projection
        self.input_proj = nn.Linear(video_feat_dim, hidden_dim)

        # Gated GNN propagation
        self.ggnn = GatedGraphConv(hidden_dim, num_layers=num_layers)

        # Session readout — soft attention over node embeddings in a session
        self.attn_q = nn.Linear(hidden_dim, hidden_dim, bias=False)  # query (last item)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim, bias=False)  # keys  (all items)
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)           # score

        # Recency gate: position-based weight blended with attention
        self.recency_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        graph:      HeteroData,
        session_id: Tensor,    # [E] session id per next_in_session edge
    ) -> dict[str, Tensor]:
        """
        Parameters
        ----------
        graph : HeteroData
            Sequential subgraph containing video nodes and
            ("video", "next_in_session", "video") edges.
        session_id : Tensor
            Session assignment per edge, shape [E].

        Returns
        -------
        dict with keys:
            'video'   : item-level embeddings [N_video, out_dim]
            'session' : session-level embeddings [N_session, out_dim]
        """
        dev = next(self.parameters()).device
        edge_index = graph["video", "next_in_session", "video"].edge_index.to(dev)
        x = graph["video"].x.to(dev)

        # Project to hidden dim
        h = F.relu(self.input_proj(x))

        # Gated GNN propagation (directed edges encode temporal order)
        h = self.norm(self.ggnn(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Session-level readout via attention
        n_sessions = int(graph["session"].num_nodes)
        session_emb = self._session_readout(h, edge_index, session_id, n_sessions)

        return {
            "video":   self.out_proj(h),
            "session": self.out_proj(session_emb),
        }

    def _session_readout(
        self,
        h:          Tensor,       # [N_video, hidden_dim]
        edge_index: Tensor,       # [2, E]
        session_id: Tensor,       # [E]
        n_sessions: int,
    ) -> Tensor:
        """Aggregate per-session embeddings with soft attention + recency gate.

        Accumulates results into Python lists and stacks at the end to avoid
        in-place index assignments (e.g. tensor[i] = ...) which corrupt the
        autograd graph and cause RuntimeError during backward.
        """
        hidden_dim  = h.shape[1]
        zeros       = h.new_zeros(hidden_dim)   # reusable zero vector on same device/dtype

        session_emb_list: list[Tensor] = []
        last_item_list:   list[Tensor] = []

        for sid in range(n_sessions):
            mask = session_id == sid
            if mask.sum() == 0:
                # Session has no edges — use zero vectors (no grad needed here)
                session_emb_list.append(zeros)
                last_item_list.append(zeros)
                continue

            src_nodes = edge_index[0][mask]
            tgt_nodes = edge_index[1][mask]
            all_nodes = torch.unique(torch.cat([src_nodes, tgt_nodes]))
            node_h    = h[all_nodes]             # [n_in_session, hidden_dim]

            # Last item in temporal order = last target node
            last_h = h[tgt_nodes[-1]]            # [hidden_dim]
            last_item_list.append(last_h)

            # Soft attention: query = last item, keys = all session items
            q     = self.attn_q(last_h.unsqueeze(0))          # [1, H]
            k     = self.attn_k(node_h)                        # [n, H]
            score = self.attn_v(torch.tanh(q + k)).squeeze(-1) # [n]
            alpha = torch.softmax(score, dim=0)                 # [n]
            session_emb_list.append((alpha.unsqueeze(-1) * node_h).sum(0))

        # Stack into [n_sessions, hidden_dim] — no in-place writes, grad flows cleanly
        session_emb = torch.stack(session_emb_list, dim=0)   # [n_sessions, hidden_dim]
        last_item   = torch.stack(last_item_list,   dim=0)   # [n_sessions, hidden_dim]

        # Recency gate: blend attention readout with last-item embedding
        # Compute gate once to avoid calling recency_gate twice on the same input
        gate        = torch.sigmoid(self.recency_gate(torch.cat([session_emb, last_item], dim=-1)))
        session_emb = gate * session_emb + (1.0 - gate) * last_item

        return session_emb