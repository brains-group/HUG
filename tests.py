"""
Testing Suite
--------------
Unit and integration tests for the full KuaiRand CVR pipeline.

Run with:
    pytest tests.py -v

Each test class maps to one pipeline component.  Tests use synthetic
mini-datasets so no real KuaiRand files are required.

Test classes:
    TestKuaiRandLoader       — data_loader.py
    TestHKGConstructor       — hkg_constructor.py
    TestStructuralGNN        — gnn_encoders.py (structural branch)
    TestSequentialGNN        — gnn_encoders.py (sequential branch)
    TestAlignmentModule      — models.py (AlignmentModule)
    TestCVRHead              — models.py (CVRHead + ips_bce_loss)
    TestKuaiCVRModel         — models.py (end-to-end forward pass)
    TestPipelineIntegration  — full stack smoke test
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

# ── Import pipeline modules ───────────────────────────────────────────────────
from data_loader import (
    KuaiRandLoader, KuaiRandData,
    LOG_STANDARD_EARLY, LOG_STANDARD_LATE, LOG_RANDOM,
    USER_FEATURES, VIDEO_BASIC, VIDEO_STATISTIC,
    SESSION_GAP_MS,
)
from hkg_constructor import HKGConstructor, HKGBundle
from gnn_encoders import StructuralGNN, SequentialGNN
from models import AlignmentModule, CVRHead, KuaiCVRModel

# ── Synthetic data factories ───────────────────────────────────────────────────

N_USERS   = 20
N_VIDEOS  = 50
N_AUTHORS = 10
N_CATS    = 8
N_ROWS    = 200   # interaction rows


def _make_log_df(n_rows: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    uids = rng.integers(0, N_USERS, n_rows)
    vids = rng.integers(0, N_VIDEOS, n_rows)
    # Ensure sequential timestamps per user
    base_times = rng.integers(1_650_000_000_000, 1_650_100_000_000, n_rows)

    return pd.DataFrame({
        "user_id":          uids,
        "video_id":         vids,
        "date":             20220421,
        "hourmin":          rng.integers(0, 2400, n_rows),
        "time_ms":          base_times,
        "is_click":         rng.integers(0, 2, n_rows),
        "is_like":          rng.integers(0, 2, n_rows),
        "is_follow":        rng.integers(0, 2, n_rows),
        "is_comment":       rng.integers(0, 2, n_rows),
        "is_forward":       rng.integers(0, 2, n_rows),
        "is_hate":          rng.integers(0, 2, n_rows),
        "long_view":        rng.integers(0, 2, n_rows),
        "play_time_ms":     rng.integers(1000, 30000, n_rows),
        "duration_ms":      rng.integers(5000, 60000, n_rows),
        "profile_stay_time": rng.integers(0, 5000, n_rows),
        "comment_stay_time": rng.integers(0, 3000, n_rows),
        "is_profile_enter": rng.integers(0, 2, n_rows),
        "is_rand":          rng.integers(0, 2, n_rows),
        "tab":              rng.integers(0, 15, n_rows),
    })


def _make_user_feat_df() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id":            list(range(N_USERS)),
        "user_active_degree": ["full_active", "high_active"] * (N_USERS // 2),
        "is_lowactive_period": [0] * N_USERS,
        "is_live_streamer":   [0] * N_USERS,
        "is_video_author":    [1] * N_USERS,
        "follow_user_num":    np.random.randint(0, 500, N_USERS),
        "fans_user_num":      np.random.randint(0, 1000, N_USERS),
        "friend_user_num":    np.random.randint(0, 100, N_USERS),
        "register_days":      np.random.randint(100, 2000, N_USERS),
        **{f"onehot_feat{i}": np.random.randint(0, 5, N_USERS) for i in range(18)},
    })


def _make_video_basic_df() -> pd.DataFrame:
    tags = [",".join(str(t) for t in np.random.randint(0, N_CATS, 3)) for _ in range(N_VIDEOS)]
    return pd.DataFrame({
        "video_id":      list(range(N_VIDEOS)),
        "author_id":     np.random.randint(0, N_AUTHORS, N_VIDEOS),
        "video_type":    ["NORMAL"] * N_VIDEOS,
        "upload_dt":     ["2022-04-01"] * N_VIDEOS,
        "upload_type":   ["ShortImport"] * N_VIDEOS,
        "visible_status": [1] * N_VIDEOS,
        "video_duration": np.random.randint(5000, 60000, N_VIDEOS),
        "server_width":   [720] * N_VIDEOS,
        "server_height":  [1280] * N_VIDEOS,
        "music_id":       np.random.randint(0, 100, N_VIDEOS),
        "music_type":     np.random.randint(0, 5, N_VIDEOS),
        "tag":            tags,
    })


def _make_video_stat_df() -> pd.DataFrame:
    return pd.DataFrame({
        "video_id":              list(range(N_VIDEOS)),
        "counts":                np.random.randint(10, 60, N_VIDEOS),
        "show_cnt":              np.random.uniform(50, 200, N_VIDEOS),
        "show_user_num":         np.random.uniform(40, 180, N_VIDEOS),
        "play_cnt":              np.random.uniform(5, 50, N_VIDEOS),
        "play_user_num":         np.random.uniform(4, 45, N_VIDEOS),
        "play_duration":         np.random.uniform(5000, 50000, N_VIDEOS),
        "complete_play_cnt":     np.random.uniform(0, 5, N_VIDEOS),
        "complete_play_user_num": np.random.uniform(0, 5, N_VIDEOS),
        "valid_play_cnt":        np.random.uniform(2, 30, N_VIDEOS),
        "valid_play_user_num":   np.random.uniform(2, 28, N_VIDEOS),
        "long_time_play_cnt":    np.random.uniform(1, 20, N_VIDEOS),
        "long_time_play_user_num": np.random.uniform(1, 18, N_VIDEOS),
        "short_time_play_cnt":   np.random.uniform(2, 20, N_VIDEOS),
        "short_time_play_user_num": np.random.uniform(2, 18, N_VIDEOS),
        "play_progress":         np.random.uniform(0.1, 0.9, N_VIDEOS),
        "comment_stay_duration": np.random.uniform(0, 5000, N_VIDEOS),
        "like_cnt":              np.random.uniform(0, 10, N_VIDEOS),
        "like_user_num":         np.random.uniform(0, 10, N_VIDEOS),
        "click_like_cnt":        np.random.uniform(0, 2, N_VIDEOS),
        "double_click_cnt":      np.random.uniform(0, 3, N_VIDEOS),
        "cancel_like_cnt":       np.random.uniform(0, 2, N_VIDEOS),
        "cancel_like_user_num":  np.random.uniform(0, 2, N_VIDEOS),
        "comment_cnt":           np.random.uniform(0, 5, N_VIDEOS),
        "comment_user_num":      np.random.uniform(0, 5, N_VIDEOS),
        "direct_comment_cnt":    np.random.uniform(0, 3, N_VIDEOS),
        "reply_comment_cnt":     np.random.uniform(0, 2, N_VIDEOS),
        "delete_comment_cnt":    np.random.uniform(0, 1, N_VIDEOS),
        "delete_comment_user_num": np.random.uniform(0, 1, N_VIDEOS),
        "comment_like_cnt":      np.random.uniform(0, 1, N_VIDEOS),
        "comment_like_user_num": np.random.uniform(0, 1, N_VIDEOS),
        "follow_cnt":            np.random.uniform(0, 3, N_VIDEOS),
        "follow_user_num":       np.random.uniform(0, 3, N_VIDEOS),
        "cancel_follow_cnt":     np.random.uniform(0, 1, N_VIDEOS),
        "cancel_follow_user_num": np.random.uniform(0, 1, N_VIDEOS),
        "share_cnt":             np.random.uniform(0, 2, N_VIDEOS),
        "share_user_num":        np.random.uniform(0, 2, N_VIDEOS),
        "download_cnt":          np.random.uniform(0, 1, N_VIDEOS),
        "download_user_num":     np.random.uniform(0, 1, N_VIDEOS),
        "report_cnt":            np.random.uniform(0, 0.5, N_VIDEOS),
        "report_user_num":       np.random.uniform(0, 0.5, N_VIDEOS),
        "reduce_similar_cnt":    np.random.uniform(0, 1, N_VIDEOS),
        "reduce_similar_user_num": np.random.uniform(0, 1, N_VIDEOS),
        "collect_cnt":           np.random.uniform(0, 2, N_VIDEOS),
        "collect_user_num":      np.random.uniform(0, 2, N_VIDEOS),
        "cancel_collect_cnt":    np.random.uniform(0, 1, N_VIDEOS),
        "cancel_collect_user_num": np.random.uniform(0, 1, N_VIDEOS),
        "direct_comment_user_num": np.random.uniform(0, 3, N_VIDEOS),
        "reply_comment_user_num": np.random.uniform(0, 2, N_VIDEOS),
        "share_all_cnt":         np.random.uniform(0, 2, N_VIDEOS),
        "share_all_user_num":    np.random.uniform(0, 2, N_VIDEOS),
        "outsite_share_all_cnt": np.random.uniform(0, 1, N_VIDEOS),
    })


@pytest.fixture(scope="module")
def tmp_data_dir():
    """Write synthetic CSV files to a temp directory, yield the path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_df = _make_log_df()
        rand_df = _make_log_df(seed=99)

        log_df.to_csv(os.path.join(tmpdir, LOG_STANDARD_EARLY), index=False)
        log_df.to_csv(os.path.join(tmpdir, LOG_STANDARD_LATE),  index=False)
        rand_df.to_csv(os.path.join(tmpdir, LOG_RANDOM),         index=False)
        _make_user_feat_df().to_csv(os.path.join(tmpdir, USER_FEATURES),  index=False)
        _make_video_basic_df().to_csv(os.path.join(tmpdir, VIDEO_BASIC),   index=False)
        _make_video_stat_df().to_csv(os.path.join(tmpdir, VIDEO_STATISTIC), index=False)

        yield Path(tmpdir)


@pytest.fixture(scope="module")
def loaded_data(tmp_data_dir) -> KuaiRandData:
    loader = KuaiRandLoader(tmp_data_dir, min_interactions=1, filter_ads=False)
    return loader.load()


@pytest.fixture(scope="module")
def hkg_bundle(loaded_data) -> HKGBundle:
    constructor = HKGConstructor(loaded_data, device="cpu", max_seq_len=20)
    return constructor.build()


# ── TestKuaiRandLoader ─────────────────────────────────────────────────────────

class TestKuaiRandLoader:

    def test_load_returns_kuairand_data(self, loaded_data):
        assert isinstance(loaded_data, KuaiRandData)

    def test_user_count(self, loaded_data):
        assert len(loaded_data.user_features) == N_USERS

    def test_video_count(self, loaded_data):
        assert len(loaded_data.video_basic) == N_VIDEOS

    def test_log_combined_has_both_policies(self, loaded_data):
        assert 0 in loaded_data.log_combined["is_rand"].values
        assert 1 in loaded_data.log_combined["is_rand"].values

    def test_play_ratio_in_bounds(self, loaded_data):
        ratios = loaded_data.log_combined["play_ratio"]
        assert ratios.min() >= 0.0
        assert ratios.max() <= 1.0

    def test_sessions_derived(self, loaded_data):
        assert "session_id" in loaded_data.session_map.columns
        assert loaded_data.session_map["session_id"].nunique() > 0

    def test_id_maps_contiguous(self, loaded_data):
        uid_vals = sorted(loaded_data.user_id_map.values())
        assert uid_vals == list(range(len(uid_vals)))

        vid_vals = sorted(loaded_data.video_id_map.values())
        assert vid_vals == list(range(len(vid_vals)))

    def test_global_cvr_in_stat(self, loaded_data):
        assert "global_cvr" in loaded_data.video_statistic.columns
        cvr = loaded_data.video_statistic["global_cvr"]
        assert cvr.min() >= 0.0
        assert cvr.max() <= 1.0

    def test_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            KuaiRandLoader("/nonexistent/path").load()

    def test_tag_list_parsed(self, loaded_data):
        tl = loaded_data.video_basic["tag_list"]
        assert tl.apply(lambda x: isinstance(x, list)).all()

    def test_ad_filter(self, tmp_data_dir):
        vb = _make_video_basic_df()
        vb.loc[0, "video_type"] = "AD"
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_log_df().to_csv(os.path.join(tmpdir, LOG_STANDARD_EARLY), index=False)
            _make_log_df().to_csv(os.path.join(tmpdir, LOG_STANDARD_LATE),  index=False)
            _make_log_df(seed=99).to_csv(os.path.join(tmpdir, LOG_RANDOM),  index=False)
            _make_user_feat_df().to_csv(os.path.join(tmpdir, USER_FEATURES), index=False)
            vb.to_csv(os.path.join(tmpdir, VIDEO_BASIC),                     index=False)
            _make_video_stat_df().to_csv(os.path.join(tmpdir, VIDEO_STATISTIC), index=False)

            loader = KuaiRandLoader(tmpdir, min_interactions=1, filter_ads=True)
            data   = loader.load()
            assert 0 not in data.video_basic["video_id"].values


# ── TestHKGConstructor ─────────────────────────────────────────────────────────

class TestHKGConstructor:

    def test_returns_hkg_bundle(self, hkg_bundle):
        assert isinstance(hkg_bundle, HKGBundle)

    def test_node_types_present(self, hkg_bundle):
        g = hkg_bundle.full_graph
        for nt in ["user", "video", "author", "category", "session"]:
            assert nt in g.node_types, f"Missing node type: {nt}"

    def test_user_video_edge_present(self, hkg_bundle):
        g = hkg_bundle.full_graph
        assert ("user", "interacted", "video") in g.edge_types

    def test_video_category_edge_present(self, hkg_bundle):
        g = hkg_bundle.full_graph
        assert ("video", "tagged_as", "category") in g.edge_types

    def test_sequential_edge_present(self, hkg_bundle):
        g = hkg_bundle.full_graph
        assert ("video", "next_in_session", "video") in g.edge_types

    def test_structural_subgraph_excludes_session(self, hkg_bundle):
        assert "session" not in hkg_bundle.structural_graph.node_types

    def test_sequential_subgraph_excludes_user(self, hkg_bundle):
        assert "user" not in hkg_bundle.sequential_graph.node_types

    def test_video_nodes_shared(self, hkg_bundle):
        n_struct = hkg_bundle.structural_graph["video"].num_nodes
        n_seq    = hkg_bundle.sequential_graph["video"].num_nodes
        assert n_struct == n_seq

    def test_edge_attr_shape(self, hkg_bundle):
        g  = hkg_bundle.full_graph
        et = ("user", "interacted", "video")
        assert g[et].edge_attr.shape[1] == 4  # [play_ratio, long_view, tab, ips]

    def test_is_rand_flag_on_edges(self, hkg_bundle):
        g  = hkg_bundle.full_graph
        et = ("user", "interacted", "video")
        assert hasattr(g[et], "is_rand")
        assert g[et].is_rand.dtype == torch.bool

    def test_node_counts_match_data(self, hkg_bundle, loaded_data):
        assert hkg_bundle.n_users  == len(loaded_data.user_id_map)
        assert hkg_bundle.n_videos == len(loaded_data.video_id_map)


# ── TestStructuralGNN ──────────────────────────────────────────────────────────

class TestStructuralGNN:

    @pytest.fixture
    def model(self) -> StructuralGNN:
        return StructuralGNN(
            user_cont_dim      = 8,
            user_onehot_vocab  = [5] * 18,
            video_feat_dim     = 13,
            author_feat_dim    = 1,
            category_feat_dim  = 1,
            hidden_dim         = 32,
            out_dim            = 16,
            num_relations      = 7,
            num_layers         = 2,
        )

    def _make_mini_graph(self) -> HeteroData:
        g = HeteroData()
        g["user"].x      = torch.rand(N_USERS, 8)
        g["user"].onehot = torch.randint(0, 5, (N_USERS, 18))
        g["user"].num_nodes = N_USERS
        g["video"].x     = torch.rand(N_VIDEOS, 13)
        g["video"].num_nodes = N_VIDEOS
        g["author"].x    = torch.zeros(N_AUTHORS, 1)
        g["author"].num_nodes = N_AUTHORS
        g["category"].x  = torch.zeros(N_CATS, 1)
        g["category"].num_nodes = N_CATS
        return g

    def _make_relation_edges(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Fake 50 edges with 7 relation types
        ei = torch.randint(0, N_USERS + N_VIDEOS + N_AUTHORS + N_CATS, (2, 50))
        rt = torch.randint(0, 7, (50,))
        return ei, rt

    def test_output_shapes(self, model):
        g  = self._make_mini_graph()
        ei, rt = self._make_relation_edges()
        out = model(g, ei, rt)
        assert out["user"].shape  == (N_USERS,  16)
        assert out["video"].shape == (N_VIDEOS, 16)

    def test_output_is_finite(self, model):
        g  = self._make_mini_graph()
        ei, rt = self._make_relation_edges()
        out = model(g, ei, rt)
        assert torch.isfinite(out["user"]).all()
        assert torch.isfinite(out["video"]).all()

    def test_gradients_flow(self, model):
        g  = self._make_mini_graph()
        ei, rt = self._make_relation_edges()
        out = model(g, ei, rt)
        loss = out["video"].sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"

    def test_train_eval_differ(self, model):
        """Dropout should produce different outputs in train vs eval."""
        g  = self._make_mini_graph()
        ei, rt = self._make_relation_edges()
        model.train()
        o1 = model(g, ei, rt)["video"]
        o2 = model(g, ei, rt)["video"]
        model.eval()
        o3 = model(g, ei, rt)["video"]
        # In train mode two passes differ (dropout); in eval they should match
        assert not torch.allclose(o1, o2, atol=1e-6)
        with torch.no_grad():
            o4 = model(g, ei, rt)["video"]
        assert torch.allclose(o3, o4, atol=1e-6)


# ── TestSequentialGNN ──────────────────────────────────────────────────────────

class TestSequentialGNN:

    @pytest.fixture
    def model(self) -> SequentialGNN:
        return SequentialGNN(
            video_feat_dim = 13,
            hidden_dim     = 32,
            out_dim        = 16,
            num_layers     = 2,
        )

    def _make_seq_graph(self) -> tuple[HeteroData, torch.Tensor]:
        g = HeteroData()
        g["video"].x         = torch.rand(N_VIDEOS, 13)
        g["video"].num_nodes = N_VIDEOS
        n_sessions = 10
        g["session"].num_nodes = n_sessions
        g["session"].x = torch.zeros(n_sessions, 1)

        # Build fake sequential edges
        srcs = torch.randint(0, N_VIDEOS, (40,))
        tgts = torch.randint(0, N_VIDEOS, (40,))
        g["video", "next_in_session", "video"].edge_index = torch.stack([srcs, tgts])
        g["video", "next_in_session", "video"].edge_attr  = torch.rand(40, 1)

        session_id = torch.randint(0, n_sessions, (40,))
        return g, session_id

    def test_output_shapes(self, model):
        g, sid = self._make_seq_graph()
        out = model(g, sid)
        assert out["video"].shape   == (N_VIDEOS, 16)
        assert out["session"].shape == (10, 16)

    def test_output_is_finite(self, model):
        g, sid = self._make_seq_graph()
        out = model(g, sid)
        assert torch.isfinite(out["video"]).all()
        assert torch.isfinite(out["session"]).all()

    def test_gradients_flow(self, model):
        g, sid = self._make_seq_graph()
        out    = model(g, sid)
        out["video"].sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"


# ── TestAlignmentModule ────────────────────────────────────────────────────────

class TestAlignmentModule:

    B = 16
    D = 16

    @pytest.fixture
    def module(self):
        return AlignmentModule(emb_dim=self.D, kg_relation_dim=8, num_heads=4)

    def _batch(self):
        return {
            "h_s_user":    torch.rand(self.B, self.D),
            "h_s_video":   torch.rand(self.B, self.D),
            "h_q_video":   torch.rand(self.B, self.D),
            "h_q_session": torch.rand(self.B, self.D),
            "kg_relation": torch.rand(self.B, 8),
        }

    def test_output_shape(self, module):
        b = self._batch()
        out = module(**b)
        assert out.shape == (self.B, self.D)

    def test_output_is_finite(self, module):
        b = self._batch()
        assert torch.isfinite(module(**b)).all()

    def test_without_kg_relation(self):
        m = AlignmentModule(emb_dim=self.D, kg_relation_dim=0, num_heads=4)
        b = self._batch()
        out = m(b["h_s_user"], b["h_s_video"], b["h_q_video"], b["h_q_session"], kg_relation=None)
        assert out.shape == (self.B, self.D)

    def test_gradients_flow(self, module):
        b   = self._batch()
        out = module(**b)
        out.sum().backward()
        for name, p in module.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"


# ── TestCVRHead ────────────────────────────────────────────────────────────────

class TestCVRHead:

    B = 32
    D = 16

    @pytest.fixture
    def head(self):
        return CVRHead(fused_dim=self.D, hidden_dims=[32, 16])

    def test_logit_shape(self, head):
        fused = torch.rand(self.B, self.D)
        assert head(fused).shape == (self.B,)

    def test_proba_in_range(self, head):
        fused = torch.rand(self.B, self.D)
        p = head.predict_proba(fused)
        assert p.min() >= 0.0 and p.max() <= 1.0

    def test_ips_loss_scalar(self, head):
        fused   = torch.rand(self.B, self.D)
        logits  = head(fused)
        labels  = torch.randint(0, 2, (self.B,))
        weights = torch.rand(self.B) + 0.1
        loss    = CVRHead.ips_bce_loss(logits, labels, weights)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_ips_loss_higher_for_wrong_predictions(self, head):
        """Loss with all-wrong predictions should exceed all-correct."""
        fused = torch.rand(self.B, self.D)
        logits_high = torch.full((self.B,), 10.0)    # predicts all 1
        logits_low  = torch.full((self.B,), -10.0)   # predicts all 0
        labels_ones = torch.ones(self.B)
        w = torch.ones(self.B)
        loss_correct = CVRHead.ips_bce_loss(logits_high, labels_ones, w)
        loss_wrong   = CVRHead.ips_bce_loss(logits_low,  labels_ones, w)
        assert loss_wrong > loss_correct

    def test_weight_clipping(self, head):
        logits  = torch.zeros(self.B)
        labels  = torch.zeros(self.B)
        weights_extreme = torch.full((self.B,), 1000.0)
        loss = CVRHead.ips_bce_loss(logits, labels, weights_extreme, clip_weight=10.0)
        assert torch.isfinite(loss)

    def test_gradients_flow(self, head):
        fused  = torch.rand(self.B, self.D, requires_grad=True)
        logits = head(fused)
        labels = torch.randint(0, 2, (self.B,)).float()
        loss   = CVRHead.ips_bce_loss(logits, labels, torch.ones(self.B))
        loss.backward()
        assert fused.grad is not None and torch.isfinite(fused.grad).all()


# ── TestKuaiCVRModel ───────────────────────────────────────────────────────────

class TestKuaiCVRModel:

    D = 16
    B = 8

    @pytest.fixture
    def model(self):
        struct = StructuralGNN(
            user_cont_dim=8, user_onehot_vocab=[5]*18,
            video_feat_dim=13, author_feat_dim=1, category_feat_dim=1,
            hidden_dim=32, out_dim=self.D, num_relations=7, num_layers=1,
        )
        seq = SequentialGNN(
            video_feat_dim=13, hidden_dim=32, out_dim=self.D, num_layers=1,
        )
        align = AlignmentModule(emb_dim=self.D, kg_relation_dim=8, num_heads=4)
        head  = CVRHead(fused_dim=self.D, hidden_dims=[32])
        return KuaiCVRModel(struct, seq, align, head)

    def _make_inputs(self):
        n_sess = 5
        # Structural graph
        sg = HeteroData()
        sg["user"].x = torch.rand(N_USERS, 8)
        sg["user"].onehot = torch.randint(0, 5, (N_USERS, 18))
        sg["user"].num_nodes = N_USERS
        sg["video"].x = torch.rand(N_VIDEOS, 13)
        sg["video"].num_nodes = N_VIDEOS
        sg["author"].x = torch.zeros(N_AUTHORS, 1)
        sg["author"].num_nodes = N_AUTHORS
        sg["category"].x = torch.zeros(N_CATS, 1)
        sg["category"].num_nodes = N_CATS

        # Sequential graph
        qg = HeteroData()
        qg["video"].x = sg["video"].x
        qg["video"].num_nodes = N_VIDEOS
        qg["session"].num_nodes = n_sess
        qg["session"].x = torch.zeros(n_sess, 1)
        srcs = torch.randint(0, N_VIDEOS, (20,))
        tgts = torch.randint(0, N_VIDEOS, (20,))
        qg["video", "next_in_session", "video"].edge_index = torch.stack([srcs, tgts])
        qg["video", "next_in_session", "video"].edge_attr  = torch.rand(20, 1)

        rel_ei = torch.randint(0, N_USERS + N_VIDEOS + N_AUTHORS + N_CATS, (2, 50))
        rel_t  = torch.randint(0, 7, (50,))
        sess_id = torch.randint(0, n_sess, (20,))

        return {
            "structural_graph":    sg,
            "sequential_graph":    qg,
            "relation_edge_index": rel_ei,
            "relation_types":      rel_t,
            "session_id":          sess_id,
            "user_idx":            torch.randint(0, N_USERS,  (self.B,)),
            "video_idx":           torch.randint(0, N_VIDEOS, (self.B,)),
            "session_idx":         torch.randint(0, n_sess,   (self.B,)),
            "kg_relation":         torch.rand(self.B, 8),
            "ips_weights":         torch.rand(self.B) + 0.1,
            "labels":              torch.randint(0, 2, (self.B,)),
        }

    def test_forward_output_keys(self, model):
        inputs = self._make_inputs()
        out    = model(**inputs)
        for key in ["logits", "proba", "loss", "fused"]:
            assert key in out, f"Missing key: {key}"

    def test_logit_shape(self, model):
        out = model(**self._make_inputs())
        assert out["logits"].shape == (self.B,)

    def test_proba_range(self, model):
        out = model(**self._make_inputs())
        p   = out["proba"]
        assert p.min() >= 0.0 and p.max() <= 1.0

    def test_loss_is_finite(self, model):
        out = model(**self._make_inputs())
        assert torch.isfinite(out["loss"])

    def test_backward_pass(self, model):
        out = model(**self._make_inputs())
        out["loss"].backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad: {name}"

    def test_no_loss_without_labels(self, model):
        inputs = self._make_inputs()
        inputs.pop("labels")
        inputs.pop("ips_weights")
        out = model(**inputs)
        assert "loss" not in out


# ── TestPipelineIntegration ────────────────────────────────────────────────────

class TestPipelineIntegration:
    """Full stack smoke test: loader → HKG → GNNs → alignment → CVR head."""

    def test_full_pipeline_runs(self, loaded_data, hkg_bundle):
        """Verify that a mini training step completes without errors."""
        D = 16
        struct = StructuralGNN(
            user_cont_dim      = loaded_data.user_features[
                [c for c in loaded_data.user_features.columns
                 if c.endswith("_log") or c in ("activity_level","is_lowactive_period",
                                                 "is_live_streamer","is_video_author")]
            ].shape[1],
            user_onehot_vocab  = [5] * 18,
            video_feat_dim     = hkg_bundle.structural_graph["video"].x.shape[1],
            author_feat_dim    = 1,
            category_feat_dim  = 1,
            hidden_dim=32, out_dim=D, num_relations=7, num_layers=1,
        )
        seq = SequentialGNN(
            video_feat_dim = hkg_bundle.sequential_graph["video"].x.shape[1],
            hidden_dim=32, out_dim=D, num_layers=1,
        )
        align = AlignmentModule(emb_dim=D, kg_relation_dim=0, num_heads=4)
        head  = CVRHead(fused_dim=D, hidden_dims=[32])
        model = KuaiCVRModel(struct, seq, align, head)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Build minimal relation edge index for R-GCN
        sg      = hkg_bundle.structural_graph
        qg      = hkg_bundle.sequential_graph
        n_total = (sg["user"].num_nodes + sg["video"].num_nodes +
                   sg["author"].num_nodes + sg["category"].num_nodes)

        rel_ei = torch.randint(0, max(n_total, 1), (2, 30))
        rel_t  = torch.randint(0, 7, (30,))

        n_sess  = max(hkg_bundle.sequential_graph["session"].num_nodes, 1)
        n_edges = qg["video", "next_in_session", "video"].edge_index.shape[1]
        sess_id = torch.randint(0, n_sess, (n_edges,))

        B = 4
        out = model(
            structural_graph    = sg,
            sequential_graph    = qg,
            relation_edge_index = rel_ei,
            relation_types      = rel_t,
            session_id          = sess_id,
            user_idx            = torch.randint(0, hkg_bundle.n_users,  (B,)),
            video_idx           = torch.randint(0, hkg_bundle.n_videos, (B,)),
            session_idx         = torch.randint(0, n_sess, (B,)),
            kg_relation         = None,
            ips_weights         = torch.ones(B),
            labels              = torch.randint(0, 2, (B,)),
        )

        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()

        assert torch.isfinite(out["loss"]), "Integration loss is not finite"
        assert out["proba"].shape == (B,)

    def test_hkg_structural_sequential_video_features_identical(self, hkg_bundle):
        """Video node feature tensors must be shared between subgraphs."""
        x_struct = hkg_bundle.structural_graph["video"].x
        x_seq    = hkg_bundle.sequential_graph["video"].x
        assert torch.allclose(x_struct, x_seq)


# ── TestRealData ───────────────────────────────────────────────────────────────

class TestRealData:
    """
    End-to-end tests that run against the real KuaiRand-1K files.

    All tests in this class are skipped automatically when --data-dir is not
    supplied on the pytest command line.

    Run with:
        pytest tests.py -v --data-dir /path/to/KuaiRand-1K/data

    Tests are grouped into three phases:
        Phase 1 — Data loading      : scale, schema, and value-range checks
        Phase 2 — HKG construction  : graph structure and connectivity checks
        Phase 3 — Model forward pass: a full training step on real embeddings
    """

    # ── Shared fixtures (session-scoped so files are read once) ────────────────

    @pytest.fixture(scope="class")
    def real_data(self, real_data_dir) -> KuaiRandData:
        if real_data_dir is None:
            pytest.skip("--data-dir not provided; skipping real-data tests")
        loader = KuaiRandLoader(real_data_dir, min_interactions=10, filter_ads=True)
        return loader.load()

    @pytest.fixture(scope="class")
    def real_bundle(self, real_data, device) -> HKGBundle:
        constructor = HKGConstructor(real_data, device=str(device), max_seq_len=50)
        return constructor.build()

    # ── Phase 1: Data loading ──────────────────────────────────────────────────

    def test_real_user_count(self, real_data):
        """KuaiRand-1K has exactly 1,000 users before interaction filtering."""
        # After min_interactions filter some users may be dropped, but the
        # raw user_features table must have at most 1,000 rows.
        assert 1 <= len(real_data.user_features) <= 1_000

    def test_real_video_count(self, real_data):
        """KuaiRand-1K has ~4.4M videos; after AD filtering still > 100K."""
        assert len(real_data.video_basic) > 100_000, (
            f"Expected >100K videos after AD filter, got {len(real_data.video_basic)}"
        )

    def test_real_interaction_count(self, real_data):
        """Combined log should have millions of interactions."""
        n = len(real_data.log_combined)
        assert n > 1_000_000, f"Expected >1M combined interactions, got {n}"

    def test_real_random_policy_present(self, real_data):
        """Random-policy interactions (is_rand=1) must exist."""
        rand_rows = (real_data.log_combined["is_rand"] == 1).sum()
        assert rand_rows > 0, "No random-policy interactions found"

    def test_real_is_click_binary(self, real_data):
        """is_click must be 0 or 1 only."""
        unique = set(real_data.log_combined["is_click"].unique())
        assert unique <= {0, 1}, f"Unexpected is_click values: {unique}"

    def test_real_play_ratio_bounds(self, real_data):
        """play_ratio derived field must be in [0, 1]."""
        pr = real_data.log_combined["play_ratio"]
        assert pr.min() >= 0.0, f"play_ratio min = {pr.min()}"
        assert pr.max() <= 1.0, f"play_ratio max = {pr.max()}"

    def test_real_global_cvr_bounds(self, real_data):
        """Global CVR per video must be in [0, 1]."""
        cvr = real_data.video_statistic["global_cvr"]
        assert cvr.min() >= 0.0
        assert cvr.max() <= 1.0

    def test_real_sessions_derived(self, real_data):
        """Session map must exist and cover all interactions."""
        sm = real_data.session_map
        assert len(sm) == len(real_data.log_combined), (
            "session_map row count does not match log_combined"
        )
        assert sm["session_id"].nunique() > 1_000, (
            f"Expected >1K sessions, got {sm['session_id'].nunique()}"
        )

    def test_real_id_maps_contiguous(self, real_data):
        """Re-indexed IDs must be contiguous from 0."""
        uid_vals = sorted(real_data.user_id_map.values())
        assert uid_vals == list(range(len(uid_vals))), "user_id_map not contiguous"
        vid_vals = sorted(real_data.video_id_map.values())
        assert vid_vals == list(range(len(vid_vals))), "video_id_map not contiguous"

    def test_real_no_null_user_ids_in_log(self, real_data):
        assert real_data.log_combined["user_id"].isna().sum() == 0

    def test_real_no_null_video_ids_in_log(self, real_data):
        assert real_data.log_combined["video_id"].isna().sum() == 0

    def test_real_tag_lists_populated(self, real_data):
        """At least half the videos should have at least one category tag."""
        has_tag = real_data.video_basic["tag_list"].apply(lambda x: len(x) > 0)
        frac    = has_tag.mean()
        assert frac > 0.5, f"Only {frac:.1%} of videos have tags — check tag parsing"

    def test_real_onehot_columns_present(self, real_data):
        """All 18 encrypted user features must be in the user_features table."""
        for i in range(18):
            assert f"onehot_feat{i}" in real_data.user_features.columns, (
                f"Missing onehot_feat{i}"
            )

    # ── Phase 2: HKG construction ──────────────────────────────────────────────

    def test_real_hkg_node_types(self, real_bundle):
        for nt in ["user", "video", "author", "category", "session"]:
            assert nt in real_bundle.full_graph.node_types, f"Missing node type: {nt}"

    def test_real_hkg_edge_types(self, real_bundle):
        g = real_bundle.full_graph
        for et in [
            ("user", "interacted",  "video"),
            ("user", "clicked",     "video"),
            ("video", "tagged_as",  "category"),
            ("video", "made_by",    "author"),
            ("video", "next_in_session", "video"),
        ]:
            assert et in g.edge_types, f"Missing edge type: {et}"

    def test_real_hkg_user_count(self, real_bundle, real_data):
        assert real_bundle.n_users == len(real_data.user_id_map)

    def test_real_hkg_video_count(self, real_bundle, real_data):
        assert real_bundle.n_videos == len(real_data.video_id_map)

    def test_real_hkg_has_sessions(self, real_bundle):
        assert real_bundle.n_sessions > 1_000, (
            f"Expected >1K sessions, got {real_bundle.n_sessions}"
        )

    def test_real_hkg_clicked_edges_subset_of_interacted(self, real_bundle):
        """Clicked edge count must be <= total interacted edge count."""
        g = real_bundle.full_graph
        n_interacted = g["user", "interacted", "video"].edge_index.shape[1]
        n_clicked    = g["user", "clicked",    "video"].edge_index.shape[1]
        assert n_clicked <= n_interacted, (
            f"clicked ({n_clicked}) > interacted ({n_interacted}) — impossible"
        )

    def test_real_hkg_ips_weights_present(self, real_bundle):
        et = ("user", "interacted", "video")
        g  = real_bundle.full_graph
        assert hasattr(g[et], "is_rand"), "is_rand flag missing on interaction edges"

    def test_real_hkg_video_features_finite(self, real_bundle):
        x = real_bundle.full_graph["video"].x
        assert torch.isfinite(x).all(), "Non-finite values in video node features"

    def test_real_hkg_user_features_finite(self, real_bundle):
        x = real_bundle.full_graph["user"].x
        assert torch.isfinite(x).all(), "Non-finite values in user node features"

    def test_real_structural_excludes_session(self, real_bundle):
        assert "session" not in real_bundle.structural_graph.node_types

    def test_real_sequential_excludes_user(self, real_bundle):
        assert "user" not in real_bundle.sequential_graph.node_types

    def test_real_video_features_shared_across_subgraphs(self, real_bundle):
        x_struct = real_bundle.structural_graph["video"].x
        x_seq    = real_bundle.sequential_graph["video"].x
        assert torch.allclose(x_struct, x_seq), (
            "Video features differ between structural and sequential subgraphs"
        )

    def test_real_seq_edge_index_in_bounds(self, real_bundle):
        """next_in_session edge indices must not exceed video node count."""
        ei      = real_bundle.sequential_graph["video", "next_in_session", "video"].edge_index
        n_video = real_bundle.sequential_graph["video"].num_nodes
        assert ei.max() < n_video, (
            f"Sequential edge index {ei.max()} >= n_video {n_video}"
        )

    # ── Phase 3: Model forward + backward on real embeddings ──────────────────

    def test_real_model_forward_backward(self, real_bundle, real_data, device):
        """One forward + backward step using real graph dimensions and the target device."""
        sg = real_bundle.structural_graph
        qg = real_bundle.sequential_graph

        user_cont_cols = [
            c for c in real_data.user_features.columns
            if c.endswith("_log") or c in (
                "activity_level", "is_lowactive_period",
                "is_live_streamer", "is_video_author",
            )
        ]
        user_cont_dim  = len(user_cont_cols)
        video_feat_dim = sg["video"].x.shape[1]
        D = 32

        # Infer actual onehot vocab sizes from real data — do NOT hardcode.
        # Each onehot_featN column may have a different max value in the real dataset.
        onehot_cols   = [f"onehot_feat{i}" for i in range(18)
                         if f"onehot_feat{i}" in real_data.user_features.columns]
        onehot_vocabs = [
            int(real_data.user_features[col].max()) + 1
            for col in onehot_cols
        ]

        struct = StructuralGNN(
            user_cont_dim     = user_cont_dim,
            user_onehot_vocab = onehot_vocabs,
            video_feat_dim    = video_feat_dim,
            author_feat_dim   = 1,
            category_feat_dim = 1,
            hidden_dim        = 64,
            out_dim           = D,
            num_relations     = 7,
            num_layers        = 2,
        )
        seq = SequentialGNN(
            video_feat_dim = video_feat_dim,
            hidden_dim     = 64,
            out_dim        = D,
            num_layers     = 2,
        )
        align = AlignmentModule(emb_dim=D, kg_relation_dim=0, num_heads=4)
        head  = CVRHead(fused_dim=D, hidden_dims=[64, 32])
        model = KuaiCVRModel(struct, seq, align, head).to(device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Build merged relation edge index for R-GCN, moved to device
        n_user   = sg["user"].num_nodes
        n_video  = sg["video"].num_nodes
        offsets  = {"user": 0, "video": n_user, "author": n_user + n_video}

        rel_parts, rel_type_parts = [], []
        rel_map = {
            ("user",  "clicked",    "video"): 0,
            ("user",  "liked",      "video"): 1,
            ("user",  "hated",      "video"): 2,
            ("user",  "commented",  "video"): 3,
            ("user",  "forwarded",  "video"): 4,
            ("video", "made_by",    "author"): 5,
            ("video", "tagged_as",  "category"): 6,
        }
        for et, rel_id in rel_map.items():
            if et not in sg.edge_types:
                continue
            ei      = sg[et].edge_index.cpu()   # start on CPU for indexing
            shifted = ei.clone()
            shifted[0] += offsets.get(et[0], 0)
            shifted[1] += offsets.get(et[2], n_user + n_video)
            if shifted.shape[1] > 2000:
                idx     = torch.randperm(shifted.shape[1])[:2000]
                shifted = shifted[:, idx]
            rel_parts.append(shifted)
            rel_type_parts.append(torch.full((shifted.shape[1],), rel_id, dtype=torch.long))

        if rel_parts:
            rel_ei = torch.cat(rel_parts,      dim=1).to(device)
            rel_t  = torch.cat(rel_type_parts, dim=0).to(device)
        else:
            rel_ei = torch.zeros(2, 0, dtype=torch.long, device=device)
            rel_t  = torch.zeros(0, dtype=torch.long, device=device)

        # Real sequential edges — already on device from HKGConstructor
        seq_et  = ("video", "next_in_session", "video")
        sess_id = qg[seq_et].session_id.to(device)
        n_sess  = max(real_bundle.n_sessions, 1)

        B = 16
        out = model(
            structural_graph    = sg,
            sequential_graph    = qg,
            relation_edge_index = rel_ei,
            relation_types      = rel_t,
            session_id          = sess_id,
            user_idx            = torch.randint(0, real_bundle.n_users,  (B,), device=device),
            video_idx           = torch.randint(0, real_bundle.n_videos, (B,), device=device),
            session_idx         = torch.randint(0, n_sess, (B,), device=device),
            kg_relation         = None,
            ips_weights         = torch.ones(B, device=device),
            labels              = torch.randint(0, 2, (B,), device=device),
        )

        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()

        assert torch.isfinite(out["loss"]), f"Loss is not finite: {out['loss']}"
        assert out["proba"].shape == (B,)
        assert out["proba"].min() >= 0.0 and out["proba"].max() <= 1.0

        grad_norms = [
            p.grad.norm().item()
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert len(grad_norms) > 0, "No gradients computed"
        assert all(torch.isfinite(torch.tensor(g)) for g in grad_norms), (
            "Non-finite gradients found"
        )