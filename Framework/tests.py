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

import json
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
    SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST,
    assign_split, chronological_cutoffs, compute_video_statistics,
)
from hkg_constructor import HKGConstructor, HKGBundle, snapshot_bundle, video_feature_matrix
from temporal import (
    PrefixBatcher, SnapshotStore, assign_snapshots, build_interactions, snapshot_boundaries,
)
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
def cutoffs(loaded_data) -> tuple[int, int]:
    return chronological_cutoffs(loaded_data.log_combined["time_ms"].values)


@pytest.fixture(scope="module")
def timed_bundle(loaded_data) -> HKGBundle:
    return HKGConstructor(loaded_data, device="cpu").build()


@pytest.fixture(scope="module")
def hkg_bundle(timed_bundle, cutoffs) -> HKGBundle:
    """Snapshot as of the validation cutoff — what training-period rows near t_val see."""
    return snapshot_bundle(timed_bundle, cutoffs[0])


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
        stats = compute_video_statistics(loaded_data.log_combined)
        assert "global_cvr" in stats.columns
        cvr = stats["global_cvr"]
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


# ── TestCausality ──────────────────────────────────────────────────────────────

def _edge_pairs(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return set(map(tuple, edge_index.t().tolist()))


@pytest.fixture(scope="module")
def inter(loaded_data):
    return build_interactions(loaded_data, max_prefix=5)


@pytest.fixture(scope="module")
def boundaries(inter):
    return snapshot_boundaries(inter.time, inter.t_val, inter.t_test, period_hours=4)


class TestCausality:
    """Every input for an interaction at time t must come from interactions before t."""

    def test_split_is_chronological(self, loaded_data, cutoffs):
        t = loaded_data.log_combined["time_ms"].values
        split = assign_split(t, *cutoffs)
        assert t[split == SPLIT_TRAIN].max() < t[split == SPLIT_VAL].min()
        assert t[split == SPLIT_VAL].max()   < t[split == SPLIT_TEST].min()

    def test_interaction_split_matches_shared_cutoffs(self, inter, loaded_data):
        assert (inter.t_val, inter.t_test) == chronological_cutoffs(inter.time)
        assert np.all(np.diff(inter.time) >= 0)

    def test_boundaries_include_split_cutoffs(self, boundaries, inter):
        assert inter.t_val in boundaries and inter.t_test in boundaries

    def test_snapshot_starts_before_row(self, inter, boundaries):
        k = assign_snapshots(inter.time, boundaries)
        assert np.all(boundaries[k] <= inter.time)
        # no snapshot period mixes splits
        for kk in np.unique(k):
            assert len(np.unique(inter.split[k == kk])) == 1

    def test_timed_edges_carry_time(self, timed_bundle):
        g = timed_bundle.full_graph
        for et in g.edge_types:
            if et in [("video", "made_by", "author"), ("video", "tagged_as", "category")]:
                continue
            assert "edge_time" in g[et], et

    def test_snapshot_edges_strictly_before_cutoff(self, timed_bundle, boundaries):
        for b in boundaries:
            view = snapshot_bundle(timed_bundle, int(b))
            for g in (view.full_graph, view.structural_graph, view.sequential_graph):
                for et in g.edge_types:
                    if "edge_time" in g[et]:
                        assert (g[et].edge_time < b).all(), (et, b)

    def test_snapshot_pairs_from_past_only(self, timed_bundle, loaded_data, cutoffs):
        b   = cutoffs[0]
        log = loaded_data.log_combined
        past = log[log["time_ms"] < b]
        pairs = set(zip(past["user_id"].map(loaded_data.user_id_map).tolist(),
                        past["video_id"].map(loaded_data.video_id_map).tolist()))
        g = snapshot_bundle(timed_bundle, b).full_graph
        for et in g.edge_types:
            if et[0] == "user" and et[2] == "video":
                assert _edge_pairs(g[et].edge_index) <= pairs, et

    def test_store_video_features_from_past_only(self, timed_bundle, loaded_data, boundaries):
        store = SnapshotStore(timed_bundle, loaded_data, boundaries, rel_master=None)
        vb = loaded_data.video_basic.sort_values("video_id")
        log = loaded_data.log_combined
        for k, b in enumerate(boundaries):
            expected = video_feature_matrix(vb, compute_video_statistics(log[log["time_ms"] < b]))
            assert np.allclose(store.video_x[k].numpy(), expected)
        assert np.allclose(store.video_x[0].numpy()[:, 2:], 0.0)  # nothing before the first row

    def test_session_prefix_is_strictly_earlier(self, inter, loaded_data):
        sm = loaded_data.session_map
        sm = sm[sm["user_id"].isin(loaded_data.user_id_map) & sm["video_id"].isin(loaded_data.video_id_map)]
        sm = sm.sort_values("time_ms", kind="stable").reset_index(drop=True)
        sess = sm["session_id"].values
        batch = PrefixBatcher(inter)(np.arange(len(inter.time)))
        prefix = batch["prefix_items"].numpy()
        assert (prefix >= 0).any(axis=1).sum() > 0, "fixture has no non-empty prefixes"
        for i in range(len(inter.time)):
            earlier = np.flatnonzero((sess == sess[i]) & (inter.time < inter.time[i]))
            earlier = earlier[np.argsort(inter.time[earlier], kind="stable")][-inter.max_prefix:]
            got = prefix[i][prefix[i] >= 0]
            assert sorted(got.tolist()) == sorted(inter.video[earlier].tolist()), i

    def test_predictions_invariant_to_future(self, loaded_data, inter, boundaries):
        """Rewrite everything at or after T; scores of rows before T must not move."""
        import copy
        import main as pipeline

        T = int(np.quantile(inter.time, 0.6))

        def scores(data):
            it     = build_interactions(data, max_prefix=5)
            bundle = HKGConstructor(data, device="cpu").build()
            snaps  = assign_snapshots(it.time, boundaries)
            store  = SnapshotStore(
                bundle, data, boundaries,
                rel_master=pipeline.build_relation_edge_index(bundle.structural_graph))
            batcher = PrefixBatcher(it)
            out = np.empty(len(it.time))
            model.eval()
            with torch.no_grad():
                for k in np.unique(snaps):
                    rows = np.flatnonzero(snaps == k)
                    emb  = pipeline.encode_snapshot(model, store, int(k), "dual")
                    b    = batcher(rows)
                    out[rows] = model.forward_from_embeddings(
                        emb, b["user_idx"], b["video_idx"], b["prefix_items"])["proba"].numpy()
            return out, it.time

        torch.manual_seed(0)
        model = pipeline.build_model(loaded_data, HKGConstructor(loaded_data).build(),
                                     hidden_dim=16, out_dim=8, device=torch.device("cpu"),
                                     num_layers=2)

        base, t = scores(loaded_data)

        future = copy.copy(loaded_data)
        rng = np.random.default_rng(1)
        for name in ("log_combined", "session_map"):
            df = getattr(loaded_data, name).copy()
            late = df["time_ms"].values >= T
            for col in ["is_click", "is_like", "is_follow", "is_comment",
                        "is_forward", "is_hate", "long_view"]:
                df.loc[late, col] = 1 - df.loc[late, col]
            df.loc[late, "video_id"] = rng.permutation(df.loc[late, "video_id"].values)
            setattr(future, name, df)
        perturbed, t2 = scores(future)

        assert np.array_equal(t, t2)
        early = t < T
        assert early.any() and (~early).any()
        np.testing.assert_array_equal(base[early], perturbed[early])
        assert not np.allclose(base[~early], perturbed[~early])   # the test can see changes


# ── TestGraphConnectivity (spec 01) ────────────────────────────────────────────

def _triples(ei: torch.Tensor, et: torch.Tensor) -> list[tuple[int, int, int]]:
    return sorted(zip(ei[0].tolist(), ei[1].tolist(), et.tolist()))


def _check_reverse_complete(ei, et, forward_ids: dict, reverse_ids: dict):
    for fwd_et, fid in forward_ids.items():
        if fwd_et not in reverse_ids:
            continue
        rid = reverse_ids[fwd_et]
        fwd = ei[:, et == fid]
        rev = ei[:, et == rid]
        assert fwd.shape[1] == rev.shape[1], fwd_et
        assert sorted(zip(fwd[1].tolist(), fwd[0].tolist())) == \
               sorted(zip(rev[0].tolist(), rev[1].tolist())), fwd_et


class TestGraphConnectivity:
    """Reverse relations, author/category identity, uncapped time-masked edges."""

    @pytest.fixture(scope="class")
    def master(self, timed_bundle):
        import main as pipeline
        return pipeline.build_relation_edge_index(timed_bundle.structural_graph)

    def test_reverse_completeness(self, master):
        import main as pipeline
        ei, et, _ = master
        assert et.max().item() < pipeline.NUM_RELATIONS
        _check_reverse_complete(ei, et, pipeline.REL_MAP, pipeline.REVERSE_REL_MAP)

    def test_reverse_edges_are_causal(self, master, timed_bundle, loaded_data, boundaries):
        import main as pipeline
        ei, et, etime = master
        store = SnapshotStore(timed_bundle, loaded_data, boundaries, rel_master=master)
        for k, b in enumerate(boundaries):
            keep = etime < b
            kei, ket = store.rel(k)
            assert torch.equal(kei, ei[:, keep]) and torch.equal(ket, et[keep])
            # every kept behavioural edge, forward or reverse, is older than the cutoff
            behavioural = etime[keep] != pipeline.STATIC_EDGE_TIME
            assert (etime[keep][behavioural] < b).all()
            # a reverse edge survives exactly when its forward edge does
            _check_reverse_complete(kei, ket, pipeline.REL_MAP, pipeline.REVERSE_REL_MAP)

    def test_mask_equals_per_view_build(self, master, timed_bundle, boundaries):
        import main as pipeline
        ei, et, etime = master
        for b in boundaries[[0, len(boundaries) // 2, -1]]:
            keep = etime < b
            view = snapshot_bundle(timed_bundle, int(b))
            vei, vet, _ = pipeline.build_relation_edge_index(view.structural_graph)
            assert _triples(ei[:, keep], et[keep]) == _triples(vei, vet)

    def test_hgt_reverse_parity(self, timed_bundle):
        import main as pipeline
        from gnn_encoders import SingleHGT
        ei, et, _ = pipeline.build_full_relation_edge_index(timed_bundle)
        assert et.max().item() < SingleHGT.NUM_EDGE_TYPES
        forward_ids = {e: i for i, e in enumerate(SingleHGT.EDGE_TYPES)}
        _check_reverse_complete(ei, et, forward_ids, SingleHGT.REVERSE_TYPE_IDS)
        seq_id = forward_ids[("video", "next_in_session", "video")]
        assert ("video", "next_in_session", "video") not in SingleHGT.REVERSE_TYPE_IDS
        assert (et == seq_id).any()

    def _encoder(self, loaded_data, bundle, num_layers):
        import main as pipeline
        torch.manual_seed(0)
        model = pipeline.build_model(loaded_data, bundle, hidden_dim=16, out_dim=8,
                                     device=torch.device("cpu"), num_layers=num_layers)
        model.eval()
        return model.structural_gnn

    def test_users_receive_graph_signal(self, master, timed_bundle, loaded_data):
        import main as pipeline
        ei, et, _ = master
        sg = timed_bundle.structural_graph
        clicked = pipeline.REL_MAP[("user", "clicked", "video")]
        u = int(ei[0, et == clicked][0])                     # a user with clicks

        enc = self._encoder(loaded_data, timed_bundle, num_layers=2)
        with torch.no_grad():
            base = enc(sg, ei, et)["user"][u]
            drop = ~(((et == clicked) & (ei[0] == u)) |
                     ((et == clicked + len(pipeline.REL_MAP)) & (ei[1] == u)))
            assert not torch.allclose(enc(sg, ei[:, drop], et[drop])["user"][u], base)

        # L=1: a user with no incident edges equals the empty-graph output
        enc1 = self._encoder(loaded_data, timed_bundle, num_layers=1)
        iso = ~((ei[0] == u) | (ei[1] == u))
        with torch.no_grad():
            isolated = enc1(sg, ei[:, iso], et[iso])["user"][u]
            empty    = enc1(sg, ei[:, :0], et[:0])["user"][u]
        assert torch.allclose(isolated, empty, atol=1e-6)

    def test_kg_reaches_videos_and_users(self, master, timed_bundle, loaded_data):
        ei, et, _ = master
        sg = timed_bundle.structural_graph
        n_user, n_video = sg["user"].num_nodes, sg["video"].num_nodes
        a = 0
        a_node = n_user + n_video + a

        enc = self._encoder(loaded_data, timed_bundle, num_layers=2)
        with torch.no_grad():
            base = enc(sg, ei, et)
            enc.author_emb.weight[a] += 1.0
            pert = enc(sg, ei, et)

        # nodes reachable from the author in <= 2 message-passing steps
        hop1 = set(ei[1, ei[0] == a_node].tolist())
        hop2 = set(ei[1, torch.isin(ei[0], torch.tensor(sorted(hop1)))].tolist()) if hop1 else set()
        reach = hop1 | hop2

        a_videos = [v for v in hop1 if n_user <= v < n_user + n_video]
        assert a_videos, "fixture author has no videos"
        for v in a_videos:
            assert not torch.allclose(base["video"][v - n_user], pert["video"][v - n_user])
        a_users = [x for x in hop2 if x < n_user]
        assert a_users, "no user clicked the author's videos"
        for x in a_users:
            assert not torch.allclose(base["user"][x], pert["user"][x])

        for x in range(n_user):
            if x not in reach:
                assert torch.equal(base["user"][x], pert["user"][x])
        for v in range(n_video):
            if n_user + v not in reach:
                assert torch.equal(base["video"][v], pert["video"][v])


# ── TestStructuralGNN ──────────────────────────────────────────────────────────

class TestStructuralGNN:

    @pytest.fixture
    def model(self) -> StructuralGNN:
        return StructuralGNN(
            user_cont_dim      = 8,
            user_onehot_vocab  = [5] * 18,
            video_feat_dim     = 13,
            n_authors          = N_AUTHORS,
            n_categories       = N_CATS,
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
            video_feat_dim=13, n_authors=N_AUTHORS, n_categories=N_CATS,
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
            n_authors          = hkg_bundle.n_authors,
            n_categories       = hkg_bundle.n_categories,
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
    def real_cutoffs(self, real_data) -> tuple[int, int]:
        return chronological_cutoffs(real_data.log_combined["time_ms"].values)

    @pytest.fixture(scope="class")
    def real_bundle(self, real_data, real_cutoffs) -> HKGBundle:
        """Snapshot as of the validation cutoff."""
        timed = HKGConstructor(real_data, device="cpu").build()
        return snapshot_bundle(timed, real_cutoffs[0])

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

    def test_real_global_cvr_bounds(self, real_data, real_cutoffs):
        """Global CVR per video must be in [0, 1]."""
        log = real_data.log_combined
        cvr = compute_video_statistics(log[log["time_ms"] < real_cutoffs[0]])["global_cvr"]
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
        """
        One training step through the pipeline's own path at real dimensions:
        build_model -> encode_graph on the t_val snapshot (reverse relations,
        subsampled for speed) -> forward_from_embeddings with real causal
        session prefixes -> backward.
        """
        import main as pipeline

        model = pipeline.build_model(real_data, real_bundle, hidden_dim=64, out_dim=32,
                                     device=device, num_layers=2)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        ei, et, _ = pipeline.build_relation_edge_index(real_bundle.structural_graph)
        idx = torch.randperm(ei.shape[1])[:20_000]
        emb = model.encode_graph(real_bundle.structural_graph, real_bundle.sequential_graph,
                                 ei[:, idx].to(device), et[idx].to(device))

        inter = build_interactions(real_data, max_prefix=50)
        rows  = np.random.default_rng(0).choice(inter.rows(SPLIT_TRAIN), size=64, replace=False)
        batch = {k: v.to(device) for k, v in PrefixBatcher(inter)(rows).items()}
        assert (batch["prefix_items"] >= 0).any(), "sampled rows have no session prefixes"

        out = model.forward_from_embeddings(
            emb, batch["user_idx"], batch["video_idx"], batch["prefix_items"],
            ips_weights=batch["ips_weight"], labels=batch["label"])
        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()

        assert torch.isfinite(out["loss"]), f"Loss is not finite: {out['loss']}"
        assert out["proba"].shape == (64,)
        assert out["proba"].min() >= 0.0 and out["proba"].max() <= 1.0

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, "No gradients computed"
        assert all(torch.isfinite(g).all() for g in grads), "Non-finite gradients found"
        # encoders are frozen: encode_graph runs under no_grad
        assert all(p.grad is None for p in model.structural_gnn.parameters())

    def test_real_no_test_pair_in_graph(self, real_data, real_bundle, real_cutoffs):
        """Sample 1000 test pairs never seen before the cutoff; none may be graph edges."""
        log = real_data.log_combined
        u = log["user_id"].map(real_data.user_id_map)
        v = log["video_id"].map(real_data.video_id_map)
        test = log["time_ms"].values >= real_cutoffs[1]
        early = log["time_ms"].values < real_cutoffs[0]
        pairs_early = set(zip(u[early].tolist(), v[early].tolist()))
        test_pairs = [p for p in zip(u[test].tolist(), v[test].tolist())
                      if p not in pairs_early]
        rng = np.random.default_rng(0)
        idx = rng.choice(len(test_pairs), size=min(1000, len(test_pairs)), replace=False)
        sample = {test_pairs[i] for i in idx}
        g = real_bundle.full_graph
        for et in g.edge_types:
            if et[0] == "user" and et[2] == "video":
                ei = g[et].edge_index.cpu()
                assert not (_edge_pairs(ei) & sample), et


# ── TestHUG (spec 03/04) ───────────────────────────────────────────────────────

def _hug_args(**kw):
    import main as pipeline
    argv = ["--data-dir", "unused", "--cache-dir", "none", "--device", "cpu", "--quiet",
            "--emb-dim", "8", "--min-id-count", "2", "--snapshot-hours", "4",
            "--batch-size", "64", "--max-epochs", "2", "--seq-layers", "1", "--max-seq-len", "5"]
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        argv += [flag] if v is True else ([] if v is False else [flag, str(v)])
    return pipeline.parse_args(argv)


def _rewrite_after(data, T, seed=1):
    """Copy of `data` with every interaction at or after T rewritten."""
    import copy
    out = copy.copy(data)
    rng = np.random.default_rng(seed)
    for name in ("log_combined", "session_map"):
        df = getattr(data, name).copy()
        late = df["time_ms"].values >= T
        for col in ["is_click", "is_like", "is_follow", "is_comment",
                    "is_forward", "is_hate", "long_view"]:
            df.loc[late, col] = 1 - df.loc[late, col]
        df.loc[late, "video_id"] = rng.permutation(df.loc[late, "video_id"].values)
        setattr(out, name, df)
    return out


@pytest.fixture(scope="module")
def hug_data(loaded_data, timed_bundle):
    import hug_train
    return hug_train.prepare(_hug_args(), data=loaded_data, bundle=timed_bundle)


def _hug_model(args, d, seed=0):
    import hug_train
    torch.manual_seed(seed)
    return hug_train.build(args, d)


class TestHUG:

    def test_history_strictly_earlier(self, hug_data):
        it = hug_data.inter
        for i in range(len(it.time)):
            rows = hug_data.hist_seq[hug_data.hist_start[i]:hug_data.hist_end[i]]
            assert (it.time[rows] < it.time[i]).all()
            assert (it.user[rows] == it.user[i]).all()
            assert (it.label[rows] == 1).all()
            assert len(rows) <= 5

    def test_gradient_routing(self, hug_data):
        import hug_train
        for frozen in (False, True):
            args = _hug_args(freeze_graph=frozen)
            model = _hug_model(args, hug_data)
            snaps = hug_train.Snapshots(hug_data, model, torch.device("cpu"))
            b = hug_train.HugBatcher(hug_data, args.max_seq_len)(hug_data.train_rows[:64])
            vx = snaps.enter(int(hug_data.snap[hug_data.train_rows[0]]))
            model.train()
            out = model(b, vx, model.graph_tables(vx, train=True), train=True)
            out["loss"].backward()
            if frozen:
                assert model.gcn.a.grad is None
            else:
                assert model.gcn.a.grad is not None and model.gcn.a.grad.abs().sum() > 0
                assert model.inp.user_id.weight.grad.abs().sum() > 0
            assert model.inp.video_proj.weight.grad.abs().sum() > 0
            assert all(p.grad is not None for p in model.seq.parameters() if p.requires_grad
                       and p is not model.seq.empty)
            assert model.head[0].weight.grad.abs().sum() > 0

    def test_rellightgcn_hand_computation(self):
        from hug import RelLightGCN
        # 3 users, 3 videos; relation 0 user->video, relation 1 video->user
        rel = [("user", "video"), ("video", "user")]
        gcn = RelLightGCN(rel, num_layers=1)
        src = torch.tensor([0, 0, 1, 2]); dst = torch.tensor([0, 1, 1, 2])
        gcn.set_graph([(src, dst), (dst, src)], {"user": 3, "video": 3})
        h0 = {"user": torch.arange(6.).view(3, 2), "video": torch.arange(6., 12.).view(3, 2)}
        g, _ = gcn(h0)
        w = torch.softmax(gcn.a[0], 0)
        exp_v = torch.zeros(3, 2)
        du, dv = torch.bincount(src, minlength=3).float(), torch.bincount(dst, minlength=3).float()
        for s, t in zip(src.tolist(), dst.tolist()):
            exp_v[t] += h0["user"][s] / (dv[t] * du[s]).sqrt()
        exp_u = torch.zeros(3, 2)
        for s, t in zip(dst.tolist(), src.tolist()):
            exp_u[t] += h0["video"][s] / (du[t] * dv[s]).sqrt()
        assert torch.allclose(g["video"], (h0["video"] + w[0] * exp_v) / 2, atol=1e-6)
        assert torch.allclose(g["user"], (h0["user"] + w[1] * exp_u) / 2, atol=1e-6)
        # destination-sized == full-size (all nodes in one index space)
        full = RelLightGCN([("all", "all"), ("all", "all")], num_layers=1)
        full.set_graph([(src, dst + 3), (dst + 3, src)], {"all": 6})
        gf, _ = full({"all": torch.cat([h0["user"], h0["video"]])})
        assert torch.allclose(gf["all"], torch.cat([g["user"], g["video"]]), atol=1e-6)

    def test_snapshot_degrees_from_masked_edges(self, hug_data):
        import hug_train
        args = _hug_args()
        model = _hug_model(args, hug_data)
        snaps = hug_train.Snapshots(hug_data, model, torch.device("cpu"))
        ei_all, et_all, t_all = hug_data.store.rel_master
        for k in sorted(hug_data.store.video_x)[:4]:
            snaps.enter(int(k))
            keep = t_all < hug_data.boundaries[k]
            for r, (st, dt) in enumerate(model.gcn.relations):
                m = keep & (et_all == r)
                n_dst = snaps.counts[dt]
                exp_deg = torch.bincount(ei_all[1, m] - snaps.offsets[dt], minlength=n_dst)
                adj = model.gcn._adj[r]
                assert adj._nnz() <= int(m.sum())
                got = torch.zeros(n_dst, dtype=torch.long)
                if adj._nnz():
                    # weights are 1/sqrt(deg_dst*deg_src); recover per-dst edge multiplicity
                    src = ei_all[0, m] - snaps.offsets[st]
                    deg_src = torch.bincount(src, minlength=snaps.counts[st]).float()
                    w = adj.to_dense()
                    got = (w * deg_src.unsqueeze(0).sqrt()).sum(1).pow(2).round().long()
                assert torch.equal(got, exp_deg), (k, r)

    def test_vocab_from_training_window(self, loaded_data, timed_bundle):
        import hug_train
        # make video 0 appear only at/after t_val, many times
        d0 = hug_train.prepare(_hug_args(), data=loaded_data, bundle=timed_bundle)
        t_val = d0.inter.t_val
        data = _rewrite_after(loaded_data, 10**18)          # deep copy of frames
        for name in ("log_combined", "session_map"):
            df = getattr(data, name)
            vid = df["video_id"].values.copy()
            vid[vid == 0] = 1                              # remove video 0 everywhere …
            late = np.flatnonzero(df["time_ms"].values >= t_val)[:10]
            vid[late] = 0                                  # … then add it 10× after t_val
            df["video_id"] = vid
        d = hug_train.prepare(_hug_args(), data=data, bundle=timed_bundle)
        v0 = data.video_id_map[0]
        assert d.node_data["video_vocab"][v0] == 0          # OOV despite 10 >= min_id_count

    def test_weights_independent_of_val_test(self, loaded_data, timed_bundle, tmp_path):
        import hug_train
        from runtime import set_determinism

        def fit(data, sub):
            args = _hug_args(patience=10)
            d = hug_train.prepare(args, data=data, bundle=HKGConstructor(data).build())
            set_determinism(0)
            model = hug_train.build(args, d)
            (tmp_path / sub).mkdir()
            hug_train.train(args, d, model, torch.device("cpu"), tmp_path / sub)
            return model.state_dict(), d

        base, d = fit(loaded_data, "a")
        other, _ = fit(_rewrite_after(loaded_data, d.inter.t_val), "b")
        for k in base:
            assert torch.equal(base[k], other[k]), k

    def test_predictions_invariant_to_future(self, loaded_data, timed_bundle):
        import hug_train
        args = _hug_args()
        d = hug_train.prepare(args, data=loaded_data, bundle=timed_bundle)
        T = int(np.quantile(d.inter.time[d.val_rows], 0.5))
        torch.manual_seed(0)
        model = hug_train.build(args, d)

        def scores(dd):
            m2 = hug_train.build(args, dd)
            m2.load_state_dict(model.state_dict())
            snaps = hug_train.Snapshots(dd, m2, torch.device("cpu"))
            rows = dd.val_rows
            r, s = hug_train.predict(m2, dd, rows, snaps, hug_train.HugBatcher(dd, args.max_seq_len),
                                     64, torch.device("cpu"), True, "x")
            return pd.Series(s, index=r).sort_index()

        data2 = _rewrite_after(loaded_data, T)
        d2 = hug_train.prepare(args, data=data2, bundle=HKGConstructor(data2).build())
        a, b = scores(d), scores(d2)
        early = d.inter.time[a.index.values] < T
        assert early.any() and (~early).any()
        np.testing.assert_array_equal(a.values[early], b.values[early])
        assert not np.allclose(a.values[~early], b.values[~early])

    def test_contrastive_uses_batch_nodes_and_zero_weight(self, hug_data, monkeypatch):
        import hug, hug_train
        args = _hug_args()
        model = _hug_model(args, hug_data)
        snaps = hug_train.Snapshots(hug_data, model, torch.device("cpu"))
        b = hug_train.HugBatcher(hug_data, args.max_seq_len)(hug_data.train_rows[:64])
        vx = snaps.enter(int(hug_data.snap[hug_data.train_rows[0]]))
        seen = []
        orig = hug.info_nce
        monkeypatch.setattr(hug, "info_nce", lambda z1, z2, t: seen.append(len(z1)) or orig(z1, z2, t))
        model.train()
        torch.manual_seed(0)
        model(b, vx, model.graph_tables(vx, train=True), train=True)
        assert seen == [len(torch.unique(b["user"])), len(torch.unique(b["video"]))]

        args0 = _hug_args(cl_weight=0)
        m0 = _hug_model(args0, hug_data)
        m0.train()
        out = m0(b, vx, m0.graph_tables(vx, train=True), train=True)
        touched = torch.cat([b["video"], b["hist"][b["hist"] >= 0]])
        l2 = m0.emb_l2 * m0.inp.l2_touched(b["user"], touched) / len(b["user"])
        assert torch.equal(out["loss"], out["bce"] + l2) or torch.allclose(out["loss"], out["bce"] + l2, atol=0, rtol=0)

    def test_eval_deterministic(self, hug_data):
        import hug_train
        args = _hug_args()
        model = _hug_model(args, hug_data)
        snaps = hug_train.Snapshots(hug_data, model, torch.device("cpu"))
        bat = hug_train.HugBatcher(hug_data, args.max_seq_len)
        r1, s1 = hug_train.predict(model, hug_data, hug_data.val_rows, snaps, bat, 64, torch.device("cpu"), True, "a")
        r2, s2 = hug_train.predict(model, hug_data, hug_data.val_rows, snaps, bat, 64, torch.device("cpu"), True, "b")
        assert np.array_equal(r1, r2) and np.array_equal(s1, s2)

    def test_asof_statistics_strictly_earlier_with_ties(self):
        from features import asof_video_statistics, VIDEO_STAT_COLS, FEEDBACK_COLS
        rng = np.random.default_rng(0)
        n = 300
        fr = pd.DataFrame({"video_id": rng.integers(0, 5, n), "time_ms": rng.integers(0, 40, n)})
        for c in FEEDBACK_COLS:
            fr[c] = rng.integers(0, 2, n)
        got = asof_video_statistics(fr)
        for i in range(n):
            past = fr[(fr.video_id == fr.video_id[i]) & (fr.time_ms < fr.time_ms[i])]
            show = len(past)
            assert got[i, VIDEO_STAT_COLS.index("show_cnt_log")] == pytest.approx(np.log1p(show), abs=1e-6)
            assert got[i, VIDEO_STAT_COLS.index("play_cnt_log")] == pytest.approx(np.log1p(past.is_click.sum()), abs=1e-6)
            cvr = past.long_view.sum() / show if show else 0.0
            assert got[i, VIDEO_STAT_COLS.index("global_cvr")] == pytest.approx(cvr, abs=1e-6)

    def test_holdout_disjoint_and_excluded(self, hug_data):
        from features import holdout_rows
        it = hug_data.inter
        train_all = it.rows(SPLIT_TRAIN)
        assert set(hug_data.holdout) <= set(train_all)
        assert np.array_equal(hug_data.holdout, holdout_rows(train_all, 0.05, 0))
        assert not set(hug_data.holdout) & set(hug_data.train_rows)

    def test_resume_is_exact(self, hug_data, tmp_path):
        import hug_train
        from runtime import set_determinism

        def go(sub, epochs, resume=False):
            args = _hug_args(patience=10, max_epochs=epochs, resume=resume)
            set_determinism(0)
            model = hug_train.build(args, hug_data)
            state = hug_train.train(args, hug_data, model, torch.device("cpu"), tmp_path / sub)
            return model.state_dict(), state

        (tmp_path / "straight").mkdir(); (tmp_path / "resumed").mkdir()
        a, sa = go("straight", 3)
        go("resumed", 1)
        b, sb = go("resumed", 3, resume=True)
        for k in a:
            assert torch.equal(a[k], b[k]), k
        assert sa["best_epoch"] == sb["best_epoch"]
        assert [h["val"]["auc"] for h in sa["history"]] == [h["val"]["auc"] for h in sb["history"]]

    def test_test_guard(self, tmp_path):
        from test_guard import TestAccessDenied, authorize_test_access
        with pytest.raises(TestAccessDenied):
            authorize_test_access(tmp_path, None, False, "job", "h")
        with pytest.raises(TestAccessDenied):
            authorize_test_access(tmp_path, "bogus", False, "job", "h")
        (tmp_path / ".test_tokens").mkdir()
        (tmp_path / ".test_tokens" / "tok").write_text("")
        authorize_test_access(tmp_path, "tok", False, "job", "h")
        with pytest.raises(TestAccessDenied):                      # one-time
            authorize_test_access(tmp_path, "tok", False, "job", "h")
        authorize_test_access(tmp_path, None, True, "manual", "h2")
        log = (tmp_path / "test_access.log").read_text().splitlines()
        assert len(log) == 2 and "token" in log[0] and "manual-override" in log[1]

    def test_main_refuses_eval_test_without_token(self, tmp_path):
        import hug_train
        from test_guard import TestAccessDenied
        args = _hug_args(eval_test=True, run_dir=str(tmp_path / "job"))
        with pytest.raises(TestAccessDenied):
            hug_train.run(args)

    def test_hug_module_dataset_agnostic(self):
        import inspect, hug, data_loader, features
        src = inspect.getsource(hug)
        names = set(data_loader.KuaiRandLoader.LOG_USECOLS) | set(features.USER_CAT_COLS) \
            | set(features.VIDEO_CAT_COLS) | set(features.VIDEO_STAT_COLS) | set(features.FEEDBACK_COLS) \
            | {"tab", "music_id", "author_id", "kuairand", "KuaiRand"}
        hits = [n for n in names if f'"{n}"' in src or f"'{n}'" in src or n.lower() in src.lower().split()]
        assert not hits, hits

    @pytest.mark.parametrize("flags", [
        {}, {"no_graph": True}, {"no_seq": True}, {"no_graph": True, "no_seq": True},
        {"freeze_graph": True}, {"cl_weight": 0}, {"graph_tokens": True},
    ])
    def test_arm_flags(self, loaded_data, timed_bundle, flags):
        import hug_train
        args = _hug_args(**flags)
        d = hug_train.prepare(args, data=loaded_data, bundle=timed_bundle)
        model = hug_train.build(args, d)
        assert (model.gcn is None) == bool(flags.get("no_graph"))
        assert (model.seq is None) == bool(flags.get("no_seq"))
        if flags.get("no_graph"):
            assert d.store.rel_master[0].shape[1] == 0           # no snapshot matrices
        snaps = hug_train.Snapshots(d, model, torch.device("cpu"))
        b = hug_train.HugBatcher(d, args.max_seq_len)(d.train_rows[:32])
        vx = snaps.enter(int(d.snap[d.train_rows[0]]))
        model.train()
        tables = model.graph_tables(vx, train=True) if model.gcn is not None else None
        out = model(b, vx, tables, train=True)
        out["loss"].backward()
        assert torch.isfinite(out["loss"])


# ── Baseline parity (spec 04 W5 tests 1, 11–14) ────────────────────────────────

BASELINE_CSV = Path(__file__).resolve().parents[1] / "Baselines" / "data" / "processed" / "kuairand_1k_csv"


def test_fuxictr_vocab_fits_on_train_only(tmp_path):
    """A categorical value seen only in validation maps to OOV."""
    pytest.importorskip("fuxictr")
    from fuxictr.preprocess import FeatureProcessor, build_dataset
    import pickle
    tr = pd.DataFrame({"c": ["a", "b", "a", "b"] * 5, "y": [0, 1] * 10})
    va = pd.DataFrame({"c": ["zzz_val_only", "a"], "y": [0, 1]})
    tr.to_csv(tmp_path / "train.csv", index=False)
    va.to_csv(tmp_path / "valid.csv", index=False)
    params = dict(dataset_id="toy", data_root=str(tmp_path), data_format="csv",
                  train_data=str(tmp_path / "train.csv"), valid_data=str(tmp_path / "valid.csv"),
                  test_data=None, min_categr_count=1,
                  feature_cols=[{"name": "c", "active": True, "dtype": "str", "type": "categorical"}],
                  label_col={"name": "y", "dtype": "float"})
    fe = FeatureProcessor(**params)
    build_dataset(fe, **params)
    vocab = json.loads((tmp_path / "toy" / "feature_vocab.json").read_text())["c"]
    assert "zzz_val_only" not in vocab
    with open(tmp_path / "toy" / "feature_processor.pkl", "rb") as f:
        fe2 = pickle.load(f)
    enc = fe2.transform(pd.DataFrame({"c": ["zzz_val_only"], "y": [0]}))
    assert int(enc["c"].iloc[0]) == vocab["__OOV__"]


class TestBaselineParityReal:
    """HUG's per-row inputs equal the baseline CSV columns (needs --data-dir and preprocess.py output)."""

    @pytest.fixture(scope="class")
    def real_hug(self, real_data_dir):
        if real_data_dir is None:
            pytest.skip("--data-dir not provided")
        if not (BASELINE_CSV / "valid.csv").exists():
            pytest.skip("run Baselines/preprocess.py first")
        import hug_train
        args = _hug_args(data_dir=str(real_data_dir), snapshot_hours=24, max_seq_len=50,
                         min_id_count=5, no_graph=True, emb_dim=64, cache_dir="none")
        args.min_interactions = 10
        return hug_train.prepare(args), args

    @pytest.fixture(scope="class")
    def sample(self, real_hug):
        from features import USER_CAT_COLS, VIDEO_CAT_COLS, VIDEO_STAT_COLS
        va = pd.read_csv(BASELINE_CSV / "valid.csv", dtype={"hist_video_ids": str},
                         keep_default_na=False, low_memory=False)
        return va.sample(10_000, random_state=0).sort_values("row_id").reset_index(drop=True)

    def test_history_parity(self, real_hug, sample):
        import hug_train
        d, args = real_hug
        rows = sample["row_id"].to_numpy()
        b = hug_train.HugBatcher(d, args.max_seq_len)(rows)
        inv = dict(d.inter.frame[["video_id"]].assign(n=d.inter.video)
                   .drop_duplicates().set_index("n")["video_id"].items())     # node idx -> raw id
        for i, h in enumerate(b["hist"].numpy()):
            hug_hist = [str(inv[v]) for v in h if v >= 0]
            csv_hist = sample["hist_video_ids"].iloc[i].split() if sample["hist_video_ids"].iloc[i] else []
            assert hug_hist == csv_hist, rows[i]

    def test_feature_parity(self, real_hug, sample):
        import hug_train
        from features import VIDEO_STAT_COLS, video_categoricals, VIDEO_CAT_COLS
        d, args = real_hug
        rows = sample["row_id"].to_numpy()
        b = hug_train.HugBatcher(d, args.max_seq_len)(rows)
        assert np.array_equal(b["ctx_cat"][:, 0].numpy(), sample["tab"].to_numpy())
        assert np.array_equal(b["ctx_cat"][:, 1].numpy(), sample["hour"].to_numpy())
        np.testing.assert_allclose(b["ctx_num"].numpy(), sample[VIDEO_STAT_COLS].to_numpy(), atol=1e-5)
        assert np.array_equal(d.inter.frame["video_id"].to_numpy()[rows], sample["video_id"].to_numpy())
        assert np.array_equal(d.inter.label[rows].astype(int), sample["is_click"].to_numpy())
        # video categoricals in the CSV are the shared encoding HUG embeds
        vc = d.video_cat_raw.set_index("video_id")
        for c in VIDEO_CAT_COLS:
            assert (vc.loc[sample["video_id"], c].to_numpy() == sample[c].astype(str).to_numpy()).all(), c

    def test_bucket_keys_shared(self, real_hug, sample):
        d, _ = real_hug
        rows = sample["row_id"].to_numpy()
        for name, col in [("video_train_count", "bucket_video_train_count"),
                          ("coldwarm", "bucket_coldwarm"), ("history_len", "bucket_history_len")]:
            assert (d.buckets[name][rows] == sample[col].astype(str).to_numpy()).all(), name

    def test_holdout_excluded_from_baseline_train(self, real_hug):
        d, _ = real_hug
        train_ids = pd.read_csv(BASELINE_CSV / "train.csv", usecols=["row_id"])["row_id"].to_numpy()
        hold_ids  = pd.read_csv(BASELINE_CSV / "holdout.csv", usecols=["row_id"])["row_id"].to_numpy()
        assert np.array_equal(np.sort(hold_ids), d.holdout)
        assert not np.isin(hold_ids, train_ids).any()
        assert len(train_ids) + len(hold_ids) == len(d.inter.rows(SPLIT_TRAIN))

