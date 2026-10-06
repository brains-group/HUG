"""
Spec 06 tests: dataset adapters (MIND, ZhihuRec, KuaiRand) and the causal
protocol under delayed labels.  Synthetic fixtures are written in each dataset's
native file format.  Real-data checks are separate (HUG_REAL_DATA=1).
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tests import _hug_args, tmp_data_dir  # noqa: F401  (fixture)

DAY_S = 86_400
ZHIHU_T0 = 1525276800            # 2018-05-02 16:00 UTC = 2018-05-03 00:00 CST


# ── fixtures in native formats ────────────────────────────────────────────────

def write_mind(root: Path, n_users: int = 40, n_news: int = 80, seed: int = 0) -> Path:
    """MINDlarge-shaped train (11-09..11-14) and dev (11-15) releases."""
    rng = np.random.default_rng(seed)
    news = [f"N{i}" for i in range(1, n_news + 1)]
    cats = [f"c{i % 4}" for i in range(n_news)]
    subs = [f"s{i % 7}" for i in range(n_news)]                 # s0 sits under c0 and c3, …
    ent = {}
    for n in news:
        k = rng.integers(0, 4)
        ent[n] = [{"Label": "x", "Type": "P", "WikidataId": f"Q{int(q)}",
                   "Confidence": float(np.round(rng.uniform(0.3, 1.0), 3)),
                   "OccurrenceOffsets": [0], "SurfaceForms": ["x"]}
                  for q in rng.choice(30, k, replace=False)]
    hist = {u: list(rng.choice(news, rng.integers(0, 6), replace=False)) if u % 4 else []
            for u in range(n_users)}
    rows = {"train": [], "dev": []}
    imp_id = 1
    for u in range(n_users):
        for _ in range(rng.integers(4, 10)):
            day = int(rng.integers(9, 16))
            t = dt.datetime(2019, 11, day, int(rng.integers(0, 24)), int(rng.integers(0, 60)),
                            int(rng.integers(0, 60)))
            cand = rng.choice(news, rng.integers(3, 9), replace=False)
            lab = (rng.random(len(cand)) < 0.25).astype(int)
            lab[rng.integers(len(cand))] = 1                        # ≥ 1 click per impression
            imps = " ".join(f"{c}-{l}" for c, l in zip(cand, lab))
            line = f"{imp_id}\tU{u}\t{t.strftime('%m/%d/%Y %I:%M:%S %p')}\t{' '.join(hist[u])}\t{imps}\n"
            rows["dev" if day == 15 else "train"].append((t, line))
            imp_id += 1
    for split in ("train", "dev"):
        d = root / f"MINDlarge_{split}"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "behaviors.tsv", "w") as f:
            for _, line in sorted(rows[split], key=lambda x: int(x[1].split("\t")[0])):
                f.write(line)
        with open(d / "news.tsv", "w") as f:
            for n, c, s in zip(news, cats, subs):
                f.write("\t".join([n, c, s, "t", "a", "u", json.dumps(ent[n]), "[]"]) + "\n")
        with open(d / "entity_embedding.vec", "w") as f:
            for q in range(0, 30, 2):                               # half the entities covered
                f.write(f"Q{q}\t" + "\t".join(f"{x:.6f}" for x in rng.normal(size=100)) + "\t\n")
    return root


def zhihu_tables(n_users: int = 40, n_answers: int = 60, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        t = ZHIHU_T0 + int(rng.integers(0, DAY_S))
        for _ in range(rng.integers(6, 14)):
            t += int(rng.integers(0, 40_000))
            for a in rng.choice(n_answers, rng.integers(1, 4), replace=False):   # same-ts batches
                c = 0
                if rng.random() < 0.35:
                    c = t + int(rng.choice([0, 5, 30, 300, 1200, 3000, -40]))
                rows.append((u, int(a), t, c))
    imp = pd.DataFrame(rows, columns=["user_id", "answer_id", "impression_ts", "click_ts"])
    imp = imp.sort_values(["user_id", "impression_ts"], kind="stable").reset_index(drop=True)
    anon = rng.random(n_answers) < 0.15
    ans = pd.DataFrame({
        "answer_id": range(n_answers),
        "question_id": np.where(rng.random(n_answers) < 0.05, np.nan, rng.integers(0, 20, n_answers)),
        "anonymous": anon.astype(int),
        "author_id": np.where(anon, np.nan, rng.integers(0, 25, n_answers)),
        "high_value": rng.integers(0, 2, n_answers), "editor_rec": rng.integers(0, 2, n_answers),
        "create_ts": np.where(rng.random(n_answers) < 0.1, 0, ZHIHU_T0 - rng.integers(0, 50 * DAY_S, n_answers)),
        "has_pic": rng.integers(0, 2, n_answers), "has_video": rng.integers(0, 2, n_answers),
        **{c: rng.integers(0, 1000, n_answers) for c in
           ["n_thanks", "n_likes", "n_comments", "n_collections", "n_dislikes", "n_reports", "n_helpless"]},
        "tokens": "1 2 3",
        "topics": [" ".join(map(str, rng.choice(30, rng.integers(0, 4), replace=False))) for _ in range(n_answers)],
    })
    ans.loc[3, "create_ts"] = ZHIHU_T0 + 10 * DAY_S                # created after its impressions
    usr = pd.DataFrame({
        "user_id": range(n_users),
        "register_ts": ZHIHU_T0 - rng.integers(-DAY_S, 900 * DAY_S, n_users),
        "gender": rng.integers(0, 3, n_users), "login_freq": rng.integers(0, 5, n_users),
        **{c: rng.integers(0, 500, n_users) for c in
           ["n_followers", "n_topics_followed", "n_questions_followed", "n_answers", "n_questions",
            "n_comments", "n_thanks_recv", "n_comments_recv", "n_likes_recv", "n_dislikes_recv"]},
        "register_type": rng.integers(0, 6, n_users), "register_platform": rng.integers(0, 4, n_users),
        **{c: rng.integers(0, 2, n_users) for c in
           ["from_android", "from_iphone", "from_ipad", "from_pc", "from_mobile_web"]},
        "device_model": rng.integers(0, 50, n_users), "device_brand": rng.integers(0, 9, n_users),
        "platform": rng.integers(0, 4, n_users), "province": rng.integers(0, 30, n_users),
        "city": rng.integers(0, 90, n_users),
        "topics_followed": [" ".join(map(str, rng.choice(30, 3, replace=False))) for _ in range(n_users)],
    })
    qs = pd.DataFrame({"question_id": range(20), "create_ts": ZHIHU_T0 - 1000,
                       **{c: rng.integers(0, 100, 20) for c in
                          ["n_answers", "n_followers", "n_invitations", "n_comments"]},
                       "tokens": "4 5",
                       "topics": [" ".join(map(str, rng.choice(30, rng.integers(0, 3), replace=False)))
                                  for _ in range(20)]})
    au = pd.DataFrame({"author_id": range(25), "excellent_author": rng.integers(0, 2, 25),
                       "n_followers": rng.integers(0, 1000, 25), "excellent_answerer": rng.integers(0, 2, 25)})
    return {"inter_impression": imp, "info_answer": ans, "info_user": usr, "info_question": qs,
            "info_author": au, "info_topic": pd.DataFrame({"topic_id": range(30)})}


def write_zhihu(root: Path, tables: dict[str, pd.DataFrame]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        with gzip.open(root / f"{name}.csv.gz", "wt") as f:
            df.to_csv(f, header=False, index=False, float_format="%.0f")
    return root


def _args(dataset: str, data_dir: Path, **kw):
    base = {"dataset": dataset, "data_dir": str(data_dir), "user_frac": 1.0}
    base.update(kw)
    return _hug_args(**base)


def _load(dataset, data_dir, **kw):
    from datasets import load_dataset
    return load_dataset(_args(dataset, data_dir, **kw))


@pytest.fixture(scope="module")
def mind_dir(tmp_path_factory):
    return write_mind(tmp_path_factory.mktemp("mind"))


@pytest.fixture(scope="module")
def zhihu_dir(tmp_path_factory):
    return write_zhihu(tmp_path_factory.mktemp("zhihu"), zhihu_tables())


@pytest.fixture(scope="module")
def mind_ds(mind_dir):
    return _load("mind", mind_dir)


@pytest.fixture(scope="module")
def zhihu_ds(zhihu_dir):
    return _load("zhihurec", zhihu_dir)


# ── A2: adapter contract ──────────────────────────────────────────────────────

def _check_contract(ds):
    it = ds.inter
    N = len(it.time)
    df = ds.interactions
    assert list(df.columns[:5]) == ["user", "item", "time", "label", "label_time"]
    assert not df[["user", "item", "time", "label", "label_time"]].isna().any().any()
    assert (np.diff(it.time) >= 0).all()
    assert (ds.label_time >= it.time).all()
    assert it.user.min() >= 0 and it.user.max() < ds.n_users
    assert it.video.min() >= 0 and it.video.max() < ds.n_items
    assert set(np.unique(it.label)) <= {0.0, 1.0}
    assert ds.ctx_cat.shape == (N, len(ds.ctx_cat_names)) and ds.ctx_cat.dtype == np.int64
    assert ds.ctx_num.shape == (N, len(ds.ctx_num_names)) and ds.ctx_num.dtype == np.float32
    assert np.isfinite(ds.ctx_num).all()
    assert len(ds.session) == N
    assert len(ds.user_x) == ds.n_users and len(ds.user_onehot) == ds.n_users
    assert len(ds.item_cat) == ds.n_items and set(ds.item_cat_cols) <= set(ds.item_cat.columns)
    g = ds.graph
    assert g.node_types[:2] == ["user", "video"]
    assert g.counts["user"] == ds.n_users and g.counts["video"] == ds.n_items
    ei, et, tt = g.master()
    total = sum(g.counts.values())
    assert ei.shape[1] == len(et) == len(tt)
    if ei.shape[1]:
        assert ei.min() >= 0 and ei.max() < total
        off = g.offsets
        for r, (st, dt_) in enumerate(g.relations):
            m = et == r
            if m.any():   # every edge of relation r connects nodes of its declared types
                assert ei[0, m].min() >= off[st] and ei[0, m].max() < off[st] + g.counts[st]
                assert ei[1, m].min() >= off[dt_] and ei[1, m].max() < off[dt_] + g.counts[dt_]
    for li, ln in ds.metadata_links:
        assert len(li) == len(ln) and (len(li) == 0 or li.max() < ds.n_items)
    assert ds.item_features.at(int(it.time.max()) + 1).shape[0] == ds.n_items
    assert ds.fingerprint["t_val"] == it.t_val


def test_contract_mind(mind_ds):
    _check_contract(mind_ds)
    assert mind_ds.metadata_evidence_types == ["subcategory", "entity"]


def test_contract_zhihurec(zhihu_ds):
    _check_contract(zhihu_ds)
    assert zhihu_ds.metadata_evidence_types == ["author", "question"]


def test_contract_kuairand(tmp_data_dir):
    from datasets import load_dataset
    args = _hug_args(data_dir=str(tmp_data_dir))
    args.min_interactions = 1
    ds = load_dataset(args)
    _check_contract(ds)
    assert (ds.label_time == ds.inter.time).all()


def test_mind_entities_and_transe(mind_dir):
    ds = _load("mind", mind_dir, entity_init="transe")
    assert ds.stats["entity_pairs_dropped_low_confidence"] > 0          # fixture has conf < 0.5
    x = ds.meta_features["entity"]
    assert x.shape == (ds.graph.counts["entity"], 100)
    assert 0 < (np.abs(x).sum(1) > 0).sum() < len(x)                      # missing rows fall back to 0


# ── A3: label timing ──────────────────────────────────────────────────────────

def test_label_time_definitions(zhihu_ds, zhihu_dir):
    imp = pd.read_csv(zhihu_dir / "inter_impression.csv.gz", header=None,
                      names=["u", "a", "t", "c"])
    it = zhihu_ds.inter
    # recover each canonical row's raw (t, c): canonical order is a stable sort by t
    raw = imp.iloc[np.argsort(imp.t.to_numpy(), kind="stable")].reset_index(drop=True)
    clicked = raw.c.to_numpy() > 0
    want = np.where(clicked, np.maximum(raw.c, raw.t), raw.t + 900) * 1000
    np.testing.assert_array_equal(zhihu_ds.label_time, want)
    np.testing.assert_array_equal(it.label, clicked.astype(np.float32))
    assert (clicked & (raw.c < raw.t)).any()                  # negative delays exercised (clamped)


def test_label_derived_inputs_use_label_time(zhihu_ds):
    """History, click stats, click/skip edges: only labels known strictly before t."""
    from features import click_history
    ds = zhihu_ds
    it = ds.inter
    u, t, lt, y, item = it.user, it.time, ds.label_time, it.label, it.video
    seq, hs, he = click_history(u, t, np.arange(len(t)), y, 1000, label_time=lt)
    leaked = 0
    for i in range(len(t)):
        known = np.flatnonzero((u == u[i]) & (y == 1) & (lt < t[i]))
        assert set(seq[hs[i]:he[i]]) == set(known)
        naive = np.flatnonzero((u == u[i]) & (y == 1) & (t < t[i]))
        leaked += len(set(naive) - set(known))
        # click stats: known clicks / labelled rows; exposure count sees every earlier impression
        same = item == item[i]
        shows = (same & (t < t[i])).sum()
        clicks = (same & (y == 1) & (lt < t[i])).sum()
        labelled = (same & (lt < t[i])).sum()
        np.testing.assert_allclose(ds.ctx_num[i, :3],
                                   [np.log1p(shows), np.log1p(clicks), clicks / labelled if labelled else 0],
                                   rtol=1e-6, atol=1e-6)
    assert leaked > 0          # the fixture has clicks impressed before t but known after t

    # graph: a clicked/skipped edge is visible in a snapshot only once its label is known
    ei, et, tt = ds.graph.master()
    names = ds.graph.relation_names
    off = ds.graph.offsets
    b = int(np.quantile(t, 0.6))
    for rel, pos in (("clicked", 1.0), ("skipped", 0.0)):
        r = names.index(rel)
        vis = (et == r) & (tt < b)
        got = sorted(zip((ei[0, vis] - off["user"]).tolist(), (ei[1, vis] - off["video"]).tolist()))
        rows = (y == pos) & (lt < b)
        assert got == sorted(zip(u[rows].tolist(), item[rows].tolist()))


def test_same_timestamp_rows_never_see_each_other(zhihu_ds):
    from features import asof_group_count
    it = zhihu_ds.inter
    key = it.user * (it.time.max() + 1) + it.time
    _, cnt = np.unique(key, return_counts=True)
    assert (cnt > 1).any()                                    # batches exist in the fixture
    shows = asof_group_count(it.video, it.time)
    for i in np.flatnonzero(np.isin(key, np.unique(key)[cnt > 1]))[:200]:
        assert shows[i] == ((it.video == it.video[i]) & (it.time < it.time[i])).sum()


# ── A4: prediction invariance with delayed labels ─────────────────────────────

def _hug_scores(data_dir, model_state=None):
    import hug_train
    from runtime import set_determinism
    args = _args("zhihurec", data_dir, snapshot_hours=6, min_id_count=1)
    d = hug_train.prepare(args)
    set_determinism(0)
    m = hug_train.build(args, d)
    if model_state is not None:
        m.load_state_dict(model_state)
    snaps = hug_train.Snapshots(d, m, torch.device("cpu"))
    r, s = hug_train.predict(m, d, d.val_rows, snaps, hug_train.HugBatcher(d, args.max_seq_len),
                             64, torch.device("cpu"), True, "x")
    return pd.Series(s, index=r).sort_index(), m.state_dict(), d


def test_predictions_invariant_to_delayed_labels(tmp_path):
    tabs = zhihu_tables(seed=3)
    imp = tabs["inter_impression"]
    t = imp.impression_ts.to_numpy()
    T = int(np.sort(t)[int(0.75 * len(t))]) + 300            # inside the val window, 300 s after a row
    # make sure some impressions before T have their click land at/after T
    near = np.flatnonzero((t < T) & (t > T - 600))[:6]
    assert len(near) > 0
    imp.loc[near, "click_ts"] = T + 50
    base = write_zhihu(tmp_path / "base", tabs)
    a, state, d = _hug_scores(base)
    assert d.inter.t_val < T * 1000 < d.inter.t_test

    # (1) flip the labels of impressions whose label is known only at/after T
    t1 = {k: v.copy() for k, v in tabs.items()}
    t1["inter_impression"].loc[near, "click_ts"] = 0          # now a non-click: known at t + 900 ≥ T
    assert ((t[near] + 900) >= T).all()
    b, _, _ = _hug_scores(write_zhihu(tmp_path / "flip", t1), state)
    early = (np.asarray(a.index) >= 0) & (d.inter.time[a.index.values] < T * 1000)
    assert early.any()
    np.testing.assert_array_equal(a.values[early], b.values[early])

    # (2) rewrite every row at or after T (labels, click times and answers)
    t2 = {k: v.copy() for k, v in tabs.items()}
    late = t2["inter_impression"].impression_ts.to_numpy() >= T
    rng = np.random.default_rng(9)
    im2 = t2["inter_impression"]
    im2.loc[late, "click_ts"] = np.where(im2.loc[late, "click_ts"] > 0, 0, im2.loc[late, "impression_ts"] + 7)
    im2.loc[late, "answer_id"] = rng.permutation(im2.loc[late, "answer_id"].to_numpy())
    c, _, _ = _hug_scores(write_zhihu(tmp_path / "late", t2), state)
    np.testing.assert_array_equal(a.values[early], c.values[early])
    # … and the rewritten rows themselves do change
    assert not np.array_equal(a.values[~early], c.values[~early])


# ── A5: HUG/baseline parity ───────────────────────────────────────────────────

def _baseline_frame(ds):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Baselines"))
    import preprocess
    return preprocess.generic_baseline_frame(ds), preprocess.generic_columns(ds)


@pytest.mark.parametrize("name", ["mind", "zhihurec"])
def test_hug_baseline_parity(name, mind_dir, zhihu_dir):
    import hug_train
    data_dir = mind_dir if name == "mind" else zhihu_dir
    args = _args(name, data_dir, snapshot_hours=24, max_seq_len=50)
    from datasets import load_dataset
    ds = load_dataset(args)
    d = hug_train.prepare(args, ds=ds)
    bf, cols = _baseline_frame(ds)
    assert len(bf) == len(d.inter.time)
    np.testing.assert_array_equal(bf["is_click"].to_numpy(), d.inter.label.astype(int))
    np.testing.assert_array_equal(bf["split"].to_numpy(), d.inter.split)
    for i in range(len(bf)):
        toks = d.hist_seq[d.hist_start[i]:d.hist_end[i]]
        want = " ".join(map(str, d.tok_item[toks]))
        assert bf["hist_video_ids"].iat[i] == want
    for j, n in enumerate(ds.ctx_cat_names):
        np.testing.assert_array_equal(bf[n].to_numpy(), d.ctx_cat[:, j])
    for j, n in enumerate(ds.ctx_num_names):
        np.testing.assert_array_equal(bf[n].to_numpy(np.float32), d.ctx_num[:, j])
    for k in ("video_train_count", "coldwarm", "history_len"):
        np.testing.assert_array_equal(bf[f"bucket_{k}"].to_numpy(), d.buckets[k])


# ── A6: day_snap ──────────────────────────────────────────────────────────────

def test_day_snap_on_midnights_and_mind_days(mind_ds):
    info = mind_ds.split_info
    assert info["rule_used"] == "day_snap"
    for c in info["cutoffs"]:
        assert c % (24 * 3_600_000) == 0                       # tz offset 0: UTC midnight
    days = [dt.datetime.utcfromtimestamp(c / 1000).strftime("%m-%d") for c in info["cutoffs"]]
    assert days == ["11-14", "11-15"]
    it = mind_ds.inter
    for split, (lo, hi) in {0: ("11-09", "11-13"), 1: ("11-14", "11-14"), 2: ("11-15", "11-15")}.items():
        d = pd.to_datetime(it.time[it.split == split], unit="ms").strftime("%m-%d")
        assert d.min() == lo and d.max() == hi


def test_day_snap_local_midnight_and_fallback():
    from datasets.base import nearest_midnight, split_cutoffs
    H = 3_600_000
    # Beijing (UTC+8): 2018-05-11 21:56 CST snaps to 05-12 00:00 CST = 05-11 16:00 UTC
    t = int(pd.Timestamp("2018-05-11 13:56:00", tz="UTC").value // 10**6)
    m = nearest_midnight(t, 8)
    assert (m + 8 * H) % (24 * H) == 0 and m == int(pd.Timestamp("2018-05-11 16:00", tz="UTC").value // 10**6)
    # Zhihu-shaped: both quantiles snap to the same midnight → guard falls back to the exact cut
    base = int(pd.Timestamp("2018-05-03", tz="UTC").value // 10**6)
    times = np.sort(np.r_[base + np.arange(700) * 60_000 * 13,           # long train stretch
                          base + 8 * 24 * H + np.arange(300) * 60_000 * 2])   # dense last day
    t_val, t_test, info = split_cutoffs(times, "day_snap", 8)
    assert info["rule_used"] == "quantile_fallback"
    assert [t_val, t_test] == info["quantile_cutoffs"]
    _, _, ok = split_cutoffs(np.arange(0, 10 * 24 * H, H), "day_snap", 0)
    assert ok["rule_used"] == "day_snap"


def test_zhihu_fallback_recorded(zhihu_ds):
    assert zhihu_ds.split_info["rule_used"] in ("day_snap", "quantile_fallback")
    assert zhihu_ds.fingerprint["split_rule_used"] == zhihu_ds.split_info["rule_used"]


# ── A7: subsampling ───────────────────────────────────────────────────────────

def test_user_subsample(mind_dir, zhihu_dir):
    from datasets.base import sample_users
    u = np.repeat(np.arange(100), 3)
    assert np.array_equal(sample_users(u, 0.3, 0), sample_users(u, 0.3, 0))
    assert len(sample_users(u, 0.3, 0)) == 30 and not np.array_equal(sample_users(u, 0.3, 0), sample_users(u, 0.3, 1))

    imp = pd.read_csv(zhihu_dir / "inter_impression.csv.gz", header=None, names=["u", "a", "t", "c"])
    full = imp.groupby("u").size()
    ds = _load("zhihurec", zhihu_dir, user_frac=0.5)
    ds2 = _load("zhihurec", zhihu_dir, user_frac=0.5)
    assert ds.fingerprint == ds2.fingerprint
    kept = sample_users(imp.u.to_numpy(), 0.5, 0)
    assert ds.n_users == len(kept)
    per_user = np.bincount(ds.inter.user, minlength=ds.n_users)
    np.testing.assert_array_equal(per_user, full.loc[kept].to_numpy())     # users fully in
    # statistics are computed within the sample
    it = ds.inter
    for i in range(0, len(it.time), 7):
        shows = ((it.video == it.video[i]) & (it.time < it.time[i])).sum()
        assert ds.ctx_num[i, 0] == pytest.approx(np.log1p(shows))

    m = _load("mind", mind_dir, user_frac=0.5)
    beh = pd.concat([pd.read_csv(mind_dir / f"MINDlarge_{s}" / "behaviors.tsv", sep="\t", header=None)
                     for s in ("train", "dev")])
    assert m.n_users == len(sample_users(beh[1].to_numpy(), 0.5, 0))


# ── A8: pre-window history (MIND) ─────────────────────────────────────────────

def test_pre_window_history(mind_dir):
    import hug_train
    from hug_train import PRE_WINDOW_GAP
    args = _args("mind", mind_dir, snapshot_hours=24, max_seq_len=50, min_id_count=1)
    from datasets import load_dataset
    ds = load_dataset(args)
    d = hug_train.prepare(args, ds=ds)
    it = d.inter
    pw_user, pw_item = ds.pre_window
    assert len(pw_user) > 0
    b = hug_train.HugBatcher(d, args.max_seq_len)(np.arange(len(it.time)))
    for i in range(len(it.time)):
        pre = pw_item[pw_user == it.user[i]]
        toks = d.hist_seq[d.hist_start[i]:d.hist_end[i]]
        n_pre = int(d.tok_pre[toks].sum())
        assert n_pre == len(pre)                                  # no truncation in the fixture
        assert d.tok_pre[toks[:n_pre]].all() and not d.tok_pre[toks[n_pre:]].any()   # first
        np.testing.assert_array_equal(d.tok_item[toks[:n_pre]], pre)
        # in-window part: clicks with label_time < t, in order
        known = np.flatnonzero((it.user == it.user[i]) & (it.label == 1) & (ds.label_time < it.time[i]))
        np.testing.assert_array_equal(np.sort(toks[n_pre:]), known)
        # batcher: pre-window tokens carry the PRE_WINDOW gap bucket and are never same-session
        L = len(toks)
        if n_pre:
            gap = b["hist_gap"][i, -L:][:n_pre]
            assert (gap == PRE_WINDOW_GAP).all()
            assert (b["hist_sess"][i, -L:][:n_pre] == 0).all()
    # pre-window clicked edges exist from snapshot 0 on
    ei, et = d.store.rel(0)
    r = ds.graph.relation_names.index("clicked")
    off = ds.graph.offsets
    vis = set(zip((ei[0, et == r] - off["user"]).tolist(), (ei[1, et == r] - off["video"]).tolist()))
    assert set(zip(pw_user.tolist(), pw_item.tolist())) <= vis


# ── A9: no snapshot features ──────────────────────────────────────────────────

SNAPSHOT_NAMES = {"n_followers", "n_topics_followed", "n_questions_followed", "n_answers", "n_questions",
                  "n_comments", "n_thanks_recv", "n_comments_recv", "n_likes_recv", "n_dislikes_recv",
                  "n_thanks", "n_likes", "n_collections", "n_dislikes", "n_reports", "n_helpless",
                  "n_invitations", "topics_followed", "questions_followed"}


def test_no_snapshot_features_by_default(zhihu_ds, zhihu_dir):
    def leaks(names):
        return [n for n in names if any(s in n for s in SNAPSHOT_NAMES)]
    assert not leaks(zhihu_ds.ctx_num_names)
    bt = zhihu_ds.baseline_tables()
    assert not leaks(list(bt.user_cats.columns) + list(bt.item_cats.columns))
    _, cols = _baseline_frame(zhihu_ds)
    assert not leaks(cols)
    assert not leaks(list(zhihu_ds.item_cat.columns))
    with_snap = _load("zhihurec", zhihu_dir, snapshot_features=True)
    assert leaks(with_snap.ctx_num_names)
    assert with_snap.fingerprint != zhihu_ds.fingerprint


# ── A10: dataset isolation ────────────────────────────────────────────────────

def test_generic_modules_dataset_agnostic():
    """No dataset column names in the generic modules (spec 04 test 19, extended)."""
    import inspect
    import features
    import fusion
    import hug
    import hkg_constructor
    import temporal
    from datasets import kuairand, mind, zhihurec
    names = set(kuairand.COLUMN_NAMES) | set(mind.COLUMN_NAMES) | set(zhihurec.COLUMN_NAMES)
    for mod in (hug, fusion, temporal, features, hkg_constructor):
        src = inspect.getsource(mod)
        words = set(src.lower().replace("(", " ").replace(")", " ").replace(",", " ")
                    .replace("[", " ").replace("]", " ").replace(":", " ").split())
        # literals for every name; bare words only for identifier-like names ("author_id"),
        # since plain words such as "impressions" also appear in prose
        hits = [n for n in names if f'"{n}"' in src or f"'{n}'" in src
                or (("_" in n or n != n.lower()) and n.lower() in words)]
        # dataset names may appear in prose, never as string literals (no per-dataset branches)
        hits += [n for n in ("kuairand", "mind", "zhihurec") if f'"{n}"' in src or f"'{n}'" in src]
        assert not hits, (mod.__name__, hits)


# ── A11 / real data (HUG_REAL_DATA=1) ─────────────────────────────────────────

real = pytest.mark.skipif(os.environ.get("HUG_REAL_DATA") != "1", reason="set HUG_REAL_DATA=1")


@real
@pytest.mark.parametrize("name", ["mind", "zhihurec"])
def test_real_parity_5000_rows(name):
    import hug_train
    from datasets import load_dataset
    args = _hug_args(dataset=name, data_dir=None,
                     cache_dir=str(Path(__file__).resolve().parents[1] / "cache" / "1k"),
                     snapshot_hours=24, max_seq_len=50)
    args.data_dir = str(Path(__file__).resolve().parents[1] / {"mind": "MIND/extracted",
                                                               "zhihurec": "ZhihuRec/raw"}[name])
    args.user_frac = {"mind": 0.2, "zhihurec": 0.125}[name]
    ds = load_dataset(args)
    d = hug_train.prepare(args, ds=ds)
    bf, _ = _baseline_frame(ds)
    rows = np.random.default_rng(0).choice(len(bf), 5000, replace=False)
    for i in rows:
        toks = d.hist_seq[d.hist_start[i]:d.hist_end[i]]
        assert bf["hist_video_ids"].iat[i] == " ".join(map(str, d.tok_item[toks]))
    for j, n in enumerate(ds.ctx_num_names):
        np.testing.assert_array_equal(bf[n].to_numpy(np.float32)[rows], d.ctx_num[rows, j])
