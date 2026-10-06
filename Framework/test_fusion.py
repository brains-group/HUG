"""Spec 05 tests: fusion heads (Framework/fusion.py) and their integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from tests import (  # noqa: F401  (fixtures)
    _hug_args, _rewrite_after, hug_data, loaded_data, timed_bundle, tmp_data_dir, cutoffs,
)
from hkg_constructor import HKGConstructor

N4_PRE_SPEC05_FINGERPRINT = "60d53e57ca7db1c58674a9bb02cc3d4474fb25f1d3edc0c57ff32a934c33c4ea"
# KuaiRand data pipeline before the spec 06 adapter refactor (regression_fingerprint.py)
KUAIRAND_PRE_SPEC06_DATA_FINGERPRINT = "f54615d07436e3efcfc6c7bd04e69d59063592fb0c7ee9c471185d64532c35b1"


def _sparse(**kw):
    from fusion import SparseSPFusion
    torch.manual_seed(0)
    return SparseSPFusion(6, 4, 3, 8, 0.0, m_shared=12, m_private=10, k_shared=3,
                          k_private_min=2, k_private_max=8, **kw)


def _views(b=32):
    g = torch.Generator().manual_seed(1)
    return (torch.randn(b, 6, generator=g), torch.randn(b, 4, generator=g),
            torch.randn(b, 3, generator=g), torch.rand(b, 2, generator=g) * 5)


# 1 ─────────────────────────────────────────────────────────────────────────────
def test_topk_exact_and_monotone():
    f = _sparse()
    f.set_n_ref([4.0, 4.0])
    e_G, e_S, ctx, ev = _views()
    p = dict(zip(("G", "S"), f.project(e_G, e_S)))
    sh, pr = f.codes(p, ev)
    kp = f.k_private(ev)
    for i, v in enumerate(("G", "S")):
        assert ((sh[v] != 0).sum(-1) <= 3).all()
        assert ((pr[v] != 0).sum(-1) <= kp[:, i]).all()
        # exactly k when the ReLU leaves enough positive entries
        a = torch.relu(f.enc_pr[v](p[v]))
        ok = (a > 0).sum(-1) >= kp[:, i]
        assert ((pr[v] != 0).sum(-1)[ok] == kp[ok, i]).all()
    assert (kp >= 2).all() and (kp <= 8).all()
    grid = torch.linspace(0, 10, 50).unsqueeze(1).repeat(1, 2)
    k = f.k_private(grid)[:, 0]
    assert (k[1:] >= k[:-1]).all()
    fixed = _sparse(k_schedule="fixed", k_private_fixed=5)
    fixed.set_n_ref([4.0, 4.0])
    assert (fixed.k_private(ev) == 5).all()


# 2 ─────────────────────────────────────────────────────────────────────────────
def test_evidence_causal_with_ties():
    from features import click_history, view_evidence
    rng = np.random.default_rng(0)
    n = 300
    fr = pd.DataFrame({"video_id": rng.integers(0, 8, n), "time_ms": rng.integers(0, 30, n)})
    # videos 6 and 7 have no author: they must get metadata_evidence = 0, not a pooled count
    meta = pd.DataFrame({"video_id": range(8), "author_id": [0, 0, 1, 1, 2, 2, np.nan, np.nan]})
    user = rng.integers(0, 4, n)
    click = rng.integers(0, 2, n)
    has = meta.author_id.notna()
    links = [(meta.video_id[has].to_numpy(), meta.author_id[has].to_numpy().astype(int))]
    for max_len in (1_000, 3):                      # untruncated and truncated histories
        _, hs, he = click_history(user, fr.time_ms.to_numpy(), np.arange(n), click, max_len)
        ev = view_evidence(fr.video_id.to_numpy(), fr.time_ms.to_numpy(), he - hs, links)
        auth = fr.video_id.map(dict(zip(meta.video_id, meta.author_id))).to_numpy()
        tt = fr.time_ms.to_numpy()
        for i in range(n):
            t = tt[i]
            item = int(((fr.video_id == fr.video_id[i]) & (tt < t)).sum())
            a = 0 if np.isnan(auth[i]) else int(((auth == auth[i]) & (tt < t)).sum())
            uc = min(int(((user == user[i]) & (click == 1) & (tt < t)).sum()), max_len)
            assert ev[i, 0] == pytest.approx(np.log1p(item) + np.log1p(a), abs=1e-5)
            assert ev[i, 1] == pytest.approx(np.log1p(uc) + np.log1p(item), abs=1e-5)


def test_missing_metadata_neighbour_contributes_zero():
    from features import metadata_evidence
    item, t = np.array([0, 1, 2, 3, 0]), np.array([0, 1, 2, 3, 4])
    # only item 0 has an author; items 1-3 have none and must not be pooled
    me = metadata_evidence(item, t, [(np.array([0]), np.array([5]))])
    np.testing.assert_allclose(me, np.log1p([0, 0, 0, 0, 1]))


def test_metadata_evidence_multiple_neighbours():
    """MIND-style: an item links several entities; the row takes the max as-of count."""
    from features import metadata_evidence
    item = np.array([0, 1, 2, 0, 2, 1])
    t = np.array([0, 1, 1, 2, 3, 3])
    links = [(np.array([0, 0, 1, 2]), np.array([7, 8, 8, 9]))]   # 0→{7,8}, 1→{8}, 2→{9}
    me = metadata_evidence(item, t, links)
    # brute force: per row, max over its neighbours of earlier rows linking that neighbour
    nbs = {0: [7, 8], 1: [8], 2: [9]}
    want = [max(sum(1 for j in range(6) if t[j] < t[i] and nb in nbs[item[j]]) for nb in nbs[item[i]])
            for i in range(6)]
    np.testing.assert_allclose(me, np.log1p(want))


def test_sequence_evidence_uses_encoder_history(hug_data):
    """n_S's history term is log1p of the encoder's history length (hist_end − hist_start)."""
    from features import asof_group_count
    d = hug_data
    item_prior = np.log1p(asof_group_count(d.inter.video, d.inter.time))
    np.testing.assert_allclose(d.evidence[:, 1],
                               np.log1p(d.hist_end - d.hist_start) + item_prior, atol=1e-5)


def test_n_ref_from_training_rows_only(loaded_data, timed_bundle):
    import hug_train
    args = _hug_args(fusion="sparse")
    d = hug_train.prepare(args, data=loaded_data, bundle=timed_bundle)
    m1 = hug_train.build(args, d)
    data2 = _rewrite_after(loaded_data, d.inter.t_val)
    d2 = hug_train.prepare(args, data=data2, bundle=HKGConstructor(data2).build())
    m2 = hug_train.build(args, d2)
    assert torch.equal(m1.head.n_ref, m2.head.n_ref)
    assert np.array_equal(d.evidence[d.train_rows], d2.evidence[d2.train_rows])


@pytest.mark.parametrize("fusion", ["sparse", "evgate"])
def test_training_without_n_ref_raises(fusion):
    from fusion import build_fusion
    args = _hug_args(fusion=fusion)
    head = build_fusion(fusion, 6, 4, 3, args)
    e_G, e_S, ctx, ev = _views()
    head.train()
    with pytest.raises(RuntimeError, match="n_ref"):
        head.fuse(e_G, e_S, ctx, ev)
    head.set_n_ref([2.0, 3.0])
    head.fuse(e_G, e_S, ctx, ev)                 # fine once set


def test_evgate_uses_normalised_evidence():
    from fusion import build_fusion
    head = build_fusion("evgate", 6, 4, 3, _hug_args(fusion="evgate"))
    head.set_n_ref([2.0, 3.0])
    head.eval()
    e_G, e_S, ctx, ev = _views()
    # evidence at or above n_ref saturates at s_v = 1, so its scale no longer matters
    a = head.fuse(e_G, e_S, ctx, torch.full_like(ev, 10.0))["logit"]
    b = head.fuse(e_G, e_S, ctx, torch.full_like(ev, 1e6))["logit"]
    assert torch.equal(a, b)
    c = head.fuse(e_G, e_S, ctx, torch.zeros_like(ev))["logit"]
    assert not torch.equal(a, c)


@pytest.mark.parametrize("fusion", ["sparse", "evgate"])
def test_n_ref_round_trips_checkpoint_and_resume(hug_data, tmp_path, fusion):
    import hug_train
    args = _hug_args(fusion=fusion, max_epochs=1, patience=10)
    model = hug_train.build(args, hug_data)
    model.head.set_n_ref([1.25, 2.5])            # distinctive, not the data's value
    hug_train.train(args, hug_data, model, torch.device("cpu"), tmp_path)
    for ck in ("best.pt", "last.pt"):
        sd = torch.load(tmp_path / ck, weights_only=False)
        sd = sd["model"] if "model" in sd else sd
        fresh = hug_train.build(args, hug_data)
        fresh.load_state_dict(sd)
        assert torch.equal(fresh.head.n_ref, torch.tensor([1.25, 2.5]))
        assert bool(fresh.head.n_ref_set)
    # --resume: a rebuilt model (n_ref from the data) picks the stored n_ref back up
    args2 = _hug_args(fusion=fusion, max_epochs=2, patience=10, resume=True)
    m2 = hug_train.build(args2, hug_data)
    assert not torch.equal(m2.head.n_ref, torch.tensor([1.25, 2.5]))
    state = hug_train.train(args2, hug_data, m2, torch.device("cpu"), tmp_path)
    assert state["epoch"] == 2
    assert torch.equal(m2.head.n_ref, torch.tensor([1.25, 2.5]))


# 3 ─────────────────────────────────────────────────────────────────────────────
def test_atoms_unit_norm_after_step():
    f = _sparse()
    f.set_n_ref([3.0, 3.0])
    opt = torch.optim.Adam(f.parameters(), lr=0.1)
    out = f.fuse(*_views())
    (out["logit"].sum() + out["aux_loss"]).backward()
    opt.step()
    f.renormalize()
    for D in (f.D_sh, f.D_pr["G"], f.D_pr["S"]):
        assert torch.allclose(D.norm(dim=-1), torch.ones(len(D)), atol=1e-6)


# 4 ─────────────────────────────────────────────────────────────────────────────
def _grads(f, **w):
    for k in ("w_rec", "w_align", "w_dec"):
        setattr(f, k, w.get(k, 0.0))
    f.zero_grad()
    e_G, e_S, ctx, ev = _views()
    out = f.fuse(e_G, e_S, ctx, ev)
    out["aux_loss"].backward()
    return {n: (p.grad is not None and p.grad.abs().sum() > 0) for n, p in f.named_parameters()}


def test_gradient_routing():
    f = _sparse()
    f.set_n_ref([3.0, 3.0])
    g = _grads(f, w_rec=1.0)
    assert not any(v for n, v in g.items() if n.startswith("proj_"))      # detached target
    assert g["D_sh"] and g["D_pr.G"] and g["enc_sh.weight"] and g["enc_pr.G.weight"]
    g = _grads(f, w_align=1.0)
    assert not g["D_pr.G"] and not g["D_pr.S"]
    assert g["proj_G.0.weight"] and g["enc_sh.weight"]
    assert not g["enc_pr.G.weight"] and not g["enc_pr.S.weight"]     # alignment: shared only
    g = _grads(f, w_dec=1.0)
    assert not g["enc_sh.weight"] and g["enc_pr.G.weight"] and g["enc_pr.S.weight"]   # private only


# 5 ─────────────────────────────────────────────────────────────────────────────
def test_losses_behave():
    from fusion import cross_cov_penalty, info_nce
    g = torch.Generator().manual_seed(0)
    z = torch.randn(256, 16, generator=g)
    aligned = info_nce(z, z + 0.01 * torch.randn(256, 16, generator=g), 0.2)
    shuffled = info_nce(z, z[torch.randperm(256, generator=g)], 0.2)
    assert aligned < shuffled
    a, b = torch.randn(4096, 8, generator=g), torch.randn(4096, 8, generator=g)
    assert cross_cov_penalty(a, b) < 1e-3
    assert cross_cov_penalty(a, a + 0.1 * b) > 0.05


# 6 ─────────────────────────────────────────────────────────────────────────────
def test_dead_atom_resampling():
    f = _sparse()
    f.set_n_ref([3.0, 3.0])
    with torch.no_grad():                   # private-G atom 0 can never fire …
        f.enc_pr["G"].weight[0] = 0.0
        f.enc_pr["G"].bias[0] = -100.0
    f.fuse(*_views())
    opt = torch.optim.Adam(f.parameters(), lr=1e-3)
    out = f.fuse(*_views())
    (out["logit"].sum() + out["aux_loss"]).backward()
    opt.step()                                  # populate Adam moments
    f.fuse(*_views())
    f.idle_G[0] = 99                        # … and has been idle long enough to be dead
    live_before = f.D_pr["G"].data[1:].clone()
    old0 = f.D_pr["G"].data[0].clone()
    st_D, st_W = opt.state[f.D_pr["G"]], opt.state[f.enc_pr["G"].weight]
    live_m = st_D["exp_avg"][1:].clone()
    f.track_and_resample(step=100, every=100, dead_after=100, optimizer=opt)
    assert not torch.equal(f.D_pr["G"].data[0], old0)
    assert torch.equal(f.D_pr["G"].data[1:], live_before)
    assert f.idle_G[0] == 0
    # optimiser moments zeroed for the reset dictionary and encoder rows only
    for key in ("exp_avg", "exp_avg_sq"):
        assert (st_D[key][0] == 0).all()
        assert (st_W[key][0] == 0).all()
        assert opt.state[f.enc_pr["G"].bias][key][0] == 0
    assert torch.equal(st_D["exp_avg"][1:], live_m)
    assert st_W["exp_avg_sq"][1:].abs().sum() > 0


# ── spec 05b ──────────────────────────────────────────────────────────────────
def test_05b_loss_scales_at_init():
    """L_rec ≈ 2 (two views, per-dimension mean) and L_align ≈ 1 (chance) at init; aux/bce < 2."""
    from fusion import SparseSPFusion
    torch.manual_seed(0)
    f = SparseSPFusion(48, 32, 8, 64, 0.0)                    # defaults: w 0.1 / 0.3 / 0.1
    f.set_n_ref([3.0, 3.0])
    g = torch.Generator().manual_seed(1)
    B = 4096
    e_G, e_S = torch.randn(B, 48, generator=g), torch.randn(B, 32, generator=g)
    ctx, ev = torch.randn(B, 8, generator=g), torch.rand(B, 2, generator=g) * 5
    p = dict(zip(("G", "S"), f.project(e_G, e_S)))
    sh, pr = f.codes(p, ev)
    rec = sum((p[v] - (sh[v] @ f.D_sh + pr[v] @ f.D_pr[v])).pow(2).mean(-1).mean() for v in p)
    # spec says ≈ 2 (p_v alone); random unit atoms add ≈ 0.8/dim/view at init (measured 3.55)
    assert 1.5 < float(rec) < 4.5
    from fusion import align_loss
    assert 0.9 < float(align_loss(sh["G"], sh["S"], 0.2)) < 1.1
    out = f.fuse(e_G, e_S, ctx, ev)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        out["logit"], torch.randint(0, 2, (B,), generator=g).float())
    assert float(out["aux_loss"]) / float(bce) < 2.0


def test_05b_align_uses_1024_rows():
    from fusion import ALIGN_ROWS, align_loss
    torch.manual_seed(0)
    z = torch.randn(5000, 16)
    # perfectly aligned views: loss well below chance; computed on a 1024-row subset
    assert float(align_loss(z, z.clone(), 0.05)) < 0.2
    assert ALIGN_ROWS == 1024


def test_05b_one_shared_encoder():
    f = _sparse()
    f.set_n_ref([3.0, 3.0])
    assert not hasattr(f, "enc")
    assert f.enc_sh.weight.shape == (f.m_sh, 8)
    e_G, e_S, ctx, ev = _views()
    for view in ("G", "S"):               # a gradient from either view reaches E_sh
        f.zero_grad()
        p = dict(zip(("G", "S"), f.project(e_G, e_S)))
        sh, _ = f.codes(p, ev)
        sh[view].sum().backward()
        assert f.enc_sh.weight.grad.abs().sum() > 0
        assert f.enc_pr["G"].weight.grad is None and f.enc_pr["S"].weight.grad is None


def test_05b_dead_window_in_rows(hug_data):
    """dead_after = one epoch's rows in steps; resampled at the next 100-step check only then."""
    import math
    import hug_train
    args = _hug_args(fusion="sparse", max_epochs=1)
    m = hug_train.build(args, hug_data)
    steps = math.ceil(len(hug_data.train_rows) / args.batch_size)
    f = _sparse()
    f.set_n_ref([3.0, 3.0])
    f.dead_after = 250                      # e.g. 2M rows / 8192
    f.fuse(*_views())
    with torch.no_grad():
        f.enc_pr["G"].weight[[0, 1]] = 0.0
        f.enc_pr["G"].bias[[0, 1]] = -100.0
    f.fuse(*_views())
    f.idle_G[0], f.idle_G[1] = 249, 200      # atom 0 reaches the window at step 300, atom 1 not
    old = f.D_pr["G"].data[:2].clone()
    f.track_and_resample(step=299)           # not a check step
    assert torch.equal(f.D_pr["G"].data[:2], old)
    f.track_and_resample(step=300)           # check: atom 0 idle 251 ≥ 250, atom 1 idle 202
    assert not torch.equal(f.D_pr["G"].data[0], old[0])
    assert torch.equal(f.D_pr["G"].data[1], old[1])
    # no check before one epoch's worth of steps has run
    f.idle_G[1] = 10_000
    f.dead_after = 10_000
    f.track_and_resample(step=300)
    assert torch.equal(f.D_pr["G"].data[1], old[1])
    # the trainer sets the window from the training rows
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        hug_train.train(args, hug_data, m, torch.device("cpu"), Path(tmp))
    assert m.head.dead_after == steps


@pytest.mark.parametrize("fusion", ["sparse", "misa"])
def test_05b_dense_skip_width(hug_data, fusion):
    import hug_train
    base = hug_train.build(_hug_args(fusion=fusion), hug_data)
    skip = hug_train.build(_hug_args(fusion=fusion, fusion_dense_skip=True), hug_data)
    w0, w1 = base.head.head[0].in_features, skip.head.head[0].in_features
    fd = base.head.proj_G[0].out_features
    assert w1 - w0 == 2 * fd
    if fusion == "sparse":
        ctx = w0 - (base.head.m_sh + 2 * base.head.m_pr)
        assert w1 == 2 * fd + base.head.m_sh + 2 * base.head.m_pr + ctx
    snaps = hug_train.Snapshots(hug_data, skip, torch.device("cpu"))
    b = hug_train.HugBatcher(hug_data, 5)(hug_data.train_rows[:32])
    vx = snaps.enter(int(hug_data.snap[hug_data.train_rows[0]]))
    skip.train()
    out = skip(b, vx, skip.graph_tables(vx, train=True), train=True)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])


# 7 ─────────────────────────────────────────────────────────────────────────────
def test_concat_bitwise_matches_pre_spec05():
    from regression_fingerprint import fingerprint
    assert fingerprint() == N4_PRE_SPEC05_FINGERPRINT


def test_kuairand_adapter_data_bitwise_matches_pre_spec06():
    """Spec 06 A1: split, histories, features, snapshot edges/features, baseline frame."""
    from regression_fingerprint import data_fingerprint_hash
    assert data_fingerprint_hash() == KUAIRAND_PRE_SPEC06_DATA_FINGERPRINT


# 8 ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("fusion,extra", [
    ("concat", {}), ("gate", {}), ("evgate", {}), ("moe", {}), ("misa", {}),
    ("sparse", {}), ("sparse", {"k_schedule": "fixed"}),
])
def test_every_variant_runs(hug_data, fusion, extra):
    import hug_train
    args = _hug_args(fusion=fusion, **extra)
    model = hug_train.build(args, hug_data)
    snaps = hug_train.Snapshots(hug_data, model, torch.device("cpu"))
    b = hug_train.HugBatcher(hug_data, args.max_seq_len)(hug_data.train_rows[:48])
    vx = snaps.enter(int(hug_data.snap[hug_data.train_rows[0]]))
    model.train()
    out = model(b, vx, model.graph_tables(vx, train=True), train=True)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    # all aux weights 0 → loss = BCE + L2 (+ CL, here disabled)
    args0 = _hug_args(fusion=fusion, w_rec=0, w_align=0, w_dec=0, cl_weight=0, **extra)
    m0 = hug_train.build(args0, hug_data)
    if hasattr(m0.head, "balance_weight"):
        m0.head.balance_weight = 0.0
    m0.train()
    o0 = m0(b, vx, m0.graph_tables(vx, train=True), train=True)
    touched = torch.cat([b["video"], b["hist"][b["hist"] >= 0]])
    l2 = m0.emb_l2 * m0.inp.l2_touched(b["user"], touched) / len(b["user"])
    assert torch.allclose(o0["loss"], o0["bce"] + l2, atol=1e-6)


@pytest.mark.parametrize("flag", ["no_graph", "no_seq"])
def test_fusion_requires_both_views(hug_data, flag):
    import hug_train
    with pytest.raises(ValueError):
        hug_train.build(_hug_args(fusion="sparse", **{flag: True}), hug_data)
    hug_train.build(_hug_args(fusion="concat", **{flag: True}), hug_data)


# 9 ─────────────────────────────────────────────────────────────────────────────
def test_sparse_predictions_invariant_to_future(loaded_data, timed_bundle):
    import hug_train
    args = _hug_args(fusion="sparse")
    d = hug_train.prepare(args, data=loaded_data, bundle=timed_bundle)
    T = int(np.quantile(d.inter.time[d.val_rows], 0.5))
    torch.manual_seed(0)
    model = hug_train.build(args, d)

    def scores(dd):
        m2 = hug_train.build(args, dd)
        m2.load_state_dict(model.state_dict())
        snaps = hug_train.Snapshots(dd, m2, torch.device("cpu"))
        r, s = hug_train.predict(m2, dd, dd.val_rows, snaps, hug_train.HugBatcher(dd, args.max_seq_len),
                                 64, torch.device("cpu"), True, "x")
        return pd.Series(s, index=r).sort_index()

    data2 = _rewrite_after(loaded_data, T)
    d2 = hug_train.prepare(args, data=data2, bundle=HKGConstructor(data2).build())
    a, b = scores(d), scores(d2)
    early = d.inter.time[a.index.values] < T
    np.testing.assert_array_equal(a.values[early], b.values[early])


# 10 ────────────────────────────────────────────────────────────────────────────
def test_sparse_weights_independent_of_val_test(loaded_data, tmp_path):
    import hug_train
    from runtime import set_determinism

    def fit(data, sub):
        args = _hug_args(fusion="sparse", patience=10)
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


# 11 ────────────────────────────────────────────────────────────────────────────
def test_analysis_outputs(tmp_data_dir, tmp_path):
    import json
    import hug_train
    for run in ("a", "b"):
        args = _hug_args(data_dir=str(tmp_data_dir), run_dir=str(tmp_path / run), fusion="sparse",
                         max_epochs=1)
        args.min_interactions = 1
        hug_train.run(args)
    out = json.loads((tmp_path / "a" / "final_metrics.json").read_text())
    fs = out["fusion_stats"]
    assert 0.0 <= fs["rho_mean"] <= 1.0
    for per in fs["rho_by_bucket"].values():
        assert all(0.0 <= v <= 1.0 for v in per.values())
    ca = np.load(tmp_path / "a" / "val_codes.npz")
    cb = np.load(tmp_path / "b" / "val_codes.npz")
    val = np.load(tmp_path / "a" / "val_preds.npz")["row_id"]
    assert set(ca["row_id"]) <= set(val)
    assert np.array_equal(np.sort(ca["row_id"]), np.sort(cb["row_id"]))      # stable sample
    assert ca["sh_G_idx"].shape == ca["sh_G_val"].shape
