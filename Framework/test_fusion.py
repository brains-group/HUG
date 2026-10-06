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
    f.n_ref.copy_(torch.tensor([4.0, 4.0]))
    e_G, e_S, ctx, ev = _views()
    p = dict(zip(("G", "S"), f.project(e_G, e_S)))
    sh, pr = f.codes(p, ev)
    kp = f.k_private(ev)
    for i, v in enumerate(("G", "S")):
        assert ((sh[v] != 0).sum(-1) <= 3).all()
        assert ((pr[v] != 0).sum(-1) <= kp[:, i]).all()
        # exactly k when the ReLU leaves enough positive entries
        a = torch.relu(f.enc[v](p[v]))
        ok = (a[:, f.m_sh:] > 0).sum(-1) >= kp[:, i]
        assert ((pr[v] != 0).sum(-1)[ok] == kp[ok, i]).all()
    assert (kp >= 2).all() and (kp <= 8).all()
    grid = torch.linspace(0, 10, 50).unsqueeze(1).repeat(1, 2)
    k = f.k_private(grid)[:, 0]
    assert (k[1:] >= k[:-1]).all()
    fixed = _sparse(k_schedule="fixed", k_private_fixed=5)
    assert (fixed.k_private(ev) == 5).all()


# 2 ─────────────────────────────────────────────────────────────────────────────
def test_evidence_causal_with_ties():
    from features import view_evidence
    rng = np.random.default_rng(0)
    n = 300
    fr = pd.DataFrame({"video_id": rng.integers(0, 6, n), "time_ms": rng.integers(0, 30, n)})
    meta = pd.DataFrame({"video_id": range(6), "author_id": [0, 0, 1, 1, 2, 2]})
    user = rng.integers(0, 4, n)
    click = rng.integers(0, 2, n)
    ev = view_evidence(fr, meta, user, click)
    auth = fr.video_id.map(dict(zip(meta.video_id, meta.author_id))).to_numpy()
    for i in range(n):
        t = fr.time_ms[i]
        item = int(((fr.video_id == fr.video_id[i]) & (fr.time_ms < t)).sum())
        a = int(((auth == auth[i]) & (fr.time_ms.to_numpy() < t)).sum())
        uc = int(((user == user[i]) & (click == 1) & (fr.time_ms.to_numpy() < t)).sum())
        assert ev[i, 0] == pytest.approx(np.log1p(item) + np.log1p(a), abs=1e-5)
        assert ev[i, 1] == pytest.approx(np.log1p(uc) + np.log1p(item), abs=1e-5)


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


# 3 ─────────────────────────────────────────────────────────────────────────────
def test_atoms_unit_norm_after_step():
    f = _sparse()
    f.n_ref.copy_(torch.tensor([3.0, 3.0]))
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
    f.n_ref.copy_(torch.tensor([3.0, 3.0]))
    g = _grads(f, w_rec=1.0)
    assert not any(v for n, v in g.items() if n.startswith("proj_"))      # detached target
    assert g["D_sh"] and g["D_pr.G"] and g["enc.G.weight"]
    g = _grads(f, w_align=1.0)
    assert not g["D_pr.G"] and not g["D_pr.S"]
    assert g["proj_G.0.weight"]
    # only shared encoder rows get alignment gradient
    f.zero_grad(); f.w_rec = f.w_dec = 0; f.w_align = 1.0
    f.fuse(*_views())["aux_loss"].backward()
    assert f.enc["G"].weight.grad[f.m_sh:].abs().sum() == 0
    f.zero_grad(); f.w_align = 0; f.w_dec = 1.0
    f.fuse(*_views())["aux_loss"].backward()
    assert f.enc["G"].weight.grad[:f.m_sh].abs().sum() == 0


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
    f.n_ref.copy_(torch.tensor([3.0, 3.0]))
    with torch.no_grad():                   # private-G atom 0 can never fire …
        f.enc["G"].weight[f.m_sh + 0] = 0.0
        f.enc["G"].bias[f.m_sh + 0] = -100.0
    f.fuse(*_views())
    f.idle_G[0] = 9_999                     # … and has been idle long enough to be dead
    live_before = f.D_pr["G"].data[1:].clone()
    old0 = f.D_pr["G"].data[0].clone()
    f.track_and_resample(step=1000, every=1000, dead_after=10_000)
    assert not torch.equal(f.D_pr["G"].data[0], old0)
    assert torch.equal(f.D_pr["G"].data[1:], live_before)
    assert f.idle_G[0] == 0


# 7 ─────────────────────────────────────────────────────────────────────────────
def test_concat_bitwise_matches_pre_spec05():
    from regression_fingerprint import fingerprint
    assert fingerprint() == N4_PRE_SPEC05_FINGERPRINT


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
