"""
Deterministic CPU fingerprint of a short HUG N4 training run (spec 05 test 7).

    python regression_fingerprint.py            # prints sha256 of params + val predictions

Builds a fixed synthetic dataset (seeded), trains N4 for 2 epochs on CPU with
--fusion concat semantics, and hashes every parameter and the validation
scores.  Run against an older checkout (sys.path) to confirm a refactor is
bitwise result-preserving.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def synthetic_dir(root: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    n_users, n_videos, n_rows = 20, 50, 600

    def log(n, s):
        r = np.random.default_rng(s)
        return pd.DataFrame({
            "user_id": r.integers(0, n_users, n), "video_id": r.integers(0, n_videos, n),
            "time_ms": r.integers(1_650_000_000_000, 1_650_100_000_000, n),
            **{c: r.integers(0, 2, n) for c in ["is_click", "is_like", "is_follow", "is_comment",
                                                "is_forward", "is_hate", "long_view",
                                                "is_profile_enter", "is_rand"]},
            "play_time_ms": r.integers(1000, 30000, n), "duration_ms": r.integers(5000, 60000, n),
            "tab": r.integers(0, 15, n)})

    root.mkdir(parents=True, exist_ok=True)
    log(n_rows, 1).to_csv(root / "log_standard_4_08_to_4_21_1k.csv", index=False)
    log(n_rows, 2).to_csv(root / "log_standard_4_22_to_5_08_1k.csv", index=False)
    log(n_rows // 3, 3).to_csv(root / "log_random_4_22_to_5_08_1k.csv", index=False)
    pd.DataFrame({
        "user_id": range(n_users), "user_active_degree": ["full_active", "high_active"] * (n_users // 2),
        "is_lowactive_period": 0, "is_live_streamer": 0, "is_video_author": 1,
        "follow_user_num": rng.integers(0, 500, n_users), "fans_user_num": rng.integers(0, 1000, n_users),
        "friend_user_num": rng.integers(0, 100, n_users), "register_days": rng.integers(100, 2000, n_users),
        **{f"onehot_feat{i}": rng.integers(0, 5, n_users) for i in range(18)},
    }).to_csv(root / "user_features_1k.csv", index=False)
    pd.DataFrame({
        "video_id": range(n_videos), "author_id": rng.integers(0, 10, n_videos), "video_type": "NORMAL",
        "upload_dt": "2022-04-01", "upload_type": "ShortImport", "visible_status": 1,
        "video_duration": rng.integers(5000, 60000, n_videos), "server_width": 720, "server_height": 1280,
        "music_id": rng.integers(0, 100, n_videos), "music_type": rng.integers(0, 5, n_videos),
        "tag": [",".join(str(t) for t in rng.integers(0, 8, 3)) for _ in range(n_videos)],
    }).to_csv(root / "video_features_basic_1k.csv", index=False)
    return root


def fingerprint() -> str:
    import hug_train
    import main as pipeline
    from runtime import set_determinism
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = synthetic_dir(Path(tmp) / "data")
        args = pipeline.parse_args([
            "--data-dir", str(data_dir), "--cache-dir", "none", "--device", "cpu", "--quiet",
            "--min-interactions", "1", "--emb-dim", "8", "--min-id-count", "2",
            "--snapshot-hours", "4", "--batch-size", "64", "--max-epochs", "2",
            "--patience", "10", "--seq-layers", "1", "--max-seq-len", "5"])
        d = hug_train.prepare(args)
        set_determinism(0)
        model = hug_train.build(args, d)
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        hug_train.train(args, d, model, torch.device("cpu"), run_dir)
        snaps = hug_train.Snapshots(d, model, torch.device("cpu"))
        _, scores = hug_train.predict(model, d, d.val_rows, snaps,
                                      hug_train.HugBatcher(d, args.max_seq_len), 64,
                                      torch.device("cpu"), True, "val")
        h = hashlib.sha256()
        for t in model.state_dict().values():            # values in registration order
            h.update(t.detach().cpu().numpy().tobytes())
        h.update(scores.astype(np.float32).tobytes())
        return h.hexdigest()


if __name__ == "__main__":
    print(fingerprint())
