# Prompt for the agent on the idea cluster (ZhihuRec site)

Paste everything below the line into a Claude session on the idea A100 machine.

---

**Role.** You run the **ZhihuRec** experiments for the HUG paper (WWW'27; full paper due
2026-10-25) on this machine: one A100 80GB. Design and code happen on another machine
("brains"): an architect session writes the specs and a coding session implements them. Code
reaches you only through git. **Don't modify anything under `Framework/`, `Baselines/`,
`scripts/` or `experiments/`.** If something needs a code fix, stop and report it with the error
and your proposed diff, and it will be relayed. Don't touch other users' GPU processes. In
particular, leave the vLLM server on the other A100 alone.

**1. Code.** The repo already exists here.
- Run `git status` first. If there are local changes or untracked files from earlier work,
  **don't discard them**. List them and wait for a decision.
- GitHub moved the repo to `git@github.com:brains-group/HUG.git`. The old `GCIC` URL still
  redirects, but run `git remote set-url origin git@github.com:brains-group/HUG.git` anyway.
- Once it's clean: `git fetch origin && git checkout docs/technical-report-and-revision-plan
  && git pull --ff-only`. The commit must be `0155b4d` or later. Report the hash. Spec 06
  (the MIND/ZhihuRec adapters) is already merged into this branch, so you don't need any other
  branch.
- For context, read `docs/specs/04-heavy-run-readiness.md`, `05-sparse-fusion.md`,
  `06-datasets.md` and `docs/HEAVY_RUN.md`. Spec 06 is the one that matters for you.

**2. Environment.** If an `hkg-env` conda env already exists, check it against
`environment.yml`: python 3.10, torch 2.5.1 + CUDA 12.1, pyg 2.6.1, fuxictr 2.3.9, numpy 1.26.4,
pandas 2.3.3, scikit-learn 1.7.2, polars 1.39.3, pyarrow 23.0.1.
- If any version differs, update the env to match. Or, if the existing env is used by other
  work, create a new one with `conda env create -f environment.yml -n hkg-env-www27`.
- Then run `bash Baselines/setup_fuxictr.sh` (it pins FuxiCTR at `b7dff73`).
- The driver here is 595 / CUDA 13.2, which runs CUDA 12.1 wheels fine.
- Check that `python -c "import torch; print(torch.cuda.is_available(),
  torch.cuda.get_device_name(0))"` reports the A100.
- Use that env's interpreter for everything, and report its path.

**3. Data: download it here; it can't be copied from brains.** Both datasets are git-ignored.
Download them into these paths under the repo root, then check every file against the checksums
below. **A checksum mismatch means stop and report.**

KuaiRand-1K (1.1 GB tarball, 4.4 GB extracted; needed for the port check):
```bash
curl -L -o /tmp/KuaiRand-1K.tar.gz "https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz?download=1"
tar -xzf /tmp/KuaiRand-1K.tar.gz -C .        # must yield ./KuaiRand-1K/data/*.csv
```

ZhihuRec (THUIR Seafile share; research use only). Download only these 8 files and skip
`info_token.csv.gz`, which isn't used:
```bash
mkdir -p ZhihuRec/raw
for f in info_answer.csv.gz info_author.csv.gz info_question.csv.gz info_topic.csv.gz \
         info_user.csv.gz inter_impression.csv.gz inter_query.csv.gz README.md; do
  curl -L --fail -o "ZhihuRec/raw/$f" \
    "https://cloud.tsinghua.edu.cn/d/d6c045c55aa14bb39ebc/files/?p=%2F$f&dl=1"
done
```

Checksums from brains (`sha256sum -c` against this list):
```
e98841eabb3b078c812c4646bf0c5e3f0c947a4183dd950cb091e32377b9a849  KuaiRand-1K/data/log_random_4_22_to_5_08_1k.csv
355355897a84baa4df26b78b0271bb9a27b127a39cd7aa4a898cf86db0bc1810  KuaiRand-1K/data/log_standard_4_08_to_4_21_1k.csv
548daf771e54e2b73086cc8e7c6f56787d44421f5026c2100421810e47ae9dd4  KuaiRand-1K/data/log_standard_4_22_to_5_08_1k.csv
07813068ac9ca1071dd456c6d84bd20d0dd81a8ff6596a22e1bbe2b12dc5ea6d  KuaiRand-1K/data/user_features_1k.csv
18cb8b9133635a5e53d4e33b1644137b8673b615cc1aaa500b6d3d09695efae3  KuaiRand-1K/data/video_features_basic_1k.csv
5951175389697705dec4a4f992f891f0df3d32584b024f63e96191dc9f0939c5  KuaiRand-1K/data/video_features_statistic_1k.csv
829cab56a760ccf49adb617ba4cc40fcbf1e4a5ec26801579a8224e1c7ef7056  ZhihuRec/raw/info_answer.csv.gz
94d2715c7c90bae09a26a932b6ba4ba8681d52c1db4fa339cb428da8c512f8a2  ZhihuRec/raw/info_author.csv.gz
c46b331fe9bb7a1ece2bbce63ff25f682886fcf03b703024f63926e670215bdd  ZhihuRec/raw/info_question.csv.gz
d39610fa640f18cd00ab04cc2823495d91e3ed424c727ed511ae830ce0a740c2  ZhihuRec/raw/info_topic.csv.gz
5aa09ab37284c6b85da4301152798788655186989e0803490727b2d721b581a7  ZhihuRec/raw/info_user.csv.gz
9fe25bcdef578761ca01c1ba1cb3c883f5f13add1046c4c7601ecc17c3c9a95c  ZhihuRec/raw/inter_impression.csv.gz
ae7fb8dec5e6fddda8adc9b32ad5034cf7ae48acf50dfc0aa1fd74ae65db7cb0  ZhihuRec/raw/inter_query.csv.gz
13073a0a667e8dacc69204b0e9f501474940679f4ddc008b90b7eb34a811f2b9  ZhihuRec/raw/README.md
```

If an old `Framework/cache/` or `Baselines/data/` exists here, move it aside (for example to
`Framework/cache.old`) rather than reusing it, because it may have been built by older code.

**4. Port check. Do this now; it doesn't wait for anything on brains.**
a. Synthetic tests: `cd Framework && python -m pytest tests.py test_fusion.py test_harness.py -x -q`.
   They must all pass.
b. Data fingerprint: `python Framework/fingerprint.py --data-dir KuaiRand-1K/data --out
   runs/port/data_fingerprint.json`. It must show exactly `t_val=1651305018060`,
   `t_test=1651544133374` and rows `train=8187361, val=1169625, test=2339250`. Any difference
   means the data or code doesn't match brains. Stop and report.
c. One N4 epoch: `python Framework/main.py --model-type hug --cache-dir Framework/cache --device
   cuda:0 --run-dir runs/port/N4 --seed 42 --max-epochs 1 --quiet`. Brains' reference for this
   exact run: **val AUC 0.7720** (AP 0.6604, n=1,169,625), 1746 s on a shared H100. Expect the
   AUC to match to within about ±0.003. A100 and H100 floating-point arithmetic differs
   slightly, so results won't be bitwise identical. A bigger gap is a bug, so report it.
d. Report back: the commit hash; the env path; the checksum result; the pytest summary; the
   fingerprint JSON; val AUC/AP; seconds per epoch; peak GPU memory (it's recorded in
   `runs/port/N4/final_metrics.json` → `history`); and `nvidia-smi` at peak. The seconds per
   epoch sets the ZhihuRec budget.

**5. ZhihuRec smoke (right after the port check).** Read `docs/specs/06-datasets-results.md`
first. Its GPU items are still pending, and you're running the ZhihuRec ones.
- Build the bundle and the baseline CSVs: `Baselines/preprocess.py --dataset zhihurec`.
- Run one HUG N4 epoch with `--dataset zhihurec`, and one 50-step smoke of each baseline on
  ZhihuRec.
- Do a **dry run** of the queue restricted to ZhihuRec:
  `python scripts/run_queue.py experiments/heavy_run.yaml --gpus 0 --dry-run --max-steps 50
  --only '*_zhihurec'`. Check the job list it selects. It must contain only `*_zhihurec` jobs.
- Report the bundle's sizes and splits against the results doc (12,504,877 rows; quantile
  fallback split), seconds per epoch, peak GPU memory, the dry-run outcome and a time
  estimate for the 59 ZhihuRec jobs.

You may also be asked to run the Spec 05b fusion probe (`docs/specs/05b-fusion-fixes.md`,
§Probe) on KuaiRand here, when its code lands. Run it only when asked, and run one job at a
time on this GPU. **Don't
launch the real run until it's approved.**

**6. Rules for the real run.**
- Use `--runs-root runs/heavy` and run `nohup` as in `docs/HEAVY_RUN.md`.
- Validation-only stages first. Never run `final_eval` or anything with `--eval-test` without
  explicit sign-off.
- Never edit code mid-run. If you `git pull`, check that `Framework/runtime.py`'s
  `CODE_COMPAT_VERSION` didn't change. If it did, stop and ask.
- Results go back to brains by copying `runs/heavy/` (job directories are named by hash, so
  merging is safe). If the machines can't reach each other, use a tarball through whatever
  transfer path is available, and report its sha256.
- Report progress by job counts and any failures. Don't summarise results as conclusions; the
  architect reads the numbers.
