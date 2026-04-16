# HUG — Graph-based CVR Prediction with Heterogeneous Unified Knowledge Graphs

A video recommendation system that predicts Click-Through Rate (CVR) on the [KuaiRand](https://kuairand.com/) dataset. Implements both baseline models (DIN, BST) and a novel **dual-GNN architecture** with Knowledge Graph alignment.

---

## Overview

The core idea is to build a **Heterogeneous Knowledge Graph (HKG)** over users, videos, authors, and categories, then either run one Graph Neural Network (GNN) or run two GNNs in parallel. 

In the 1 GNN case:
- **Heterogeneous Graph Transformer** (HGT) - a unified encoder that runs over the full constructed HKG

In the 2 GNN case:

- **StructuralGNN** (R-GCN) — captures global preference patterns from static user-video interactions and metadata relationships
- **SequentialGNN** (GGNN) — captures session dynamics from temporal video sequences

A **KG-guided cross-attention alignment module** fuses both representations before a final CVR classifier.

| Model | Test AUC |
|---|---|
| DIN (baseline) | 0.6903 |
| BST (baseline) | 0.7009 |
| HUG-Unified | 0.7260 |
| HUG-Dual (Knowledge Graph Alignment Off) | 0.7344 |
| **HUG-Dual (Knowledge Graph Alignment On)** | **0.7398** |

---

## Project Structure

```
GCIC/
├── Framework/              # Dual-GNN model (main codebase)
│   ├── main.py             # Training & evaluation entry point
│   ├── data_loader.py      # KuaiRand CSV loader
│   ├── hkg_constructor.py  # Builds the heterogeneous knowledge graph
│   ├── gnn_encoders.py     # StructuralGNN (R-GCN) & SequentialGNN (GGNN)
│   ├── models.py           # AlignmentModule & CVRHead
│   └── tests.py            # Unit tests
│
├── Baselines/              # FuxiCTR-based DIN & BST baselines
│   ├── config/
│   │   ├── dataset_config.yaml
│   │   └── model_config.yaml
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── Plots/
│   └── view_results.py     # Bar chart comparisons across models
│
├── runs/                   # Output metrics & logs
└── environment.yml
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate gcic
```

For baselines, also install FuxiCTR:

```bash
pip install fuxictr pyyaml
```

---

## Data

Download **KuaiRand-1K** (or KuaiRand-27K) and place the CSV files in a directory. The expected files are:

```
KuaiRand-1K/data/
├── log_standard_4_08_to_4_21_1k.csv      # Interaction log (period 1)
├── log_standard_4_22_to_5_08_1k.csv      # Interaction log (period 2)
├── log_random_4_22_to_5_08_1k.csv        # Random policy interactions
├── user_features_1k.csv
├── video_features_basic_1k.csv
└── video_features_statistic_1k.csv
```

Key columns: `user_id`, `video_id`, `is_click`, `time_ms`, `session_id`, `is_rand`, and video metadata (author, category, duration).

---

## Training the Dual-GNN (Framework)

```bash
python Framework/main.py \
  --data-dir /path/to/KuaiRand-1K/data \
  --model-type dual \
  --kg-alignment 1 \
  --epochs 10
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | *(required)* | Path to KuaiRand CSV files |
| `--model-type` | `dual` | `dual` (proposed) or `single` (HGT baseline) |
| `--kg-alignment` | `0` | KG relation gating dim (0 = disabled) |
| `--hidden-dim` | `128` | GNN hidden layer width |
| `--out-dim` | `64` | Output embedding dimension |
| `--epochs` | `10` | Training epochs |
| `--batch-size` | `2048` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--cache-dir` | `./cache/<scale>` | HKG cache & checkpoint directory |
| `--output-dir` | `./runs` | Metrics JSON output directory |
| `--device` | *(auto)* | `cuda` / `mps` / `cpu` |
| `--no-amp` | `False` | Disable fp16 mixed precision |
| `--eval-only` | `False` | Load checkpoint and evaluate only |
| `--multi-gpu` | `False` | Split model across 2 GPUs |
| `--test-ratio` | `0.2` | Fraction of data held out for testing |
| `--max-seq-len` | `50` | Max session sequence length |
| `--min-interactions` | `10` | Min interactions per user (cold-start filter) |

The HKG is built and serialized to `--cache-dir` on first run. Subsequent runs load it in seconds.

---

## Training the Baselines (DIN / BST)

```bash
# Step 1 — preprocess (builds behavior sequences)
python Baselines/preprocess.py

# Step 2 — train
python Baselines/train.py --model DIN --gpu 0
python Baselines/train.py --model BST --gpu 0

# Step 3 — evaluate
python Baselines/evaluate.py --model both --gpu 0
```

Hyperparameters are in [`Baselines/config/model_config.yaml`](Baselines/config/model_config.yaml). See [`Baselines/BASELINE_STEPS.md`](Baselines/BASELINE_STEPS.md) for a full walkthrough.

---

## Outputs

Results are saved under `runs/kuairand_<scale>_<model_type>/`:

- **`final_metrics.json`** — best-checkpoint evaluation
- **`history.json`** — per-epoch metrics (AUC, AP, LogLoss, nDCG@10, training time)

Example:

```json
{
  "test": {
    "auc": 0.7398,
    "ap": 0.6225,
    "log_loss": 0.5737,
    "ndcg_at_10": 1.0
  },
  "best_auc": 0.7398
}
```

Model checkpoints are saved to `--cache-dir` as `best_model_dual.pt` / `best_model_single.pt`.

---

## Plotting Results

```bash
python Plots/view_results.py
```

Generates bar charts comparing AUC, AP, and nDCG@10 across all model variants.

---

## Tests

```bash
pytest Framework/tests.py
```

---

## Notes

- **Temporal split**: train/test split is done chronologically to prevent data leakage.
- **IPS weighting**: corrects for policy bias between random (`is_rand=1`) and standard (`is_rand=0`) interactions.
- **Sessions**: defined by 30-minute inactivity gaps in `time_ms`.
