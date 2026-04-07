"""
KuaiRand Data Loader
---------------------
Reads raw CSV files from a local KuaiRand directory and returns
clean, typed DataFrames ready for HKG construction.

Supports both KuaiRand-1K and KuaiRand-27K layouts automatically.

KuaiRand-1K layout:
    log_standard_4_08_to_4_21_1k.csv
    log_standard_4_22_to_5_08_1k.csv
    log_random_4_22_to_5_08_1k.csv
    user_features_1k.csv
    video_features_basic_1k.csv
    video_features_statistic_1k.csv

KuaiRand-27K layout (multi-part files):
    log_standard_4_08_to_4_21_27k_part1.csv
    log_standard_4_08_to_4_21_27k_part2.csv
    log_standard_4_22_to_5_08_27k_part1.csv
    log_standard_4_22_to_5_08_27k_part2.csv
    log_random_4_22_to_5_08_27k.csv
    user_features_27k.csv
    video_features_basic_27k.csv
    video_features_statistic_27k_part1.csv
    video_features_statistic_27k_part2.csv
    video_features_statistic_27k_part3.csv
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── File name constants ────────────────────────────────────────────────────────

# KuaiRand-1K filenames
FILES_1K = {
    "log_std_early": ["log_standard_4_08_to_4_21_1k.csv"],
    "log_std_late":  ["log_standard_4_22_to_5_08_1k.csv"],
    "log_random":    ["log_random_4_22_to_5_08_1k.csv"],
    "user":          "user_features_1k.csv",
    "video_basic":   "video_features_basic_1k.csv",
    "video_stat":    ["video_features_statistic_1k.csv"],
}

# KuaiRand-27K filenames (multi-part splits)
FILES_27K = {
    "log_std_early": [
        "log_standard_4_08_to_4_21_27k_part1.csv",
        "log_standard_4_08_to_4_21_27k_part2.csv",
    ],
    "log_std_late": [
        "log_standard_4_22_to_5_08_27k_part1.csv",
        "log_standard_4_22_to_5_08_27k_part2.csv",
    ],
    "log_random":   ["log_random_4_22_to_5_08_27k.csv"],
    "user":          "user_features_27k.csv",
    "video_basic":   "video_features_basic_27k.csv",
    "video_stat":    [
        "video_features_statistic_27k_part1.csv",
        "video_features_statistic_27k_part2.csv",
        "video_features_statistic_27k_part3.csv",
    ],
}

# Legacy single-file constants kept for backward compatibility
LOG_STANDARD_EARLY = FILES_1K["log_std_early"][0]
LOG_STANDARD_LATE  = FILES_1K["log_std_late"][0]
LOG_RANDOM         = FILES_1K["log_random"][0]
USER_FEATURES      = FILES_1K["user"]
VIDEO_BASIC        = FILES_1K["video_basic"]
VIDEO_STATISTIC    = FILES_1K["video_stat"][0]

# Columns that are always binary (0/1) in the interaction logs
BINARY_LOG_COLS = [
    "is_click", "is_like", "is_follow", "is_comment",
    "is_forward", "is_hate", "long_view", "is_profile_enter", "is_rand",
]

# Session boundary: gap larger than this (ms) → new session
SESSION_GAP_MS = 30 * 60 * 1000  # 30 minutes


@dataclass
class KuaiRandData:
    """Container returned by KuaiRandLoader.load()."""

    log_standard:      pd.DataFrame   # combined standard-policy interaction logs
    log_random:        pd.DataFrame   # random-policy interaction logs
    user_features:     pd.DataFrame   # one row per user
    video_basic:       pd.DataFrame   # one row per video (basic metadata)
    video_statistic:   pd.DataFrame   # one row per video (aggregate statistics)

    # Derived artefacts populated during load
    log_combined:      pd.DataFrame = field(default_factory=pd.DataFrame)
    session_map:       pd.DataFrame = field(default_factory=pd.DataFrame)  # interaction → session_id

    # Re-indexed id maps (str → contiguous int)
    user_id_map:       dict = field(default_factory=dict)
    video_id_map:      dict = field(default_factory=dict)
    author_id_map:     dict = field(default_factory=dict)
    category_id_map:   dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"KuaiRandData("
            f"users={len(self.user_features)}, "
            f"videos={len(self.video_basic)}, "
            f"std_interactions={len(self.log_standard)}, "
            f"rand_interactions={len(self.log_random)}, "
            f"sessions={self.session_map['session_id'].nunique() if len(self.session_map) else 0}"
            f")"
        )


class KuaiRandLoader:
    """
    Loads KuaiRand-1K from a local directory into typed DataFrames.

    Parameters
    ----------
    data_dir : str | Path
        Root directory containing the six KuaiRand-1K CSV files.
    min_interactions : int
        Drop users with fewer than this many interactions (removes extreme
        cold-start noise before graph construction).
    filter_ads : bool
        When True, AD-type videos are removed from the video table and all
        logs referencing them.  Set False to model ads explicitly.
    """

    def __init__(
        self,
        data_dir: str | Path,
        min_interactions: int = 10,
        filter_ads: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.min_interactions = min_interactions
        self.filter_ads = filter_ads
        self._files = self._detect_scale()
        self._validate_directory()

    def _detect_scale(self) -> dict:
        """Auto-detect whether this is a 1K or 27K directory by probing filenames."""
        if (self.data_dir / FILES_27K["user"]).exists():
            logger.info("Detected KuaiRand-27K layout")
            return FILES_27K
        logger.info("Detected KuaiRand-1K layout")
        return FILES_1K

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self) -> KuaiRandData:
        """Read, clean, and join all KuaiRand-1K files.

        Returns
        -------
        KuaiRandData
            Fully populated data container.
        """
        logger.info("Loading KuaiRand-1K from %s", self.data_dir)

        log_std   = self._load_logs(standard=True)
        log_rand  = self._load_logs(standard=False)
        user_feat = self._load_user_features()
        vid_basic = self._load_video_basic()
        vid_stat  = self._load_video_statistic()

        if self.filter_ads:
            ad_ids   = set(vid_basic.loc[vid_basic["video_type"] == "AD", "video_id"])
            vid_basic = vid_basic[vid_basic["video_type"] == "NORMAL"].copy()
            log_std   = log_std[~log_std["video_id"].isin(ad_ids)].copy()
            log_rand  = log_rand[~log_rand["video_id"].isin(ad_ids)].copy()
            logger.info("Filtered %d AD videos", len(ad_ids))

        log_std  = self._filter_active_users(log_std)
        log_combined = self._combine_logs(log_std, log_rand)
        session_map  = self._derive_sessions(log_combined)

        data = KuaiRandData(
            log_standard    = log_std,
            log_random      = log_rand,
            user_features   = user_feat,
            video_basic     = vid_basic,
            video_statistic = vid_stat,
            log_combined    = log_combined,
            session_map     = session_map,
        )

        data.user_id_map     = self._build_id_map(data.user_features["user_id"])
        data.video_id_map    = self._build_id_map(data.video_basic["video_id"])
        data.author_id_map   = self._build_id_map(data.video_basic["author_id"].dropna())
        data.category_id_map = self._build_category_map(data.video_basic)

        logger.info("Load complete → %s", data)
        return data

    # ── Private helpers ────────────────────────────────────────────────────────

    def _validate_directory(self) -> None:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"data_dir not found: {self.data_dir}")
        f = self._files
        required = (
            f["log_std_early"] + f["log_std_late"] + f["log_random"]
            + [f["user"], f["video_basic"]]
            + f["video_stat"]
        )
        missing = [fn for fn in required if not (self.data_dir / fn).exists()]
        if missing:
            scale = "27K" if f is FILES_27K else "1K"
            raise FileNotFoundError(
                f"Missing KuaiRand-{scale} files in {self.data_dir}:\n"
                + "\n".join(f"  {fn}" for fn in missing)
            )

    # Columns actually needed from the interaction logs — everything else is dropped on read
    LOG_USECOLS = [
        "user_id", "video_id", "time_ms", "play_time_ms", "duration_ms",
        "is_click", "is_like", "is_follow", "is_comment", "is_forward",
        "is_hate", "long_view", "is_profile_enter", "is_rand", "tab",
    ]
    # Compact dtypes applied immediately after each file is read
    LOG_DTYPES = {
        "user_id":          np.int32,
        "video_id":         np.int32,
        "time_ms":          np.int64,
        "play_time_ms":     np.float32,
        "duration_ms":      np.float32,
        "is_click":         np.int8,
        "is_like":          np.int8,
        "is_follow":        np.int8,
        "is_comment":       np.int8,
        "is_forward":       np.int8,
        "is_hate":          np.int8,
        "long_view":        np.int8,
        "is_profile_enter": np.int8,
        "is_rand":          np.int8,
        "tab":              np.int8,
    }

    def _load_logs(self, standard: bool) -> pd.DataFrame:
        parts = (
            self._files["log_std_early"] + self._files["log_std_late"]
            if standard else self._files["log_random"]
        )
        is_rand_val = np.int8(0 if standard else 1)

        # Determine which usecols actually exist in the files (is_rand may be absent)
        first_header = pd.read_csv(
            self.data_dir / parts[0], nrows=0
        ).columns.tolist()
        usecols = [c for c in self.LOG_USECOLS if c in first_header]
        dtypes  = {c: v for c, v in self.LOG_DTYPES.items() if c in usecols}

        # Read and process one file at a time — never hold more than one raw
        # DataFrame in memory simultaneously
        chunks: list[pd.DataFrame] = []
        for fname in parts:
            logger.info("Reading %s …", fname)
            part = pd.read_csv(
                self.data_dir / fname,
                usecols=usecols,
                dtype=dtypes,
                low_memory=False,
            )
            # Fill any missing binary columns with 0
            for col in BINARY_LOG_COLS:
                if col in part.columns:
                    part[col] = part[col].fillna(0).astype(np.int8)

            # Derived play_ratio immediately while only this part is in RAM
            part["play_ratio"] = (
                part["play_time_ms"] / part["duration_ms"].replace(0, np.nan)
            ).clip(0, 1).fillna(0).astype(np.float32)

            part["is_rand"] = is_rand_val
            chunks.append(part)

        df = pd.concat(chunks, ignore_index=True)
        del chunks   # free the list of parts immediately

        df = df.sort_values(["user_id", "time_ms"]).reset_index(drop=True)
        logger.info(
            "Loaded %s log: %d rows  RAM ~%.1f GB",
            "standard" if standard else "random",
            len(df),
            df.memory_usage(deep=True).sum() / 1e9,
        )
        return df

    def _load_user_features(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_dir / self._files["user"])

        # Ordinal-encode activity degree
        activity_order = {"full_active": 3, "high_active": 2, "middle_active": 1, "UNKNOWN": 0}
        df["activity_level"] = df["user_active_degree"].map(activity_order).fillna(0).astype(np.int8)

        # Log-scale social counts
        for col in ["follow_user_num", "fans_user_num", "friend_user_num", "register_days"]:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col].fillna(0)).astype(np.float32)

        return df

    def _load_video_basic(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_dir / self._files["video_basic"])

        # Parse multi-valued tag column into a list
        df["tag_list"] = (
            df["tag"].fillna("").astype(str)
            .apply(lambda x: [int(t) for t in x.split(",") if t.strip().isdigit()])
        )

        # Aspect ratio
        df["aspect_ratio"] = (
            df["server_width"] / df["server_height"].replace(0, np.nan)
        ).fillna(1.0).astype(np.float32)

        # Duration in seconds (more readable)
        df["duration_s"] = (df["video_duration"] / 1000).fillna(0).astype(np.float32)

        return df

    def _load_video_statistic(self) -> pd.DataFrame:
        parts = self._files["video_stat"]

        # Read column names from first part to determine what exists
        first_header = pd.read_csv(
            self.data_dir / parts[0], nrows=0
        ).columns.tolist()

        # Only keep columns we actually use downstream
        stat_cols = [
            "video_id", "show_cnt", "valid_play_cnt", "play_cnt",
            "like_cnt", "follow_cnt", "share_cnt", "collect_cnt", "comment_cnt",
        ]
        usecols = [c for c in stat_cols if c in first_header]

        chunks = []
        for fname in parts:
            logger.info("Reading %s …", fname)
            part = pd.read_csv(
                self.data_dir / fname,
                usecols=usecols,
                dtype={"video_id": np.int32},
                low_memory=False,
            )
            chunks.append(part)

        df = pd.concat(chunks, ignore_index=True) if len(chunks) > 1 else chunks[0]
        del chunks

        # Global CVR prior: valid plays / shows
        df["global_cvr"] = (
            df["valid_play_cnt"] / df["show_cnt"].replace(0, np.nan)
        ).clip(0, 1).fillna(0).astype(np.float32)

        # Log-transform skewed popularity counts
        for col in ["show_cnt", "play_cnt", "like_cnt", "follow_cnt",
                    "share_cnt", "collect_cnt", "comment_cnt"]:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col].fillna(0)).astype(np.float32)

        return df

    def _filter_active_users(self, log: pd.DataFrame) -> pd.DataFrame:
        counts = log["user_id"].value_counts()
        active = counts[counts >= self.min_interactions].index
        filtered = log[log["user_id"].isin(active)].copy()
        logger.info(
            "User filter: kept %d / %d users (min_interactions=%d)",
            len(active), counts.shape[0], self.min_interactions,
        )
        return filtered

    def _combine_logs(self, log_std: pd.DataFrame, log_rand: pd.DataFrame) -> pd.DataFrame:
        combined = pd.concat([log_std, log_rand], ignore_index=True)
        combined = combined.sort_values(["user_id", "time_ms"]).reset_index(drop=True)
        return combined

    def _derive_sessions(self, log: pd.DataFrame) -> pd.DataFrame:
        """Assign a unique session_id to each (user, continuous-activity-window) block."""
        df = log[["user_id", "time_ms"]].copy()
        df["prev_time"] = df.groupby("user_id")["time_ms"].shift(1)
        df["gap_ms"]    = df["time_ms"] - df["prev_time"]

        # New session when gap > threshold OR first interaction for user
        df["new_session"] = (df["gap_ms"] > SESSION_GAP_MS) | df["gap_ms"].isna()
        df["session_id"]  = df["new_session"].cumsum().astype(np.int32)

        logger.info("Derived %d sessions for %d users",
                    df["session_id"].nunique(), log["user_id"].nunique())
        return df[["session_id"]].join(log.reset_index(drop=True))

    @staticmethod
    def _build_id_map(series: pd.Series) -> dict:
        """Map raw IDs → contiguous integers starting at 0."""
        unique = sorted(series.dropna().unique())
        return {raw_id: idx for idx, raw_id in enumerate(unique)}

    @staticmethod
    def _build_category_map(video_basic: pd.DataFrame) -> dict:
        """Collect all unique category tag integers and map to indices."""
        all_tags: set[int] = set()
        for tag_list in video_basic["tag_list"]:
            all_tags.update(tag_list)
        return {tag: idx for idx, tag in enumerate(sorted(all_tags))}