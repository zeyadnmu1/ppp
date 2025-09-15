
from __future__ import annotations
from pathlib import Path
import pandas as pd

def to_parquet(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def dedupe_tracks(df: pd.DataFrame) -> pd.DataFrame:
    if "track_id" in df.columns:
        df = df.sort_values("popularity", ascending=False).drop_duplicates("track_id", keep="first")
    return df

def merge_meta_features(meta: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    return meta.merge(feats, on="track_id", how="left")
