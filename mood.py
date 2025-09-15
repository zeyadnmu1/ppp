
from __future__ import annotations
import numpy as np, pandas as pd

def assign_mood_row(row: pd.Series) -> str:
    v = row.get("valence", np.nan); e = row.get("energy", np.nan); d = row.get("danceability", np.nan)
    if np.isfinite(v) and np.isfinite(e):
        if v >= 0.6 and e >= 0.6: return "happy"
        if v < 0.4 and e < 0.4:  return "sad"
        if (np.isfinite(d) and d >= 0.6) or e >= 0.7: return "energetic"
    return "calm"

def add_mood(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df["mood"] = df.apply(assign_mood_row, axis=1); return df
