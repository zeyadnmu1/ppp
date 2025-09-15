
from __future__ import annotations
import numpy as np, pandas as pd

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "tempo" in df.columns:
        df["tempo_z"] = (pd.to_numeric(df["tempo"], errors="coerce") - 120.0) / 30.0
    else:
        df["tempo_z"] = 0.0
    for col in ["valence","energy","danceability","acousticness","loudness","popularity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["energy_bucket"] = pd.cut(df["energy"], bins=[-np.inf,0.33,0.66,np.inf], labels=["low","mid","high"])
    df["acoustic_band"] = pd.cut(df["acousticness"], bins=[-np.inf,0.33,0.66,np.inf], labels=["low","mid","high"])
    return df
