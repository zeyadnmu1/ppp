
from __future__ import annotations
import pandas as pd

def filter_candidates(df: pd.DataFrame,
                      mood: str | None = None,
                      tempo_min: float | None = None,
                      tempo_max: float | None = None,
                      energy_min: float | None = None,
                      energy_max: float | None = None,
                      exclude_ids: set[str] | None = None) -> pd.DataFrame:
    q = df.copy()
    if mood:        q = q[q["mood"] == mood]
    if tempo_min is not None: q = q[q["tempo"] >= tempo_min]
    if tempo_max is not None: q = q[q["tempo"] <= tempo_max]
    if energy_min is not None: q = q[q["energy"] >= energy_min]
    if energy_max is not None: q = q[q["energy"] <= energy_max]
    if exclude_ids: q = q[~q["track_id"].astype(str).isin(exclude_ids)]
    return q
