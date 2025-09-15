
from __future__ import annotations
import numpy as np, pandas as pd
from auto_dj.taste.model import TasteModel, predict_proba

def mood_alignment(row: pd.Series, target_valence: float, target_energy: float) -> float:
    d = np.linalg.norm([row.get("valence",0)-target_valence, row.get("energy",0)-target_energy])
    return float(1 - min(d/np.sqrt(2), 1))

def rerank(candidates: pd.DataFrame,
           taste: TasteModel | None,
           target_valence: float,
           target_energy: float,
           alpha: float = 0.7,
           epsilon: float = 0.1,
           k: int = 25,
           max_per_artist: int = 3) -> pd.DataFrame:
    c = candidates.copy()
    align = c.apply(lambda r: mood_alignment(r, target_valence, target_energy), axis=1).astype(float).values
    taste_scores = np.zeros(len(c))
    if taste is not None:
        try: taste_scores = predict_proba(taste, c)
        except Exception: pass
    score = alpha * taste_scores + (1 - alpha) * align
    c = c.assign(score=score).sort_values("score", ascending=False)

    out = []; artist_counts = {}
    for _, row in c.iterrows():
        aids = (row.get("artist_ids") or "").split(",")
        key = aids[0] if aids and aids[0] else row.get("artist_names")
        artist_counts[key] = artist_counts.get(key, 0)
        if artist_counts[key] < max_per_artist:
            out.append(row); artist_counts[key] += 1
        if len(out) >= k: break
    topk = pd.DataFrame(out) if out else c.head(k)

    if epsilon > 0 and len(c) > k:
        explore_pool = c.iloc[k:k+10]
        n_explore = max(1, int(epsilon * k))
        if len(explore_pool) > 0:
            explore = explore_pool.sample(min(n_explore, len(explore_pool)), random_state=42)
            mix = pd.concat([topk.iloc[:k-n_explore], explore]).sort_values("score", ascending=False)
            return mix.head(k)
    return topk.head(k)
