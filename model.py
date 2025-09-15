
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Any
import joblib, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

FEATS = ["valence","energy","danceability","tempo_z","loudness","acousticness","popularity","explicit"]

@dataclass
class TasteModel:
    model: Any
    features: List[str]

def fit_logreg(tracks: pd.DataFrame, likes: pd.DataFrame) -> TasteModel:
    pos_ids = set(likes["track_id"].astype(str))
    pos = tracks[tracks["track_id"].astype(str).isin(pos_ids)]
    neg = tracks[~tracks["track_id"].astype(str).isin(pos_ids)]
    if len(pos) == 0 or len(neg) == 0:
        raise RuntimeError("Need both positive and negative examples to train taste model.")
    neg = neg.sample(min(len(neg), len(pos)*2), random_state=42)
    X = pd.concat([pos, neg])[FEATS].fillna(0); y = [1]*len(pos) + [0]*len(neg)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    lr = LogisticRegression(max_iter=200, class_weight="balanced"); lr.fit(Xtr, ytr)
    return TasteModel(model=lr, features=FEATS)

def save_taste(tm: TasteModel, path: str) -> None:
    joblib.dump({"model": tm.model, "features": tm.features}, path)

def load_taste(path: str) -> TasteModel:
    d = joblib.load(path); return TasteModel(model=d["model"], features=d["features"])

def predict_proba(tm: TasteModel, df: pd.DataFrame):
    X = df[tm.features].fillna(0)
    if hasattr(tm.model, "predict_proba"):
        return tm.model.predict_proba(X)[:,1]
    import numpy as np
    s = tm.model.decision_function(X); return 1/(1+np.exp(-s))
