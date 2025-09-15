
from __future__ import annotations
def precision_at_k(recommended_ids, true_liked_ids, k: int) -> float:
    topk = recommended_ids[:k]; hits = sum(1 for tid in topk if tid in true_liked_ids); return hits / max(1, k)
def recall_at_k(recommended_ids, true_liked_ids, k: int) -> float:
    topk = set(recommended_ids[:k]); hits = len(topk & set(true_liked_ids)); return hits / max(1, len(true_liked_ids))
