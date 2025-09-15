
import pandas as pd
from auto_dj.recs.rerank import rerank

def test_rerank_basic():
    df = pd.DataFrame({
        "track_id": ["a","b","c","d","e"],
        "track_name": ["A","B","C","D","E"],
        "artist_names": ["X","Y","Z","X","Y"],
        "artist_ids": ["x","y","z","x","y"],
        "valence": [0.8,0.2,0.6,0.4,0.7],
        "energy":  [0.7,0.3,0.8,0.2,0.9],
        "danceability":[0.7,0.4,0.6,0.3,0.8],
        "tempo":[120,90,140,80,150],
        "loudness":[-6,-12,-7,-15,-5],
        "acousticness":[0.1,0.6,0.2,0.7,0.05],
        "popularity":[50,20,70,10,80],
        "explicit":[0,0,0,0,0]
    })
    out = rerank(df, None, target_valence=0.7, target_energy=0.8, alpha=0.7, epsilon=0.0, k=3)
    assert len(out) == 3
