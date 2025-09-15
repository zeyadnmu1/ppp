
from auto_dj.features.mood import assign_mood_row
import pandas as pd

def test_mood_rules():
    assert assign_mood_row(pd.Series({"valence":0.8,"energy":0.8})) == "happy"
    assert assign_mood_row(pd.Series({"valence":0.2,"energy":0.2})) == "sad"
    assert assign_mood_row(pd.Series({"valence":0.5,"energy":0.75})) == "energetic"
    assert assign_mood_row(pd.Series({"valence":0.5,"energy":0.4,"danceability":0.7})) == "energetic"
