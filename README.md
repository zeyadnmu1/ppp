# Playlist Auto-DJ (Mood-Aware Music Recommender)

Pull Spotify playlists & audio features, learn your taste + track moods, and auto-generate playlists for a target vibe.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .      # or: pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Spotify fetch → curate
```bash
export SPOTIFY_CLIENT_ID=... SPOTIFY_CLIENT_SECRET=...
python scripts/fetch_spotify.py --list scripts/playlists.txt --out data/raw/spotify_tracks.parquet
python scripts/curate_data.py --in data/raw/spotify_tracks.parquet --out data/curated/tracks.parquet
```

## Train taste model
```bash
python scripts/retrain_taste.py --tracks data/curated/tracks.parquet --likes data/curated/likes.parquet --out models/taste.joblib
```

## Create Spotify playlist
```bash
python scripts/create_playlist.py --name "Auto-DJ – energetic workout" --csv exports/recs.csv
```

MIT © 2025


### Demo without Spotify (CSV fallback)
The app now accepts **CSV or Parquet** paths. A tiny demo CSV is included:
- `data/curated/tracks.csv`
- `data/curated/likes.csv`

In the app, set the path to `data/curated/tracks.csv` if you haven't fetched Spotify yet.
