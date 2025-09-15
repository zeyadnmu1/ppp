
import argparse, pandas as pd
from pathlib import Path
from auto_dj.io.spotify_client import make_client, fetch_playlist_tracks, fetch_audio_features
from auto_dj.io.cache import to_parquet, merge_meta_features, dedupe_tracks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="Text file with one Spotify playlist URL per line")
    ap.add_argument("--out", required=True, help="Output parquet path (raw)")
    args = ap.parse_args()

    sp = make_client()
    urls = [l.strip() for l in Path(args.list).read_text(encoding="utf-8").splitlines() if l.strip()]
    meta_frames = []
    for u in urls:
        rows = fetch_playlist_tracks(sp, u)
        if rows: meta_frames.append(pd.DataFrame(rows))
    if not meta_frames: raise SystemExit("No tracks fetched.")
    meta = pd.concat(meta_frames, ignore_index=True)
    ids = meta["track_id"].dropna().astype(str).unique().tolist()
    feats = pd.DataFrame(fetch_audio_features(sp, ids))
    df = merge_meta_features(meta, feats); df = dedupe_tracks(df)
    to_parquet(df, args.out); print(f"Wrote {args.out} with shape {df.shape}")

if __name__ == "__main__":
    main()
