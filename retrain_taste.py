
import argparse, pandas as pd
from auto_dj.io.cache import read_parquet
from auto_dj.taste.model import fit_logreg, save_taste

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--likes", required=True, help="Parquet/CSV with at least a 'track_id' column")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tracks = read_parquet(args.tracks)
    likes = pd.read_csv(args.likes) if args.likes.endswith(".csv") else read_parquet(args.likes)
    tm = fit_logreg(tracks, likes)
    save_taste(tm, args.out)
    print(f"Saved taste model → {args.out}")

if __name__ == "__main__":
    main()
