
import argparse
from auto_dj.io.cache import read_parquet, to_parquet, dedupe_tracks
from auto_dj.features.mood import add_mood
from auto_dj.features.engineer import engineer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = read_parquet(args.inp)
    df = dedupe_tracks(df)
    df = engineer(add_mood(df))
    to_parquet(df, args.out)
    print(f"Curated → {args.out}, shape={df.shape}")

if __name__ == "__main__":
    main()
