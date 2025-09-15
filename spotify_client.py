
from __future__ import annotations
import os, time
from typing import Dict, Any, List, Optional

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

BATCH = 100

def make_client() -> spotipy.Spotify:
    cid = os.getenv("SPOTIFY_CLIENT_ID")
    sec = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not cid or not sec:
        raise RuntimeError("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in env.")
    auth = SpotifyClientCredentials(client_id=cid, client_secret=sec)
    return spotipy.Spotify(auth_manager=auth, requests_timeout=15, retries=3)

def _extract_track_info(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    t = item.get("track") or item
    if not t or t.get("id") is None:
        return None
    return {
        "track_id": t["id"],
        "track_name": t.get("name"),
        "artist_names": ", ".join(a["name"] for a in (t.get("artists") or [])),
        "artist_ids": ", ".join(a["id"] for a in (t.get("artists") or [] if t.get("artists") else [])),
        "album_name": (t.get("album") or {}).get("name"),
        "album_id": (t.get("album") or {}).get("id"),
        "duration_ms": t.get("duration_ms"),
        "explicit": int(bool(t.get("explicit"))),
        "popularity": t.get("popularity"),
        "release_date": (t.get("album") or {}).get("release_date"),
    }

def fetch_playlist_tracks(sp: spotipy.Spotify, playlist_url_or_id: str) -> List[Dict[str, Any]]:
    pl_id = playlist_url_or_id.rstrip("/").split("/")[-1].split("?")[0]
    results = sp.playlist_items(pl_id, fields="items(track(...)),next", additional_types=("track",), limit=100)
    items = results.get("items", [])
    while results.get("next"):
        results = sp.next(results); items.extend(results.get("items", [])); time.sleep(0.05)
    rows = []
    for it in items:
        info = _extract_track_info(it)
        if info:
            info["source_playlist"] = pl_id
            rows.append(info)
    return rows

def fetch_audio_features(sp: spotipy.Spotify, track_ids: List[str]) -> List[Dict[str, Any]]:
    feats_all: List[Dict[str, Any]] = []
    for i in range(0, len(track_ids), BATCH):
        chunk = track_ids[i:i+BATCH]
        feats = sp.audio_features(chunk)
        for f in feats:
            if f and f.get("id"):
                feats_all.append({
                    "track_id": f["id"],
                    "danceability": f.get("danceability"),
                    "energy": f.get("energy"),
                    "valence": f.get("valence"),
                    "tempo": f.get("tempo"),
                    "loudness": f.get("loudness"),
                    "speechiness": f.get("speechiness"),
                    "acousticness": f.get("acousticness"),
                    "instrumentalness": f.get("instrumentalness"),
                    "liveness": f.get("liveness"),
                })
        time.sleep(0.05)
    return feats_all
