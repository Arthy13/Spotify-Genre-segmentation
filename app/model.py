from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import joblib

from .config import DATASET_PATH, N_CLUSTERS, RANDOM_STATE

# Likely numeric Spotify features
CANDIDATE_FEATURES = [
    "acousticness", "danceability", "energy", "instrumentalness", "liveness",
    "loudness", "speechiness", "valence", "tempo", "duration_ms"
]

# Common metadata columns we try to preserve for display
NAME_COLS = ["track_name", "name", "track", "song", "title"]
ARTIST_COLS = ["artist_name", "artists", "artist", "singer"]
ALBUM_COLS = ["album_name", "album"]
GENRE_COLS = ["playlist_genre", "genre", "playlist_subgenre"]
PLAYLIST_COLS = ["playlist_name", "playlist"]

def pick_first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

@dataclass
class Artifacts:
    features: List[str]
    scaler: StandardScaler
    kmeans: KMeans
    pca: PCA
    knn: NearestNeighbors
    silhouette: Optional[float] = None

class ModelService:
    def __init__(self, csv_path: str = DATASET_PATH, n_clusters: int = N_CLUSTERS):
        self.csv_path = csv_path
        self.n_clusters = n_clusters
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_proc: Optional[pd.DataFrame] = None
        self.X: Optional[np.ndarray] = None
        self.meta_cols: Dict[str, Optional[str]] = {}
        self.artifacts: Optional[Artifacts] = None
        self._load_and_fit()

    # ---------- Data loading & preprocessing ----------
    def _load_dataset(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset not found at {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        return df

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        present = [c for c in CANDIDATE_FEATURES if c in df.columns]
        if len(present) >= 5:
            return present
        # Fallback: use numeric columns (excluding ids)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = {"id", "track_id", "artist_id", "duration"}  # optional excludes
        num_cols = [c for c in num_cols if c not in exclude]
        # Keep at most 20 numeric features
        return num_cols[:20]

    def _extract_meta_cols(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        return {
            "track": pick_first_present(df, NAME_COLS),
            "artist": pick_first_present(df, ARTIST_COLS),
            "album": pick_first_present(df, ALBUM_COLS),
            "genre": pick_first_present(df, GENRE_COLS),
            "playlist": pick_first_present(df, PLAYLIST_COLS),
        }

    def _preprocess(self, df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler]:
        df_feat = df[features].copy()
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()
        scaler = StandardScaler()
        X = scaler.fit_transform(df_feat.values)
        df_proc = df.loc[df_feat.index].copy()  # align meta with clean rows
        return df_proc, X, scaler

    # ---------- Fit models ----------
    def _fit_models(self, X: np.ndarray) -> Artifacts:
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X)
        pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X)
        knn = NearestNeighbors(metric="cosine")
        knn.fit(X)
        sil = None
        if self.n_clusters > 1 and len(np.unique(labels)) > 1:
            sil = float(silhouette_score(X, labels))
        return Artifacts(features=self.features, scaler=self.scaler, kmeans=kmeans, pca=pca, knn=knn, silhouette=sil)

    def _load_and_fit(self):
        df = self._load_dataset()
        self.meta_cols = self._extract_meta_cols(df)
        self.features = self._select_features(df)
        self.df_proc, self.X, self.scaler = self._preprocess(df, self.features)
        self.artifacts = self._fit_models(self.X)
        # attach labels and PCA coords
        self.df_proc["cluster"] = self.artifacts.kmeans.predict(self.X)
        coords = self.artifacts.pca.transform(self.X)
        self.df_proc["pc1"] = coords[:, 0]
        self.df_proc["pc2"] = coords[:, 1]
        self.df_raw = df

    # ---------- Public helpers ----------
    def correlation(self) -> Dict[str, List]:
        corr = self.df_proc[self.features].corr()
        return {
            "columns": corr.columns.tolist(),
            "matrix": corr.values.tolist()
        }

    def scatter(self, limit: int = 3000) -> Dict[str, List]:
        df = self.df_proc
        if limit and len(df) > limit:
            df = df.sample(limit, random_state=RANDOM_STATE)
        out = {
            "pc1": df["pc1"].tolist(),
            "pc2": df["pc2"].tolist(),
            "cluster": df["cluster"].astype(int).tolist(),
        }
        # Optional color by genre/playlist if present
        genre_col = self.meta_cols.get("genre")
        playlist_col = self.meta_cols.get("playlist")
        if genre_col and genre_col in df.columns:
            out["genre"] = df[genre_col].astype(str).tolist()
        if playlist_col and playlist_col in df.columns:
            out["playlist"] = df[playlist_col].astype(str).tolist()
        return out

    def summary(self) -> Dict:
        info = {
            "n_rows_raw": int(len(self.df_raw)) if self.df_raw is not None else 0,
            "n_rows_used": int(len(self.df_proc)) if self.df_proc is not None else 0,
            "n_features": int(len(self.features)),
            "features": self.features,
            "meta_columns": self.meta_cols,
            "n_clusters": int(self.n_clusters),
            "silhouette": self.artifacts.silhouette if self.artifacts else None,
        }
        # cluster sizes
        counts = self.df_proc["cluster"].value_counts().sort_index()
        info["cluster_sizes"] = {int(k): int(v) for k, v in counts.items()}
        # top centroids for insight
        centroids = self.artifacts.kmeans.cluster_centers_
        # Map back to original scale mean-ish values via inverse transform
        try:
            centroids_orig = self.artifacts.scaler.inverse_transform(centroids)
            info["centroids"] = [
                {self.features[i]: float(centroids_orig[c, i]) for i in range(len(self.features))}
                for c in range(centroids.shape[0])
            ]
        except Exception:
            info["centroids"] = None
        return info

    def search_tracks(self, q: str, limit: int = 20) -> List[str]:
        col = self.meta_cols.get("track")
        if not col:
            return []
        ql = q.lower()
        s = self.df_proc[self.df_proc[col].astype(str).str.lower().str.contains(ql, na=False)]
        return s[col].astype(str).head(limit).tolist()

    def _get_track_index(self, track_name: str) -> Optional[int]:
        col = self.meta_cols.get("track")
        if not col:
            return None
        s = self.df_proc[self.df_proc[col].astype(str).str.lower() == track_name.lower()]
        if len(s) == 0:
            return None
        return int(s.index[0])

    def recommend(self, track_name: Optional[str] = None, features: Optional[Dict[str, float]] = None, k: int = 10) -> List[Dict]:
        if track_name is None and features is None:
            raise ValueError("Provide either track_name or features.")

        if track_name:
            idx = self._get_track_index(track_name)
            if idx is None:
                return []
            x = self.X[self.df_proc.index.get_loc(idx)].reshape(1, -1)
            song_cluster = int(self.df_proc.loc[idx, "cluster"])
        else:
            # Build vector from provided features in correct order
            vec = []
            for f in self.features:
                vec.append(features.get(f, np.nan) if features else np.nan)
            arr = np.array(vec, dtype=float).reshape(1, -1)
            # Handle missing by filling with column means from training data
            col_means = np.nanmean(self.df_proc[self.features].values, axis=0)
            mask = np.isnan(arr)
            arr[mask] = col_means[np.where(mask)[1]]
            x = self.artifacts.scaler.transform(arr)
            song_cluster = int(self.artifacts.kmeans.predict(x)[0])

        # Restrict candidates to same cluster
        cluster_mask = self.df_proc["cluster"] == song_cluster
        X_cluster = self.X[cluster_mask.values]
        idxs = self.df_proc[cluster_mask].index.values

        distances, indices = self.artifacts.knn.kneighbors(x, n_neighbors=min(k+1, len(X_cluster)))
        neighbors = []
        for rank, (dist, ind) in enumerate(zip(distances[0], indices[0])):
            real_idx = idxs[ind]
            if track_name and self.meta_cols.get("track") and \
               str(self.df_proc.loc[real_idx, self.meta_cols["track"]]).lower() == track_name.lower():
                # skip identical seed track
                continue
            row = self.df_proc.loc[real_idx]
            item = {
                "rank": len(neighbors) + 1,
                "distance": float(dist),
                "cluster": int(row["cluster"]),
            }
            # add meta
            for key, col in self.meta_cols.items():
                if col and col in self.df_proc.columns:
                    item[key] = str(row[col])
            neighbors.append(item)
            if len(neighbors) >= k:
                break
        return neighbors
