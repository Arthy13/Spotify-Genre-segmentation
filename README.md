# Spotify Songs – Genre Segmentation (Streamlit + FastAPI)

This is a **ready-to-run minor project** for your Corizo internship submission.  
It uses **Streamlit** for the dashboard and **FastAPI** for the backend API.  
Core tasks covered:
- Data preprocessing
- EDA & visualizations (distributions, correlation matrix, cluster scatter)
- Clustering for **genre/playlist segmentation** (K-Means)
- Simple **content-based recommendation** using cosine similarity within clusters

> **Dataset**: put your CSV at `data/spotify.csv` (or set `DATASET_PATH` in env).

---

## 1) Setup

```bash
# (Optional) create & activate a virtual env
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install deps
pip install -r requirements.txt
```

## 2) Start the Backend (FastAPI)

```bash
# Start API (reload for dev)
uvicorn app.main:app --reload
# It will read CSV from data/spotify.csv. Change via env:
# DATASET_PATH="path/to/your.csv" uvicorn app.main:app --reload
```

- Health check: `GET http://localhost:8000/health`
- Summary: `GET http://localhost:8000/summary`
- Correlation: `GET http://localhost:8000/correlation`
- Scatter (PCA): `GET http://localhost:8000/scatter?limit=3000`
- Search tracks: `GET http://localhost:8000/search_tracks?q=love`
- Recommend: `POST http://localhost:8000/recommend` with either a `track_name` or raw `features` in JSON.

## 3) Start the Frontend (Streamlit)

In a **new terminal** (keep API running):

```bash
streamlit run streamlit_app.py
```

- Streamlit will call the local API at `http://localhost:8000`.  
- To point to a different URL, set `API_URL` env var or edit the top of `streamlit_app.py`.

## 4) Files & Folders

```
spotify-genre-segmentation/
├─ app/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ model.py         # Data load, preprocess, cluster, recommend
│  └─ main.py          # FastAPI endpoints
├─ data/
│  └─ spotify.csv      # <-- put your dataset here (not included)
├─ models/             # Saved scaler/model/artifacts
├─ streamlit_app.py    # Streamlit dashboard UI
├─ requirements.txt
└─ README.md
```

## 5) What to Show in Your Submission

- Screenshots from Streamlit tabs:
  - **Data Overview** (head, basic stats)
  - **Correlation Matrix**
  - **Clusters** (2D scatter via PCA colored by cluster/genre)
  - **Recommendations** (input a song → similar songs)
- A short note on **insights** (e.g., clusters with high energy/danceability align with pop/EDM; acousticness correlates negatively with energy, etc.).
- Mention the model choices: **StandardScaler + KMeans (k=8)** by default; cosine similarity for recommendations.

## 6) Extend (Optional)
- Try different k (clusters) and report silhouette score.
- Use UMAP for non-linear 2D projection.
- Add filters for `playlist_genre` / `playlist_name`.
- Add endpoint `/train` to retrain with new k.
