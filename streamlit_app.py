import os
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# --------------------------- Config ---------------------------
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")





st.set_page_config(page_title="Spotify Genre Segmentation", layout="wide")

# --------------------------- Helpers --------------------------
@st.cache_data(show_spinner=False)
def get_summary():
    r = requests.get(f"{API_URL}/summary", timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def get_corr():
    r = requests.get(f"{API_URL}/correlation", timeout=60)
    r.raise_for_status()
    data = r.json()
    cols = data["columns"]
    mat = np.array(data["matrix"])
    return cols, mat

@st.cache_data(show_spinner=False)
def get_scatter(limit=3000):
    r = requests.get(f"{API_URL}/scatter", params={"limit": limit}, timeout=60)
    r.raise_for_status()
    return r.json()

def search_tracks(q, limit=20):
    r = requests.get(f"{API_URL}/search_tracks", params={"q": q, "limit": limit}, timeout=30)
    if r.status_code == 200:
        return r.json().get("results", [])
    return []

def recommend(track_name=None, features=None, k=10):
    payload = {"track_name": track_name, "features": features, "k": k}
    r = requests.post(f"{API_URL}/recommend", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["results"]

# --------------------------- UI -------------------------------
st.title("🎧 Spotify Songs – Genre Segmentation Dashboard")

# Health check
health = requests.get(f"{API_URL}/health").json()
if health.get("status") != "ok":
    st.error("Backend not ready. Start FastAPI with `uvicorn app.main:app --reload` and ensure dataset at `data/spotify.csv`.")
    st.stop()

summary = get_summary()
left, mid, right = st.columns(3)
left.metric("Rows (raw)", summary["n_rows_raw"])
mid.metric("Rows (used)", summary["n_rows_used"])
right.metric("Features used", summary["n_features"])

st.caption(f"Features: {', '.join(summary['features'])}")
st.caption(f"Clusters: {summary['n_clusters']}  |  Silhouette: {summary.get('silhouette')}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔗 Correlation", "🎯 Clusters", "✨ Recommend"])

with tab1:
    st.subheader("Sample Rows")
    # Show first 100 rows from summary via another quick call (reconstruct from scatter meta not feasible)
    st.write("Below is a random sample from the processed dataset used for clustering.")
    scatter = get_scatter(limit=1000)
    df = pd.DataFrame({
        "pc1": scatter["pc1"],
        "pc2": scatter["pc2"],
        "cluster": scatter["cluster"],
        "genre": scatter.get("genre", [None]*len(scatter["pc1"])),
        "playlist": scatter.get("playlist", [None]*len(scatter["pc1"])),
    })
    st.dataframe(df.head(100))

with tab2:
    st.subheader("Correlation Matrix")
    cols, mat = get_corr()
    fig = px.imshow(mat, labels=dict(color="corr"), x=cols, y=cols)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("PCA Scatter by Cluster")
    scatter = get_scatter(limit=3000)
    sdf = pd.DataFrame(scatter)
    color_by = st.selectbox("Color by", options=[c for c in ["cluster", "genre", "playlist"] if c in sdf.columns], index=0)
    fig = px.scatter(sdf, x="pc1", y="pc2", color=color_by, hover_data=[c for c in sdf.columns if c not in ["pc1","pc2"]])
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Get Recommendations")
    col1, col2 = st.columns([2,1])
    with col1:
        mode = st.radio("Choose input mode", ["By track name", "By manual features"])
        if mode == "By track name":
            q = st.text_input("Search a track")
            if q:
                options = search_tracks(q)
                choice = st.selectbox("Pick a track", options) if options else None
                k = st.slider("How many recommendations?", min_value=3, max_value=20, value=8)
                if st.button("Recommend") and choice:
                    results = recommend(track_name=choice, k=k)
                    st.write(pd.DataFrame(results))
            else:
                st.info("Type a few letters of the song name to search.")
        else:
            st.write("Enter feature values (leave blank for auto-fill with column means)")
            feats = {f: st.number_input(f, value=np.nan) for f in summary["features"]}
            k = st.slider("How many recommendations?", min_value=3, max_value=20, value=8)
            if st.button("Recommend"):
                results = recommend(features=feats, k=k)
                st.write(pd.DataFrame(results))

    with col2:
        st.info("Tip: Recommendations are based on cosine similarity of scaled audio features within the same KMeans cluster.")
