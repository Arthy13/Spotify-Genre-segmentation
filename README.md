# Spotify Songs – Genre Segmentation

A project using **Streamlit** for dashboard and **FastAPI** for backend.  
Features:
- Data preprocessing
- Visualizations (distributions, correlation, clusters)
- K-Means clustering for genre segmentation
- Content-based recommendations using cosine similarity

> Place your CSV at `data/spotify.csv`.

---

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
streamlit run streamlit_app.py
Streamlit connects to backend at http://localhost:8000.

Key Notes
Insights:

High energy/danceability clusters align with pop/EDM

Acousticness negatively correlates with energy

Model:

StandardScaler + KMeans (k=8)

Cosine similarity for recommendations

pgsql
Copy
Edit
