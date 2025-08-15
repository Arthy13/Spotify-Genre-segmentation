import os

# Path to your dataset
DATASET_PATH = os.getenv("DATASET_PATH", "data/spotify.csv")

# Clustering parameters
N_CLUSTERS = int(os.getenv("N_CLUSTERS", "8"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# Limit rows for heavy endpoints (like scatter) to keep UI snappy
MAX_SCATTER_ROWS = int(os.getenv("MAX_SCATTER_ROWS", "5000"))
