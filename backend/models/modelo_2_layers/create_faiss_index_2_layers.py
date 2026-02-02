import pickle
import faiss
import numpy as np
from pathlib import Path

# Ruta al archivo de features extraídos con el modelo de 2 capas
PKL_PATH = Path(__file__).parent / "dataset_features_2_layers.pkl"
INDEX_PATH = Path(__file__).parent / "faiss_index_2_layers.idx"

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)
features = np.array(data["features"], dtype="float32")

# Crear el índice FAISS (L2 o Cosine, según tu uso)
dim = features.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(features)

faiss.write_index(index, str(INDEX_PATH))
print(f"Índice FAISS creado y guardado en: {INDEX_PATH}")
