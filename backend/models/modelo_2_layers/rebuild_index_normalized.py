"""
Script para recrear el índice FAISS de 2 capas con normalización del vector final.
"""
import pickle
import faiss
import numpy as np
from pathlib import Path

# Rutas
PKL_PATH = Path(__file__).parent / "dataset_features_2_layers.pkl"
INDEX_PATH = Path(__file__).parent / "faiss_index_2_layers.idx"

print("Cargando features existentes...")
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

features = np.array(data["features"], dtype="float32")
paths = data["paths"]

print(f"Features cargados: {features.shape}")

# Normalizar cada vector (fila)
print("Normalizando vectores...")
norms = np.linalg.norm(features, axis=1, keepdims=True)
# Evitar división por cero
norms[norms == 0] = 1
features_normalized = features / norms

print(f"Features normalizados: {features_normalized.shape}")

# Crear el índice FAISS
dim = features_normalized.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(features_normalized)

# Guardar índice
faiss.write_index(index, str(INDEX_PATH))
print(f"Índice FAISS guardado en: {INDEX_PATH}")

# Guardar pkl actualizado con features normalizados
with open(PKL_PATH, "wb") as f:
    pickle.dump({"features": features_normalized, "paths": paths}, f)
print(f"Features normalizados guardados en: {PKL_PATH}")

print("¡Listo!")
