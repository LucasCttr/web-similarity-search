#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import pickle
import numpy as np
import faiss

# Asegurar que backend esté en sys.path
here = Path(__file__).resolve().parent
backend_dir = here.parent
sys.path.insert(0, str(backend_dir))

try:
    from features.features import _ensure_model, extract_mixed_features
except Exception as e:
    print(f"Error importing extractor: {e}")
    raise

print("Inicializando extractor para inferir dimensión de features (puede tardar)...")
# Inicializa modelo
_ensure_model()
# Crear dummy para inferir dimensión
try:
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    vec = extract_mixed_features(dummy)
    dim = int(len(vec))
    print(f"Dimensión inferida: {dim}")
except Exception as e:
    print(f"No se pudo inferir dimensión desde extractor: {e}")
    dim = 2048
    print(f"Usando fallback dim={dim}")

models_dir = backend_dir / 'models'
models_dir.mkdir(parents=True, exist_ok=True)

pkl_path = models_dir / 'dataset_1_Layers_avg_pool.pkl'
idx_path = models_dir / 'faiss_index.idx'

# Crear pickle vacío con features.shape = (0, dim)
empty_feats = np.empty((0, dim), dtype='float32')
empty_paths = []
with open(pkl_path, 'wb') as f:
    pickle.dump({'features': empty_feats, 'paths': empty_paths}, f)
print(f"Wrote empty PKL to: {pkl_path}")

# Crear índice vacío
index = faiss.IndexFlatL2(dim)
faiss.write_index(index, str(idx_path))
print(f"Wrote empty FAISS index to: {idx_path}")

print('Rebuild completo.')
