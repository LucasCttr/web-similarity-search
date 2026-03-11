#!/usr/bin/env python3
import sys
from pathlib import Path
import pickle
import numpy as np
import faiss

# Asegurarnos del path al backend para poder importar el extractor
here = Path(__file__).resolve().parent
backend_dir = here.parent
sys.path.insert(0, str(backend_dir))

from features.features import _ensure_model, extract_mixed_features, set_active_layer_config

ROOT = backend_dir
SITIOS_DIR = ROOT / 'sitios'
MODELS = {
    "1_layer": {
        "index_path": ROOT / "models/faiss_index.idx",
        "features_path": ROOT / "models/dataset_1_Layers_avg_pool.pkl",
    },
    "2_layers": {
        "index_path": ROOT / "models/modelo_2_layers/faiss_index_2_layers.idx",
        "features_path": ROOT / "models/modelo_2_layers/dataset_features_2_layers.pkl",
    }
}

if not SITIOS_DIR.exists():
    print(f"No existe carpeta de imágenes: {SITIOS_DIR}")
    sys.exit(1)

# Gather image files
img_files = [p for p in SITIOS_DIR.iterdir() if p.is_file()]
print(f"Encontradas {len(img_files)} archivos en {SITIOS_DIR}")
if len(img_files) == 0:
    print("No hay imágenes para procesar. Abortando.")
    sys.exit(1)

# Ensure model initialized
_ensure_model()

for model_key, paths in MODELS.items():
    print(f"\n=== Reconstruyendo modelo: {model_key} ===")
    set_active_layer_config(model_key)
    feats_list = []
    rel_paths = []
    for img in img_files:
        try:
            vec = extract_mixed_features(str(img))
            vec = np.array(vec, dtype='float32')
            feats_list.append(vec)
            rel_paths.append(str(Path('sitios') / img.name).replace('\\', '/'))
        except Exception as e:
            print(f"Error extrayendo features de {img}: {e}")
    if len(feats_list) == 0:
        print(f"No se extrajeron features para {model_key}; creando archivo vacío.")
        # infer dim via one dummy
        try:
            dummy = np.zeros((224,224,3), dtype=np.uint8)
            vec = extract_mixed_features(dummy)
            dim = int(len(vec))
        except Exception:
            dim = 2048
        features_arr = np.empty((0, dim), dtype='float32')
    else:
        features_arr = np.stack(feats_list, axis=0).astype('float32')
        dim = features_arr.shape[1]

    feats_path = paths['features_path']
    idx_path = paths['index_path']
    feats_path.parent.mkdir(parents=True, exist_ok=True)
    # Save pickle
    with open(feats_path, 'wb') as f:
        pickle.dump({"features": features_arr, "paths": rel_paths}, f)
    print(f"Guardado PKL: {feats_path} (count={features_arr.shape[0]}, dim={dim})")
    # Build FAISS index
    index = faiss.IndexFlatL2(dim)
    if features_arr.shape[0] > 0:
        index.add(features_arr)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(idx_path))
    print(f"Guardado índice FAISS: {idx_path} (ntotal={getattr(index,'ntotal',0)})")

print('\nReconstrucción completa.')
