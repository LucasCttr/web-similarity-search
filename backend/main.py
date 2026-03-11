"""
main.py
---------
Backend principal de la aplicación de búsqueda de imágenes por similitud.

Responsabilidades:
- Expone endpoints para subir imágenes y buscar imágenes similares.
- Gestiona el almacenamiento de imágenes en disco.
- Mantiene y persiste los features de imágenes y sus rutas en un archivo pickle y un índice FAISS.
- Inicializa el modelo de extracción de features al arrancar para evitar latencia en la primera búsqueda.

Endpoints principales:
- POST /upload: Sube una imagen, extrae sus features y la agrega al índice.
- POST /search: Busca imágenes similares a una imagen dada.
- GET /sitios/<archivo>: Sirve las imágenes almacenadas.

Fuente de verdad: El archivo pickle y el índice FAISS.
No se usa base de datos relacional.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from pathlib import Path
import shutil
import uuid
import faiss, pickle, numpy as np
from features.features import extract_mixed_features, _ensure_model, set_active_layer_config
import subprocess
import sys
from io import BytesIO
from PIL import Image

_ensure_model()  # Inicializa el modelo de features al arrancar
# Cargar índice FAISS

# --- Multi-model/índice ---
MODELS = {
    "1_layer": {
        "index_path": Path(__file__).parent / "models/faiss_index.idx",
        "features_path": Path(__file__).parent / "models/dataset_1_Layers_avg_pool.pkl"
    },
    "2_layers": {
        "index_path": Path(__file__).parent / "models/modelo_2_layers/faiss_index_2_layers.idx",
        "features_path": Path(__file__).parent / "models/modelo_2_layers/dataset_features_2_layers.pkl"
    }
}

# Modelo activo por defecto
active_model = "1_layer"

def load_model_data(model_key):
    idx_path = MODELS[model_key]["index_path"]
    feats_path = MODELS[model_key]["features_path"]

    # Ensure extractor is configured for this model so we can determine vector dimension if needed
    try:
        set_active_layer_config(model_key)
    except Exception:
        pass

    index = None
    dataset_features = []
    dataset_paths = []

    # Load features pickle if present
    if feats_path.exists():
        try:
            with open(feats_path, "rb") as f:
                data = pickle.load(f)
            dataset_features = list(data.get("features", []))
            dataset_paths = data.get("paths", [])
        except Exception as e:
            print(f"[LOAD MODEL] Warning: no se pudo leer features en {feats_path}: {e}")

    # Load or create FAISS index. If index file missing, create a new empty index
    if idx_path.exists():
        try:
            index = faiss.read_index(str(idx_path))
        except Exception as e:
            print(f"[LOAD MODEL] Warning: no se pudo leer índice FAISS en {idx_path}: {e}")
            index = None

    if index is None:
        # Try to infer dimension from extractor by running a dummy image through it
        try:
            import numpy as _np
            dummy = _np.zeros((224, 224, 3), dtype=_np.uint8)
            vec = extract_mixed_features(dummy)
            dim = int(len(vec))
        except Exception as e:
            print(f"[LOAD MODEL] Warning: no se pudo inferir dimensión de features: {e}")
            # Fallback conservative default
            dim = 2048
        index = faiss.IndexFlatL2(dim)

    # If we loaded an index with entries but have no dataset paths, try to rebuild
    try:
        if getattr(index, 'ntotal', 0) > 0 and len(dataset_paths) == 0:
            print(f"[LOAD MODEL] Índice {model_key} tiene {index.ntotal} entradas pero no hay paths; intentando rebuild automático...")
            try:
                rebuild_index_for_model(model_key)
                # reload after rebuild
                index, dataset_features, dataset_paths = load_model_data(model_key)
            except Exception as e:
                print(f"[LOAD MODEL] Rebuild automático falló para {model_key}: {e}")
    except Exception:
        pass

    return index, dataset_features, dataset_paths


def rebuild_index_for_model(model_key):
    """Attempt to rebuild the FAISS index for the given model_key using available scripts or PKL files.

    This will run scripts located under backend/models/<subdir> when present, or recreate the index
    from the features pickle if available.
    """
    base_models_dir = Path(__file__).parent / "models"
    # Special-case known subdir for 2_layers
    subdirs = ["modelo_2_layers", "modelos_base"]
    target_dir = None
    for sd in subdirs:
        d = base_models_dir / sd
        if d.exists():
            # check for presence of model-specific files
            if model_key == "2_layers" and sd == "modelo_2_layers":
                target_dir = d
                break
    # If no specific dir chosen, use base models dir
    if target_dir is None:
        target_dir = base_models_dir

    # Look for known rebuild scripts
    scripts = ["rebuild_index_normalized.py", "create_faiss_index_2_layers.py"]
    for s in scripts:
        script_path = target_dir / s
        if script_path.exists():
            print(f"[REBUILD] Ejecutando script: {script_path}")
            subprocess.run([sys.executable, str(script_path)], check=True)
            # after running script, try to reload index file
            return True

    # Fallback: try to recreate index from the PKL referenced in MODELS
    feats_path = MODELS[model_key]["features_path"]
    idx_path = MODELS[model_key]["index_path"]
    if feats_path.exists():
        try:
            with open(feats_path, "rb") as f:
                data = pickle.load(f)
            features = np.array(data.get("features", []), dtype="float32")
            if features.size == 0:
                raise RuntimeError("No hay features en el PKL para reconstruir el índice")
            dim = features.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(features)
            faiss.write_index(index, str(idx_path))
            print(f"[REBUILD] Índice reconstruido desde {feats_path} y guardado en {idx_path}")
            return True
        except Exception as e:
            raise RuntimeError(f"No se pudo reconstruir índice desde {feats_path}: {e}")

    raise RuntimeError(f"No se encontró script de rebuild ni PKL para el modelo {model_key}")

# Inicializar ambos modelos en memoria
model_data = {}
for key in MODELS:
    model_data[key] = {}
    model_data[key]["index"], model_data[key]["features"], model_data[key]["paths"] = load_model_data(key)
# Restaurar configuración de capas al modelo activo después de inicializar índices
try:
    set_active_layer_config(active_model)
except Exception:
    pass
def get_active():
    return model_data[active_model]["index"], model_data[active_model]["features"], model_data[active_model]["paths"]

# Normalizar rutas y limpiar entradas sin archivo físico
def _clean_dataset_and_index():
    """
    Limpia el dataset y el índice FAISS eliminando entradas cuyos archivos no existen.
    Persiste el resultado limpio en el pickle y el índice FAISS.
    """
    # Limpia solo el modelo activo
    global active_model, model_data
    index, dataset_features, dataset_paths = get_active()
    valid_feats = []
    valid_paths = []
    for feat, path in zip(dataset_features, dataset_paths):
        fname = Path(path).name
        rel_dir = Path("sitios/") if not("sitios" in fname) else Path("")
        rel = str(rel_dir / fname).replace("\\", "/")
        full_path = Path(__file__).parent / rel
        if full_path.exists():
            valid_feats.append(feat)
            valid_paths.append(path)
        else:
            print(f"[CLEAN] Eliminando entrada sin archivo físico: {full_path}")
    # Solo actualiza en memoria
    model_data[active_model]["features"] = valid_feats
    model_data[active_model]["paths"] = valid_paths

_clean_dataset_and_index()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).parent / "sitios"
UPLOAD_DIR.mkdir(exist_ok=True)

# Mounting static directories
DATASET_ROOT = Path(__file__).parent / "sitios"       # ajusta si tus imágenes están en otra carpeta
UPLOAD_ROOT = UPLOAD_DIR

app.mount("/dataset", StaticFiles(directory=DATASET_ROOT), name="dataset")
app.mount("/sitios", StaticFiles(directory=UPLOAD_ROOT), name="sitios")



@app.post('/upload')
async def upload_image(file: UploadFile = File(...)):
    """
    Sube una imagen, la guarda en disco, extrae sus features y la agrega a AMBOS índices FAISS.
    Persiste los features y rutas en ambos pickles.
    """
    try:
        # Guardar archivo en disco
        global active_model, model_data
        print(f"[UPLOAD DEBUG] Recibido archivo: {file.filename}")
        ext = Path(file.filename).suffix or '.jpg'
        image_id = str(uuid.uuid4())
        dest = UPLOAD_DIR / f"{image_id}{ext}"

        # Leer en memoria y normalizar a RGB + tamaño estándar antes de guardar
        contents = await file.read()
        print(f"[UPLOAD DEBUG] Tamaño archivo: {len(contents)} bytes")
        img = Image.open(BytesIO(contents)).convert('RGB')
        # Guardar la imagen en su tamaño original
        img.save(dest)
        print(f"[UPLOAD DEBUG] Guardado en: {dest}, existe: {dest.exists()}")

        # Verificar que el archivo se guardó
        if not dest.exists():
            print(f"[UPLOAD ERROR] Archivo NO se guardó en {dest}")
            raise Exception(f"Archivo no se guardó: {dest}")

        print(f"[UPLOAD DEBUG] Archivo verificado en disco")

        rel_for_dataset = str(Path('sitios') / dest.name).replace('\\', '/')

        # Agregar a AMBOS modelos/índices
        print(f"[UPLOAD DEBUG] Agregando a {len(MODELS)} modelos: {list(MODELS.keys())}")
        for model_key in MODELS:
            try:
                print(f"[UPLOAD DEBUG] Procesando modelo: {model_key}")
                # Cambiar configuración de capas para este modelo
                set_active_layer_config(model_key)
                
                # Extraer features con la configuración de capas de este modelo
                feats_vec = extract_mixed_features(str(dest))
                feats = feats_vec.astype("float32")
                print(f"[UPLOAD DEBUG] Features extraídas para {model_key}: shape={feats.shape}")
                
                # Agregar al índice FAISS de este modelo
                model_data[model_key]["index"].add(np.expand_dims(feats, axis=0))
                
                # Actualizar dataset en memoria
                model_data[model_key]["features"].append(feats)
                model_data[model_key]["paths"].append(rel_for_dataset)
                
                # Persistir cambios de este modelo
                feats_path = MODELS[model_key]["features_path"]
                idx_path = MODELS[model_key]["index_path"]
                with open(feats_path, "wb") as f:
                    pickle.dump({
                        "features": np.array(model_data[model_key]["features"], dtype="float32"),
                        "paths": model_data[model_key]["paths"]
                    }, f)
                faiss.write_index(model_data[model_key]["index"], str(idx_path))
                
                print(f"[UPLOAD DEBUG] ✓ Features agregadas a modelo: {model_key}")
            except Exception as model_err:
                print(f"[UPLOAD ERROR] Error agregando a modelo {model_key}: {model_err}")
                import traceback
                traceback.print_exc()

        # Restaurar configuración de capas al modelo activo
        set_active_layer_config(active_model)
        
        print(f"[UPLOAD DEBUG] Path relativo guardado: {rel_for_dataset}")
        print(f"[UPLOAD] Ruta absoluta guardada: {dest}")

        return JSONResponse({'id': image_id, 'filename': file.filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/search')
async def search_image(
    request: Request,
    file: UploadFile = File(...),
    radius: float = Form(None),
    k: int = Form(10)
):
    """
    Busca imágenes similares a la imagen recibida usando el índice FAISS.
    Devuelve una lista de resultados con url pública, id y distancia.
    """
    try:
        global active_model, model_data
        index, dataset_features, dataset_paths = get_active()
        # Ensure lists exist
        if dataset_features is None:
            dataset_features = []
        if dataset_paths is None:
            dataset_paths = []
        # Leer imagen en memoria (PIL) y extraer features directamente
        contents = await file.read()
        img = Image.open(BytesIO(contents))
        img = img.convert('RGB')  # mantener tamaño original; el extractor ya hace resize interno

        feats = extract_mixed_features(img)
        feats = np.array([feats]).astype("float32")

        # Buscar en FAISS
        # Verify dimensions match
        try:
            idx_dim = getattr(index, 'd', None)
            vec_dim = feats.shape[1]
            if idx_dim is not None and vec_dim != idx_dim:
                raise HTTPException(status_code=500, detail=f"Dimension mismatch: index.d={idx_dim} vs feature_vector={vec_dim}. Rebuild index or reset model configuration.")
            distances, indices = index.search(feats, k)
        except AssertionError as ae:
            raise HTTPException(status_code=500, detail=f"FAISS assertion error during search: {ae}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error running FAISS search: {e}")

        results = []
        base_url = str(request.base_url).rstrip('/')
        for dist, idx in zip(distances[0], indices[0]):
            # idx can be -1 for invalid results; skip those
            if idx is None or int(idx) < 0:
                print(f"[SEARCH DEBUG] skipping invalid idx={idx}")
                continue
            idx = int(idx)
            if radius is not None and dist > radius:
                continue
            if idx >= len(dataset_paths):
                print(f"[SEARCH DEBUG] idx out of range: {idx} >= {len(dataset_paths)}; skipping")
                continue

            entry = dataset_paths[idx]
            rel_path = str(entry).replace("\\", "/")
            fname = Path(rel_path).name

            # DEBUG
            full_path = UPLOAD_DIR / fname
            exists = full_path.exists()
            print(f"[SEARCH DEBUG] idx={idx}, entry={entry}, fname={fname}, existe={exists}")
            if not exists:
                print(f"[SEARCH SKIP] Omitiendo resultado porque falta archivo: {full_path}")
                continue

            # Siempre servir desde /sitios usando solo el nombre de archivo
            url = f"{base_url}/sitios/{fname}"

            image_id = fname.split('.')[0]
            results.append({
                "id": image_id,
                "url": url,
                "distance": float(dist)
            })

        return {"results": results}
    except Exception as e:
        import traceback
        print("\n========== [SEARCH ERROR] ==========")
        print(f"Archivo recibido: {getattr(file, 'filename', None)}")
        print(f"Request: {request.method} {request.url}")
        print(f"Error: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        print(f"dataset_features: {len(dataset_features) if 'dataset_features' in globals() else 'N/A'}")
        print(f"dataset_paths: {len(dataset_paths) if 'dataset_paths' in globals() else 'N/A'}")
        print(f"Index size: {index.ntotal if 'index' in globals() else 'N/A'}")
        print("====================================\n")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint para alternar el modelo activo
@app.post('/set_model')
async def set_model(model: str = Form(...)):
    global active_model
    if model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo no válido. Opciones: {list(MODELS.keys())}")
    active_model = model
    # Cambiar también la configuración de capas del extractor
    set_active_layer_config(model)
    return {"active_model": active_model}


@app.get('/debug/files')
def debug_files():
    """
    Devuelve la lista de archivos presentes en la carpeta de imágenes para debug.
    """
    """Lista archivos en sitios/ para debug."""
    files = list(UPLOAD_DIR.glob('*'))
    return {
        "sitios_dir": str(UPLOAD_DIR),
        "exists": UPLOAD_DIR.exists(),
        "files": [f.name for f in files],
        "count": len(files)
    }


@app.get('/debug/index')
def debug_index():
    """Devuelve información de índices FAISS y datasets cargados para cada modelo."""
    info = {}
    for key in MODELS:
        entry = model_data.get(key, {})
        idx = entry.get("index")
        feats = entry.get("features")
        paths = entry.get("paths")
        try:
            dim = getattr(idx, 'd', None)
            ntotal = getattr(idx, 'ntotal', None)
        except Exception:
            dim = None
            ntotal = None
        # Try to infer feature vector length from in-memory features
        feat_dim = None
        if isinstance(feats, (list, tuple)) and len(feats) > 0:
            try:
                feat_dim = len(feats[0])
            except Exception:
                feat_dim = None

        info[key] = {
            "index_dim": dim,
            "index_ntotal": ntotal,
            "features_count": len(feats) if feats is not None else 0,
            "paths_count": len(paths) if paths is not None else 0,
            "feature_vector_dim": feat_dim,
        }
    return info




if __name__ == '__main__':
    import uvicorn
    # reload=False para evitar múltiples procesos y re-import pesado de TensorFlow
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=False)
