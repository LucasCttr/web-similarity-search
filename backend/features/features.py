"""
features.py
-----------
Módulo de extracción de features para imágenes usando ResNet50 (Keras/TensorFlow).

Responsabilidades:
- Carga perezosa y gestión del modelo ResNet50 para extracción de features.
- Provee funciones para preparar imágenes y extraer vectores de características.
- Normaliza y preprocesa imágenes para que sean compatibles con el modelo.

Funciones principales:
- _ensure_model(): Inicializa el modelo y lo mantiene en caché global.
- prepare_image(): Prepara una imagen (ruta, PIL o array) para el modelo.
- extract_mixed_features(): Extrae el vector de features de una imagen.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input

_tf = None
_preprocess_input = None
_keras_image = None
_feature_extractors = None
_base_model = None

MOSTRAR_PROGRESO_TF = 0  # 0 para silenciar logs en predict

# Configuraciones de capas por modelo
LAYER_CONFIGS = {
    "1_layer": ['avg_pool'],
    "2_layers": ['conv3_block4_out', 'avg_pool']
}

# Modelo activo para extracción (por defecto 1_layer)
_active_layer_config = "1_layer"

def set_active_layer_config(config_name):
    """Cambia la configuración de capas activa para extracción de features."""
    global _active_layer_config, _feature_extractors
    if config_name not in LAYER_CONFIGS:
        raise ValueError(f"Configuración no válida: {config_name}. Opciones: {list(LAYER_CONFIGS.keys())}")
    _active_layer_config = config_name
    # Forzar recarga de extractores con las nuevas capas
    _feature_extractors = None
    _ensure_model()

def _ensure_model():
    """
    Inicializa y cachea el modelo ResNet50 y los extractores de features.
    Se llama automáticamente antes de extraer features, pero puede llamarse
    explícitamente al arrancar la app para evitar latencia en la primera búsqueda.
    """
    global _tf, _preprocess_input, _keras_image, _feature_extractors, _base_model
    
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

    # Cargar modelo base solo una vez
    if _base_model is None:
        _base_model = tf.keras.applications.ResNet50(weights='imagenet')
    
    # Crear extractores según la configuración activa
    if _feature_extractors is None:
        layer_names = LAYER_CONFIGS[_active_layer_config]
        _feature_extractors = {
            name: tf.keras.Model(
                inputs=_base_model.input,
                outputs=_base_model.get_layer(name).output,
            )
            for name in layer_names
        }

    _tf = tf
    _preprocess_input = preprocess_input
    _keras_image = keras_image
    return _tf, _preprocess_input, _keras_image, _feature_extractors


def prepare_image(img_source, target_size=(224, 224)):
    """
    Prepara una imagen para el modelo:
    - Si es ruta, la carga y redimensiona.
    - Si es PIL.Image, la redimensiona.
    - Si es np.ndarray, la usa directamente.
    Devuelve el batch listo para el modelo.
    """
    """Acepta path, PIL.Image o np.ndarray; devuelve batch listo para modelo."""
    tf, preprocess_input, keras_image, _ = _ensure_model()

    if isinstance(img_source, np.ndarray):
        img_array = img_source
    else:
        # Puede ser ruta (str/Path) o PIL.Image
        img = keras_image.load_img(img_source, target_size=target_size) if not hasattr(img_source, 'size') else img_source.resize(target_size)
        img_array = keras_image.img_to_array(img)

    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


def extract_mixed_features(img_source):
    """
    Extrae el vector de features normalizado de la imagen usando el modelo cargado.
    El vector final se normaliza para que las distancias sean comparables entre modelos.
    """
    tf, _, _, feature_extractors = _ensure_model()
    img_pre = prepare_image(img_source)
    feats = []
    for name, extractor in feature_extractors.items():
        f = extractor.predict(img_pre, verbose=MOSTRAR_PROGRESO_TF).flatten()
        f = f / np.linalg.norm(f)
        feats.append(f)
    # Concatenar y normalizar el vector final para distancias comparables
    final_feats = np.concatenate(feats)
    final_feats = final_feats / np.linalg.norm(final_feats)
    return final_feats
