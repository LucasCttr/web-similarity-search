# Proyecto: Image Search (Frontend Angular + Backend FastAPI)
Este proyecto implementa un sistema de **búsqueda por similitud** utilizando representaciones visuales de páginas web (landing pages en formato PNG).
La aplicación permite subir una imagen de consulta y devuelve los 10 objetos más similares dentro de un radio de búsqueda, junto con la distancia calculada. Se utiliza **ResNet50** como extractor de características y se gestionan más de 1.200 imágenes de referencia, con un lote de 50 consultas para evaluar precisión en los resultados.

**Demo**

Aquí hay una demostración en video del funcionamiento de la aplicación. Puedes reproducirlo directamente desde este repositorio:

<video controls width="640">
	<source src="assets/demo.mp4" type="video/mp4">
	Tu navegador no soporta la reproducción de video.
</video>

También puedes abrir el archivo directamente: [assets/demo.mp4](assets/demo.mp4)


Estructura:
- frontend/: proyecto Angular básico (componentes para subir y buscar imagenes)
- backend/: API en FastAPI con endpoints `/upload` y `/search` 

Instrucciones rápidas:

Levantar aplicación con Docker:

```
docker-compose up --build
```

Navegar a http://localhost:4200 para el frontend.

Levantar aplicación de forma manual:

Backend:
1. Abrir terminal en `backend`
2. Crear y activar un entorno virtual (opcional)

Windows PowerShell:
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:
1. Instalar Node.js y Angular CLI
2. Abrir terminal en `frontend`
```
npm install


