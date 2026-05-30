---
title: Recomendador de Películas por Portada
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Recomendador de Películas por Portada

Dado la portada de una película, encuentra las **K películas más similares** usando embeddings de CLIP y búsqueda por similitud coseno en FAISS.

## Arquitectura

La arquitectura tiene dos pipelines bien separados:

**Pipeline offline**

1. Toma un dataset con pósters + metadatos (título, sinopsis, género). Se usa MovieLens + TMDB.
2. Pasa cada póster por el image encoder de CLIP → obtiene un vector de 512 dimensiones por película.
3. Construye un índice FAISS con los vectores normalizados.
4. El índice y los metadatos se suben a HuggingFace Hub para distribución.

**Pipeline online**

1. El usuario sube una imagen de portada.
2. Se pasa por el mismo CLIP image encoder → vector de 512 dims.
3. Busca los K vecinos más cercanos por similitud coseno en FAISS.
4. Devuelve título + sinopsis + score de similitud + póster desde TMDB.

```
Pipeline offline (una vez)
  Dataset (MovieLens + TMDB)
    └── pósters ──► CLIP Image Encoder ──► FAISS Index ──► HuggingFace Hub

Pipeline online (por consulta)
  Imagen query ──► CLIP Encode (512 dims) ──► Top-K cosine similarity ──► Películas recomendadas
```

## Estructura

```
├── api.py                        # REST API con FastAPI
├── app.py                        # Entry point para HuggingFace Spaces (Gradio)
├── app/
│   └── app.py                    # Demo Gradio (local + HF Spaces)
├── src/
│   ├── data/
│   │   └── download_posters.py   # Descarga pósters y metadatos desde TMDB
│   ├── embeddings/
│   │   └── clip_encoder.py       # Encoda imágenes/texto con CLIP
│   ├── index/
│   │   └── faiss_index.py        # Construye y consulta el índice FAISS
│   ├── metrics/
│   │   └── metric_service.py     # Cálculo de Precision@K, NDCG@K, coherencia de género
│   └── recommender.py            # Clase principal MovieRecommender
├── scripts/
│   └── build_index.py            # Pipeline offline: encode → index
├── notebooks/
│   ├── build_index_colab.ipynb   # Pipeline offline en Google Colab
│   └── notebook_metricas.ipynb   # Evaluación de métricas
├── data/
│   ├── raw/                      # movies.csv y links.csv de MovieLens (no versionados)
│   └── processed/                # faiss.index, index_metadata.csv (descargados de HF Hub)
├── .env.example
└── requirements.txt
```

## Setup

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar API key de TMDB

```bash
cp .env.example .env
# Editar .env y pegar tu TMDB_API_KEY
# Obtener en: https://www.themoviedb.org/settings/api
```

### 3. Construir el índice (Google Colab)

Subir `movies.csv` y `links.csv` a Google Drive y ejecutar `notebooks/build_index_colab.ipynb`.
El notebook descarga los pósters desde TMDB, construye el índice FAISS y genera los archivos para descargar.

Colocar los archivos descargados en `data/processed/`:
- `faiss.index`
- `index_metadata.csv`

### 4. Lanzar la demo Gradio

```bash
python app/app.py
```

El app descarga el índice automáticamente desde HuggingFace Hub si no existe en `data/processed/`.

Por defecto usa el recomendador embebido. Para que consuma la REST API en lugar del recomendador local, configurar en `.env`:

```
RECOMMENDER_MODE=api
RECOMMENDER_API_URL=http://localhost:8000
```

### 5. Lanzar la REST API

```bash
python api.py
```

Endpoints disponibles en `http://localhost:8000`:

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/health` | Estado del servicio |
| POST | `/recommend/image` | Recomienda por imagen de póster |
| POST | `/recommend/text` | Recomienda por texto/sinopsis |

Documentación interactiva: `http://localhost:8000/docs`

## Métricas de evaluación

Calculadas sobre 500 muestras aleatorias con umbral Jaccard = 0.5:

| Métrica | Descripción |
|--------|-------------|
| Precision@K | Fracción de las K recomendaciones que superan el umbral de similitud de género |
| NDCG@K | Calidad del ranking — recomendaciones relevantes en posiciones altas valen más |
| Coherencia de género | Jaccard promedio entre géneros de la query y las recomendaciones |
| Aciertos | Total de recomendaciones individuales que superan el umbral |

## Modelo

- **CLIP**: `openai/clip-vit-base-patch32` — vectores de 512 dimensiones
- **Vector DB**: FAISS `IndexFlatIP` (inner product = cosine similarity sobre vectores normalizados)
- **Datos**: MovieLens Latest + TMDB API

## Despliegue

El app se despliega automáticamente en HuggingFace Spaces via GitHub Actions al hacer push a `main` o `prepro`.
