---
title: Recomendador de Películas por Portada
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Recomendador de Películas por Portada

Dado la portada de una película, encuentra las **K películas más similares** usando embeddings de CLIP y búsqueda por similitud coseno en FAISS.

## Arquitectura
La arquitectura tiene dos pipelines bien separados:

**Pipeline offline**

1. Toma un dataset con pósters + metadatos (título, sinopsis, género). Se usa MovieLens + TMDB.
2. Pasa cada póster por el image encoder de CLIP → obtiene un vector de 512 dimensiones por película.
3. También se encoda la sinopsis con el text encoder de CLIP.
4. Todo eso va a una vector DB como FAISS.

**Pipeline online**

1. El usuario sube una imagen de portada.
2. Se pasa por el mismo CLIP image encoder → vector.
3. Busca los K vecinos más cercanos por similitud coseno en la DB.
4. Devuelve título + sinopsis + score de similitud.

```
Pipeline offline (una vez)
  Dataset (MovieLens + TMDB)
    ├── pósters  ──► CLIP Image Encoder ──►
    │                                       Vector DB (FAISS)
    └── sinopsis ──► CLIP Text Encoder  ──►

Pipeline online (por consulta)
  Imagen query ──► CLIP Encode (512 dims) ──► Top-K cosine similarity ──► Películas recomendadas
```

## Estructura

```
├── data/
│   ├── raw/          # movies.csv y links.csv de MovieLens (no versionados)
│   ├── posters/      # Pósters descargados de TMDB (no versionados)
│   └── processed/    # metadata.csv, faiss.index, index_metadata.csv
├── src/
│   ├── data/
│   │   └── download_posters.py   # Descarga pósters y metadatos desde TMDB
│   ├── embeddings/
│   │   └── clip_encoder.py       # Encoda imágenes/texto con CLIP
│   ├── index/
│   │   └── faiss_index.py        # Construye y consulta el índice FAISS
│   └── recommender.py            # Clase principal MovieRecommender
├── scripts/
│   └── build_index.py            # Pipeline offline: encode → index
├── app/
│   └── app.py                    # Demo Gradio
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

### 3. Descargar MovieLens

Descargar [MovieLens Latest](https://grouplens.org/datasets/movielens/) y colocar `movies.csv` y `links.csv` en `data/raw/`.

### 4. Pipeline offline

```bash
# Descarga pósters y genera data/processed/metadata.csv
python src/data/download_posters.py

# Codifica pósters y construye el índice FAISS
python scripts/build_index.py
```

### 5. Lanzar demo

```bash
python app/app.py
```

## Modelo

- **CLIP**: `openai/clip-vit-base-patch32` — vectores de 512 dimensiones
- **Vector DB**: FAISS `IndexFlatIP` (inner product = cosine similarity sobre vectores normalizados)
