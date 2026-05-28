"""
Pipeline offline: codifica todos los pósters con DINOv2 y construye el índice FAISS.

Uso:
    python scripts/build_index_dinov2.py
"""

import pandas as pd
from pathlib import Path

from src.embeddings.dinov2_encoder import Dinov2Encoder
from src.index.faiss_index import MovieIndex

METADATA_PATH       = "data/processed/metadata.csv"
POSTER_URL_SOURCE   = "data/processed/index_metadata.csv"
INDEX_PATH          = "data/processed/faiss_dinov2.index"
INDEX_METADATA_PATH = "data/processed/index_metadata_dinov2.csv"


def build_index():
    metadata = pd.read_csv(METADATA_PATH)

    # Filtrar filas sin póster descargado
    metadata = metadata[
        metadata["poster_path"].notna() & (metadata["poster_path"] != "")
    ].reset_index(drop=True)

    # Agregar tmdb_poster_url
    url_source = pd.read_csv(POSTER_URL_SOURCE)[["tmdbId", "tmdb_poster_url"]]
    metadata = metadata.merge(url_source, on="tmdbId", how="left")

    print(f"Películas con póster disponible: {len(metadata)}")

    encoder = Dinov2Encoder()
    image_paths = metadata["poster_path"].tolist()

    print("Codificando pósters con DINOv2...")
    embeddings = encoder.encode_images_batch(image_paths)

    index = MovieIndex(dim=embeddings.shape[1])
    index.build(embeddings, metadata)

    Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    index.save(INDEX_PATH, INDEX_METADATA_PATH)
    print(f"Índice guardado en {INDEX_PATH} ({len(metadata)} películas indexadas).")


if __name__ == "__main__":
    build_index()