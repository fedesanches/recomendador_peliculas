"""Pipeline offline: codifica todos los pósters con CLIP y construye el índice FAISS.

Uso:
    python scripts/build_index.py
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.embeddings.clip_encoder import CLIPEncoder
from src.index.faiss_index import MovieIndex

METADATA_PATH = "data/processed/metadata.csv"
INDEX_PATH = "data/processed/faiss.index"
INDEX_METADATA_PATH = "data/processed/index_metadata.csv"


def build_index():
    metadata = pd.read_csv(METADATA_PATH)

    # Filtrar filas sin póster descargado
    metadata = metadata[
        metadata["poster_path"].notna() & (metadata["poster_path"] != "")
    ].reset_index(drop=True)

    print(f"Películas con póster disponible: {len(metadata)}")

    encoder = CLIPEncoder()
    image_paths = metadata["poster_path"].tolist()

    print("Codificando pósters con CLIP...")
    embeddings = encoder.encode_images_batch(image_paths)

    index = MovieIndex(dim=embeddings.shape[1])
    index.build(embeddings, metadata)

    Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    index.save(INDEX_PATH, INDEX_METADATA_PATH)
    print(f"Índice guardado en {INDEX_PATH} ({len(metadata)} películas indexadas).")


if __name__ == "__main__":
    build_index()
