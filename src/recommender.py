import pandas as pd
from pathlib import Path
from typing import Union

from src.embeddings.clip_encoder import CLIPEncoder
from src.index.faiss_index import MovieIndex

INDEX_PATH = "data/processed/faiss.index"
INDEX_METADATA_PATH = "data/processed/index_metadata.csv"


class MovieRecommender:
    """Recomienda películas similares a partir de una portada o texto."""

    def __init__(
        self,
        index_path: str = INDEX_PATH,
        metadata_path: str = INDEX_METADATA_PATH,
    ):
        self.encoder = CLIPEncoder()
        self.index = MovieIndex()
        self.index.load(index_path, metadata_path)

    def recommend_from_image(
        self, image_path: Union[str, Path], top_k: int = 5
    ) -> pd.DataFrame:
        vector = self.encoder.encode_image(image_path)
        return self.index.search(vector, top_k)

    def recommend_from_text(self, text: str, top_k: int = 5) -> pd.DataFrame:
        vector = self.encoder.encode_text(text)
        return self.index.search(vector, top_k)
