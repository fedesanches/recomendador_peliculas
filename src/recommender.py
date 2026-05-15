import pandas as pd
from pathlib import Path
from typing import Union

from src.embeddings.clip_encoder import CLIPEncoder
from src.index.faiss_index import MovieIndex

INDEX_PATH          = "data/processed/faiss.index"
INDEX_METADATA_PATH = "data/processed/index_metadata.csv"
COMBINED_INDEX_PATH          = "data/processed/faiss_combined.index"
COMBINED_INDEX_METADATA_PATH = "data/processed/index_metadata_combined.csv"


class MovieRecommender:
    def __init__(
        self,
        index_path: str = INDEX_PATH,
        metadata_path: str = INDEX_METADATA_PATH,
        combined_index_path: str = COMBINED_INDEX_PATH,
        combined_metadata_path: str = COMBINED_INDEX_METADATA_PATH,
    ):
        self.encoder = CLIPEncoder()
        self.index = MovieIndex()
        self.index.load(index_path, metadata_path)
        self.index_combined = MovieIndex()
        self.index_combined.load(combined_index_path, combined_metadata_path)

    def _get_index(self, combined: bool) -> MovieIndex:
        return self.index_combined if combined else self.index

    def recommend_from_image(
        self, image_path: Union[str, Path], top_k: int = 5, combined: bool = False
    ) -> pd.DataFrame:
        vector = self.encoder.encode_image(image_path)
        return self._get_index(combined).search(vector, top_k)

    def recommend_from_text(self, text: str, top_k: int = 5, combined: bool = False) -> pd.DataFrame:
        vector = self.encoder.encode_text(text)
        return self._get_index(combined).search(vector, top_k)
