import pandas as pd
from pathlib import Path
from typing import Union

from src.embeddings.clip_encoder import CLIPEncoder
from src.index.faiss_index import MovieIndex

INDEX_PATH          = "data/processed/faiss.index"
INDEX_METADATA_PATH = "data/processed/index_metadata.csv"
COMBINED_INDEX_PATH          = "data/processed/faiss_combined.index"
COMBINED_INDEX_METADATA_PATH = "data/processed/index_metadata_combined.csv"
SIGLIP_INDEX_PATH          = "data/processed/faiss_siglip.index"
SIGLIP_INDEX_METADATA_PATH = "data/processed/index_metadata_siglip.csv"
DINOV2_INDEX_PATH          = "data/processed/faiss_dinov2.index"
DINOV2_INDEX_METADATA_PATH = "data/processed/index_metadata_dinov2.csv"


class MovieRecommender:
    def __init__(
        self,
        index_path: str = INDEX_PATH,
        metadata_path: str = INDEX_METADATA_PATH,
        combined_index_path: str = COMBINED_INDEX_PATH,
        combined_metadata_path: str = COMBINED_INDEX_METADATA_PATH,
        siglip_index_path: str = SIGLIP_INDEX_PATH,
        siglip_metadata_path: str = SIGLIP_INDEX_METADATA_PATH,
        dinov2_index_path: str = DINOV2_INDEX_PATH,
        dinov2_metadata_path: str = DINOV2_INDEX_METADATA_PATH,
    ):
        self.encoder = CLIPEncoder()
        self.index = MovieIndex()
        self.index.load(index_path, metadata_path)
        self.index_combined = MovieIndex()
        self.index_combined.load(combined_index_path, combined_metadata_path)

        self.siglip_encoder = None
        self.siglip_index = None
        if Path(siglip_index_path).exists() and Path(siglip_metadata_path).exists():
            from src.embeddings.siglip_encoder import SiglipEncoder
            self.siglip_encoder = SiglipEncoder()
            self.siglip_index = MovieIndex(dim=768)
            self.siglip_index.load(siglip_index_path, siglip_metadata_path)

        self.dinov2_encoder = None
        self.dinov2_index = None
        if Path(dinov2_index_path).exists() and Path(dinov2_metadata_path).exists():
            from src.embeddings.dinov2_encoder import Dinov2Encoder
            self.dinov2_encoder = Dinov2Encoder()
            self.dinov2_index = MovieIndex(dim=1024)
            self.dinov2_index.load(dinov2_index_path, dinov2_metadata_path)

    def _get_clip_index(self, combined: bool) -> MovieIndex:
        return self.index_combined if combined else self.index

    def recommend_from_image(
        self, image_path: Union[str, Path], top_k: int = 5, combined: bool = False, model: str = "clip"
    ) -> pd.DataFrame:
        if model == "siglip":
            if self.siglip_encoder is None:
                raise RuntimeError("Índice SigLIP no disponible.")
            vector = self.siglip_encoder.encode_image(image_path)
            return self.siglip_index.search(vector, top_k)
        if model == "dinov2":
            if self.dinov2_encoder is None:
                raise RuntimeError("Índice DINOv2 no disponible.")
            vector = self.dinov2_encoder.encode_image(image_path)
            return self.dinov2_index.search(vector, top_k)
        vector = self.encoder.encode_image(image_path)
        return self._get_clip_index(combined).search(vector, top_k)

    def recommend_from_text(
        self, text: str, top_k: int = 5, combined: bool = False, model: str = "clip"
    ) -> pd.DataFrame:
        if model == "dinov2":
            raise RuntimeError("DINOv2 no soporta búsqueda por texto.")
        if model == "siglip":
            if self.siglip_encoder is None:
                raise RuntimeError("Índice SigLIP no disponible.")
            vector = self.siglip_encoder.encode_text(text)
            return self.siglip_index.search(vector, top_k)
        vector = self.encoder.encode_text(text)
        return self._get_clip_index(combined).search(vector, top_k)
