import faiss
import numpy as np
import pandas as pd


class MovieIndex:
    """Índice FAISS sobre vectores CLIP normalizados (cosine similarity via inner product)."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # IP = cosine sim para vectores normalizados
        self.metadata: pd.DataFrame = None

    def build(self, embeddings: np.ndarray, metadata: pd.DataFrame) -> None:
        assert embeddings.shape[1] == self.dim, (
            f"Dimensión de embeddings ({embeddings.shape[1]}) no coincide con dim={self.dim}"
        )
        self.index.add(embeddings.astype(np.float32))
        self.metadata = metadata.reset_index(drop=True)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> pd.DataFrame:
        query = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, top_k)
        results = self.metadata.iloc[indices[0]].copy()
        results["score"] = scores[0]
        return results.reset_index(drop=True)

    def save(self, index_path: str, metadata_path: str) -> None:
        faiss.write_index(self.index, index_path)
        self.metadata.to_csv(metadata_path, index=False)

    def load(self, index_path: str, metadata_path: str) -> None:
        self.index = faiss.read_index(index_path)
        self.metadata = pd.read_csv(metadata_path)
