import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Union
from transformers import AutoProcessor, AutoModel

MODEL_ID = "google/siglip-base-patch16-224"

class SiglipEncoder:
    """
    Encoder para imágenes y texto con SigLIP (768 dims, normalizados para cosine similarity).
    """

    def __init__(self, model_id: str = MODEL_ID, device: str = None):
        """
        Inicializa el encoder cargando el modelo y el procesador de Hugging Face.
         - model_id: Identificador del modelo en Hugging Face.
         - device: Dispositivo para inferencia ("cuda" o "cpu"). Si no se especifica, se detecta automáticamente.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model.eval()

    def _image_features(self, pixel_values) -> torch.Tensor:
        """
        Extrae características de la imagen y las normaliza.
         - pixel_values: Tensor de imágenes procesadas.
         - Retorna: Tensor de características normalizadas.
         """
        feats = self.model.get_image_features(pixel_values=pixel_values)
        return feats / feats.norm(dim=-1, keepdim=True)

    def _text_features(self, inputs) -> torch.Tensor:
        """
        Extrae características del texto y las normaliza.
         - inputs: Diccionario de tensores de entrada para el modelo.
         - Retorna: Tensor de características normalizadas.
        """
        feats = self.model.get_text_features(**inputs)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Codifica una imagen y retorna su vector de características normalizado.
         - image_path: Ruta de la imagen a codificar.
         - Retorna: Vector de características normalizado.
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        return self._image_features(inputs["pixel_values"]).cpu().numpy()[0]

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """
        Codifica un texto y retorna su vector de características normalizado.
         - text: Texto a codificar.
         - Retorna: Vector de características normalizado.
        """
        inputs = self.processor(
            text=[text], return_tensors="pt", truncation=True, max_length=64
        ).to(self.device)
        return self._text_features(inputs).cpu().numpy()[0]

    @torch.no_grad()
    def encode_images_batch(
        self, image_paths: List[Union[str, Path]], batch_size: int = 32
    ) -> np.ndarray:
        """Codifica un lote de imágenes y retorna una matriz de características normalizadas.
         - image_paths: Lista de rutas de las imágenes a codificar.
         - batch_size: Tamaño del lote para la codificación.
         - Retorna: Matriz de características normalizadas (num_imágenes x 768).
        """
        all_features = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch]
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(
                self.device
            )
            all_features.append(
                self._image_features(inputs["pixel_values"]).cpu().numpy()
            )
        return np.vstack(all_features)