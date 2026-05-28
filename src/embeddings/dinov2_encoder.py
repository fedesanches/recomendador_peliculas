import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Union
from transformers import AutoImageProcessor, AutoModel

MODEL_ID = "facebook/dinov2-large"


class Dinov2Encoder:
    """
    Encoder de imágenes con DINOv2 (1024 dims, normalizados para cosine similarity).
    Solo soporta búsqueda imagen→imagen (no tiene encoder de texto).
    """

    def __init__(self, model_id: str = MODEL_ID, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model.eval()

    def _image_features(self, pixel_values) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        # DINOv2: usar el CLS token como representación global de la imagen
        feats = outputs.last_hidden_state[:, 0, :]
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_image(self, image_path: Union[str, Path]) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        return self._image_features(inputs["pixel_values"]).cpu().numpy()[0]

    @torch.no_grad()
    def encode_images_batch(
        self, image_paths: List[Union[str, Path]], batch_size: int = 32
    ) -> np.ndarray:
        all_features = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            all_features.append(
                self._image_features(inputs["pixel_values"]).cpu().numpy()
            )
        return np.vstack(all_features)