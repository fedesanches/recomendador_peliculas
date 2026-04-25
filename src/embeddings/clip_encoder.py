import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Union
from transformers import CLIPProcessor, CLIPModel

MODEL_ID = "openai/clip-vit-base-patch32"


class CLIPEncoder:
    """Encoda imágenes y texto con CLIP (512 dims, normalizados para cosine similarity)."""

    def __init__(self, model_id: str = MODEL_ID, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, image_path: Union[str, Path]) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0]

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(
            text=[text], return_tensors="pt", truncation=True, max_length=77
        ).to(self.device)
        features = self.model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0]

    @torch.no_grad()
    def encode_images_batch(
        self, image_paths: List[Union[str, Path]], batch_size: int = 32
    ) -> np.ndarray:
        all_features = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch]
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(
                self.device
            )
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())
        return np.vstack(all_features)
