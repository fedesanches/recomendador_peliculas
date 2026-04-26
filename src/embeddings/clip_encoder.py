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

    def _image_features(self, pixel_values) -> torch.Tensor:
        out = self.model.vision_model(pixel_values=pixel_values)
        feats = self.model.visual_projection(out.pooler_output)
        return feats / feats.norm(dim=-1, keepdim=True)

    def _text_features(self, inputs) -> torch.Tensor:
        out = self.model.text_model(**inputs)
        feats = self.model.text_projection(out.pooler_output)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_image(self, image_path: Union[str, Path]) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        return self._image_features(inputs["pixel_values"]).cpu().numpy()[0]

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(
            text=[text], return_tensors="pt", truncation=True, max_length=77
        ).to(self.device)
        return self._text_features(inputs).cpu().numpy()[0]

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
            all_features.append(self._image_features(inputs["pixel_values"]).cpu().numpy())
        return np.vstack(all_features)
