# models/imagecaption.py
import io
import requests
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageCaptioner:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name, use_safetensors=True).to(self.device)

    def generate(self, image_input, max_new_tokens: int = 10, num_beams: int = 5) -> str:
        """
        image_input: PIL.Image or local path string or remote URL string
        returns: generated caption (str)
        """
        if isinstance(image_input, str):
            if image_input.startswith(("http://", "https://")):
                resp = requests.get(image_input, timeout=10)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            else:
                img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")

        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            early_stopping=True
        )
        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        return caption
