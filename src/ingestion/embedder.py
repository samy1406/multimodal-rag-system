import clip
from PIL import Image
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# model, preprocess = clip.load("ViT-B/32", device=device)
model = None
preprocess = None

def get_model():
    global model, preprocess
    if model is None:
        model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess
# uses the sentence transformer
def embed_text(texts: list[str]) -> np.ndarray:
    tokens = clip.tokenize(texts, truncate=True).to(device)
    text_feature = model.encode_text(tokens)
    text_feature /= text_feature.norm(dim=-1, keepdim=True)
    text_feature = text_feature.detach().cpu().numpy()
    return text_feature

# uses CLIP embedding
def embed_image(pil_images: list) -> np.ndarray:
    image_features = []
    for i in range(len(pil_images)):
        image = preprocess(pil_images[i]).unsqueeze(0).to(device)
        image_feature = model.encode_image(image)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        image_feature = image_feature.detach().cpu().numpy()
        image_features.append(image_feature)
    return np.vstack(image_features)