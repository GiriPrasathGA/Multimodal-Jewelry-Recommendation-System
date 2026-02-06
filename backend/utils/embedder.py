import torch
import clip
from PIL import Image
import cv2
import numpy as np

class CLIPEngine:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def get_image_embedding(self, image_path_or_pill):
        if isinstance(image_path_or_pill, str):
            image = self.preprocess(Image.open(image_path_or_pill)).unsqueeze(0).to(self.device)
        else:
            image = self.preprocess(image_path_or_pill).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode_image(image)
        return embedding.cpu().numpy().flatten()

    def get_sketch_embedding(self, pil_image):
        """
        SBIR Logic: Cleans the user's hand-drawing to match indexed edges.
        """
        # Convert to OpenCV format
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
        # Remove paper noise (Thresholding)
        _, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
        # Normalize line thickness (Dilation)
        kernel = np.ones((2,2), np.uint8)
        processed = cv2.dilate(thresh, kernel, iterations=1)
        # Convert back to white background/black lines for CLIP
        final_img = Image.fromarray(cv2.bitwise_not(processed)).convert("RGB")
        return self.get_image_embedding(final_img)

    def get_text_embedding(self, text):
        text_tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_text(text_tokens)
        return embedding.cpu().numpy().flatten()