import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from openai import OpenAI
from dotenv import load_dotenv

# Path-aware loading: This finds the .env in the root folder automatically
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

class OCRManager:
    def __init__(self):
        # 1. Load TrOCR Model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "microsoft/trocr-small-handwritten"
        self.processor = TrOCRProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id).to(self.device)
        
        # 2. Initialize LLM Client (Replace with your credentials)
        # os.getenv pulls the values from your .env file
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")

        if not api_key:
            print("❌ WARNING: LLM_API_KEY not found in .env file")

        self.llm_client = OpenAI(
            api_key=api_key, 
            base_url=base_url
        )

    def preprocess_image(self, pil_image):
        """Advanced OpenCV preprocessing for noisy handwriting."""
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # CLAHE for contrast, Denoising, and Adaptive Thresholding
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, h=20)
        th = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 12)
        
        return Image.fromarray(cv2.cvtColor(th, cv2.COLOR_GRAY2RGB))

    def clean_query_with_llm(self, raw_text):
        """Uses LLM to correct OCR errors and standardize jewelry terms."""
        prompt = f"""
        You are a jewelry search assistant. Your task is to clean up noisy OCR text extracted from a handwritten note.
        
        RULES:
        1. Correct spelling errors (e.g., 'diomond' -> 'diamond', 'neklace' -> 'necklace').
        2. Standardize terms: Use 'ring', 'necklace', 'earring', 'bracelet', 'bangle'.
        3. Standardize stones: Use 'diamond', 'ruby', 'emerald', 'sapphire', 'white stone'.
        4. Remove noise (special characters, random letters).
        5. Output ONLY a short, clean search query string.

        Raw OCR Text: "{raw_text}"
        Cleaned Query:"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4.1-nano", # or your specific model
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=20
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return raw_text # Fallback to raw text if API fails

    def extract_text(self, pil_image):
        try:
            # Step A: Preprocess
            processed_pil = self.preprocess_image(pil_image)
            
            # Step B: TrOCR Inference
            pixel_values = self.processor(processed_pil, return_tensors="pt").pixel_values.to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            raw_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Step C: LLM Refinement
            if raw_text.strip():
                cleaned_text = self.clean_query_with_llm(raw_text)
                return raw_text, cleaned_text
            return "", ""
        except Exception as e:
            print(f"OCR Error: {e}")
            return "", ""