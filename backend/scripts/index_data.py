import os
import sys
import glob
import numpy as np
import faiss
import pandas as pd
import cv2
from PIL import Image

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.embedder import CLIPEngine

def generate_edge_map(image_path):
    """ Converts a real photo into a 'synthetic sketch' using Canny edges. """
    img = cv2.imread(image_path, 0)
    # Canny detects the outlines of the jewelry
    edges = cv2.Canny(img, 100, 200)
    # Invert to make it black lines on white background (like a drawing)
    return Image.fromarray(cv2.bitwise_not(edges)).convert("RGB")

def index_dataset(data_dir="data/Jewellery_Data/", excel_path="data/Jewellery_Data/jewelry_descriptions.xlsx"):
    engine = CLIPEngine(model_name="ViT-B/32")
    
    # Load Excel
    if os.path.exists(excel_path):
        df_desc = pd.read_excel(excel_path)
        df_desc.columns = df_desc.columns.str.strip()
        df_desc['Image'] = df_desc['Image'].apply(lambda x: os.path.basename(str(x)))
        print("✅ Excel loaded.")
    else:
        print(f"❌ {excel_path} not found!")
        return

    image_paths = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    
    photo_embs = []
    sbir_embs = []
    metadata = []

    print(f"🚀 Processing {len(image_paths)} images for Standard + SBIR Indexing...")

    for path in image_paths:
        try:
            fname = os.path.basename(path)
            match = df_desc[df_desc['Image'] == fname]
            desc = str(match['Description'].values[0]) if not match.empty else "Jewelry"
            cat = str(match['Category'].values[0]) if not match.empty else "Item"

            # 1. Standard Photo Embedding
            photo_embs.append(engine.get_image_embedding(path))
            
            # 2. SBIR (Edge Map) Embedding
            edge_img = generate_edge_map(path)
            sbir_embs.append(engine.get_image_embedding(edge_img))

            metadata.append({"path": path, "category": cat, "description": desc})
        except Exception as e:
            print(f"Error on {path}: {e}")
            continue

    # Create Standard Index
    photo_arr = np.array(photo_embs).astype('float32')
    faiss.normalize_L2(photo_arr)
    photo_index = faiss.IndexFlatIP(512)
    photo_index.add(photo_arr)
    
    # Create SBIR Index
    sbir_arr = np.array(sbir_embs).astype('float32')
    faiss.normalize_L2(sbir_arr)
    sbir_index = faiss.IndexFlatIP(512)
    sbir_index.add(sbir_arr)

    # Save Everything
    os.makedirs("embeddings", exist_ok=True)
    os.makedirs("metadata", exist_ok=True)
    
    faiss.write_index(photo_index, "embeddings/faiss_index.bin")
    faiss.write_index(sbir_index, "embeddings/faiss_sbir_index.bin") # <--- CRITICAL FIX
    
    pd.DataFrame(metadata).to_csv("metadata/items.csv", index=False)
    np.save("embeddings/image_vectors.npy", photo_arr)
    
    print(f"✅ DONE! Created 'faiss_index.bin' and 'faiss_sbir_index.bin'")

if __name__ == "__main__":
    index_dataset()