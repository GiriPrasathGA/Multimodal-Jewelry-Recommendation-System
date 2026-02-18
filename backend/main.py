import os
import sys
import base64
import numpy as np
import pandas as pd
import faiss
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import io

# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.embedder import CLIPEngine
from utils.ocr import OCRManager
from utils.hybrid import HybridSearcher
from utils.reranker import Reranker

app = FastAPI(title="JewelUX API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for resources
engine = None
ocr = None
reranker = None
index_std = None
index_sbir = None
metadata = None
vectors = None
hybrid_searcher = None

@app.on_event("startup")
def load_resources():
    global engine, ocr, reranker, index_std, index_sbir, metadata, vectors, hybrid_searcher
    
    print("Eagerly loading all resources on startup...")
    
    # 1. Load Metadata (CSV)
    if os.path.exists("metadata/items.csv"):
        metadata = pd.read_csv("metadata/items.csv")
        print("✅ Metadata loaded.")
    else:
        print("⚠️ Warning: Metadata CSV not found")

    # 2. Load FAISS Indices
    if os.path.exists("embeddings/faiss_index.bin"):
        index_std = faiss.read_index("embeddings/faiss_index.bin")
        print("✅ Standard FAISS index loaded.")
    else:
        print("⚠️ Warning: Standard index not found")

    if os.path.exists("embeddings/faiss_sbir_index.bin"):
        index_sbir = faiss.read_index("embeddings/faiss_sbir_index.bin")
        print("✅ SBIR FAISS index loaded.")
    else:
        print("⚠️ Warning: SBIR index not found")

    # 3. Load Image Vectors (NPY)
    if os.path.exists("embeddings/image_vectors.npy"):
        vectors = np.load("embeddings/image_vectors.npy")
        print("✅ Image vectors loaded.")
    else:
        print("⚠️ Warning: image_vectors.npy not found")

    # 4. Load AI Engines/Modules
    try:
        print("⏳ Loading CLIP Engine...")
        engine = CLIPEngine()
        print("✅ CLIP Engine initialized.")
    except Exception as e:
        print(f"❌ Error loading CLIP Engine: {e}")

    try:
        print("⏳ Loading Reranker...")
        reranker = Reranker()
        print("✅ Reranker initialized.")
    except Exception as e:
        print(f"❌ Error loading Reranker: {e}")

    try:
        print("⏳ Loading OCR Manager...")
        ocr = OCRManager()
        ocr.load_model()
        print("✅ OCR Manager initialized.")
    except Exception as e:
        print(f"❌ Error loading OCR Manager: {e}")

    # 5. Initialize HybridSearcher
    if metadata is not None:
        hybrid_searcher = HybridSearcher(metadata, reranker=reranker)
        print("✅ HybridSearcher initialized.")

    import gc
    gc.collect()
    print("✨ Startup sequence complete. All resources ready.")

def get_engine():
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="CLIP Engine not initialized")
    return engine

def get_reranker():
    global reranker
    if reranker is None:
        # Fallback or warning if reranker is optional, but here we expect it pre-loaded
        print("Warning: Reranker requested but not pre-loaded")
    return reranker

def get_ocr():
    global ocr
    if ocr is None:
        raise HTTPException(status_code=503, detail="OCR Manager not initialized")
    return ocr

def get_indices():
    global index_std, index_sbir
    if index_std is None and index_sbir is None:
        raise HTTPException(status_code=503, detail="Search indices not initialized")
    return index_std, index_sbir

def get_base64_image(image_path):
    try:
        # Normalize path for Linux/Docker environment (replace Windows backslashes)
        image_path = image_path.replace("\\", "/")
        with open(image_path, "rb") as f: return base64.b64encode(f.read()).decode()
    except Exception as e: 
        print(f"Error loading image {image_path}: {e}")
        return None

# --- Pydantic Models ---
class SearchRequest(BaseModel):
    query: str
    top_k: int = 12

class SearchResponseItem(BaseModel):
    id: int
    score: float
    category: str
    description: str
    image_base64: Optional[str] = None
    path: str

# --- Helper function to format results ---
def format_results(ranked_results):
    response = []
    for res in ranked_results:
        item = res['metadata']
        # Find index in metadata to get path (though path is in item)
        # We need to make sure we return everything needed for the UI
        try:
            img_b64 = get_base64_image(item['path'])
            response.append({
                "id": int(res.get('id', 0)), # Placeholder if ID missing
                "score": float(res['score']),
                "category": item.get('category', 'Unknown'),
                "description": item.get('description', ''),
                "image_base64": img_b64,
                "path": item.get('path', '')
            })
        except Exception as e:
            print(f"Error formatting item {item}: {e}")
            continue
    return response

@app.post("/search/text", response_model=List[SearchResponseItem])
async def search_by_text(request: SearchRequest):
    engine = get_engine()
    index_std, _ = get_indices()
    hybrid_searcher = globals().get('hybrid_searcher')
    get_reranker() # Ensure reranker is ready for hybrid searcher

    if not hybrid_searcher or not engine or index_std is None:
        raise HTTPException(status_code=503, detail="Resources not initialized")

    # --- INTENT DETECTION ---
    # Simple rule-based detection for now. Can be upgraded to LLM later.
    query_lower = request.query.lower()
    detected_category = None
    
    # Map synonyms to canonical category names (based on items.csv)
    # Categories seen: 'necklace', 'ring' (from sample). 
    # Validating with simple mapping:
    category_map = {
        "ring": "ring",
        "rings": "ring",
        "band": "ring",
        "necklace": "necklace",
        "necklaces": "necklace",
        "chain": "necklace",
        "pendant": "necklace",
        "earring": "earring", # Assuming these exist or will exist
        "earrings": "earring",
        "bracelet": "bracelet", # Assuming these exist
        "bracelets": "bracelet",
        "bangle": "bracelet"
    }

    for keyword, category in category_map.items():
        if keyword in query_lower:
            detected_category = category
            break # Stop at first match (simplistic but works for "gold ring")

    query_vec = engine.get_text_embedding(request.query)
    q_vec = query_vec.reshape(1, -1).astype('float32')
    faiss.normalize_L2(q_vec)
    
    D, I = index_std.search(q_vec, k=50)
    ranked = hybrid_searcher.get_hybrid_scores(
        request.query, 
        q_vec[0], 
        I[0], 
        D[0], 
        top_k=request.top_k,
        category_filter=detected_category
        # filters removed
    )
    
    return format_results(ranked)

@app.post("/search/image", response_model=List[SearchResponseItem])
async def search_by_image(file: UploadFile = File(...), top_k: int = Form(12)):
    engine = get_engine()
    index_std, _ = get_indices()
    hybrid_searcher = globals().get('hybrid_searcher')
    get_reranker()

    if not hybrid_searcher or not engine or index_std is None:
        raise HTTPException(status_code=503, detail="Resources not initialized")
        
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    query_vec = engine.get_image_embedding(image)
    q_vec = query_vec.reshape(1, -1).astype('float32')
    faiss.normalize_L2(q_vec)
    
    # Image search typically doesn't use hybrid keywords, so query_text=""
    D, I = index_std.search(q_vec, k=50)
    ranked = hybrid_searcher.get_hybrid_scores("", q_vec[0], I[0], D[0], top_k=top_k)
    
    return format_results(ranked)

@app.post("/search/sketch", response_model=List[SearchResponseItem])
async def search_by_sketch(file: UploadFile = File(...), top_k: int = Form(12)):
    engine = get_engine()
    _, index_sbir = get_indices()
    hybrid_searcher = globals().get('hybrid_searcher')
    get_reranker()

    if not hybrid_searcher or not engine or index_sbir is None:
        raise HTTPException(status_code=503, detail="Resources not initialized")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    query_vec = engine.get_sketch_embedding(image)
    q_vec = query_vec.reshape(1, -1).astype('float32')
    faiss.normalize_L2(q_vec)
    
    # Use SBIR index for sketches
    D, I = index_sbir.search(q_vec, k=50)
    ranked = hybrid_searcher.get_hybrid_scores("", q_vec[0], I[0], D[0], top_k=top_k)
    
    return format_results(ranked)

@app.post("/search/handwriting", response_model=dict)
async def search_by_handwriting(file: UploadFile = File(...), top_k: int = Form(12), use_llm: bool = Form(True)):
    engine = get_engine()
    ocr = get_ocr()
    index_std, _ = get_indices()
    hybrid_searcher = globals().get('hybrid_searcher')
    get_reranker()

    if not hybrid_searcher or not engine or not ocr or index_std is None:
        raise HTTPException(status_code=503, detail="Resources not initialized")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    raw_ocr, cleaned_query, detected_category = ocr.extract_text(image, use_llm=use_llm)
    
    if not cleaned_query:
        # If no text found, return empty results with empty text fields
        return {"results": [], "raw_text": "", "refined_text": ""}

    query_vec = engine.get_text_embedding(cleaned_query)
    q_vec = query_vec.reshape(1, -1).astype('float32')
    faiss.normalize_L2(q_vec)
    
    D, I = index_std.search(q_vec, k=50)
    ranked = hybrid_searcher.get_hybrid_scores(
        cleaned_query, 
        q_vec[0], 
        I[0], 
        D[0], 
        top_k=top_k,
        category_filter=detected_category
    )
    
    formatted_results = format_results(ranked)
    
    return {
        "results": formatted_results,
        "raw_text": raw_ocr,
        "refined_text": cleaned_query if use_llm else "" 
    }

@app.get("/search/featured", response_model=List[SearchResponseItem])
def get_featured_items(count: int = -1):
    if not hybrid_searcher:
        raise HTTPException(status_code=503, detail="Resources not initialized")
    
    # Return all items if count is -1, else sample
    if metadata is not None and not metadata.empty:
        import random
        num_items = len(metadata)
        
        if count == -1 or count >= num_items:
            indices = list(range(num_items))
        else:
            indices = random.sample(range(num_items), count)
        
        featured = []
        for idx in indices:
            item = metadata.iloc[idx].to_dict()
            featured.append({
                'id': idx,
                'score': 1.0, 
                'metadata': item
            })
        return format_results(featured)
    return []

@app.get("/tags")
def get_tags():
    print("DEBUG: get_tags called")
    global metadata

    # Lazy load if missing
    if metadata is None:
        print("DEBUG: Metadata missing, attempting reload...")
        try:
            if os.path.exists("metadata/items.csv"):
                metadata = pd.read_csv("metadata/items.csv")
                print("DEBUG: Metadata reloaded successfully")
            else:
                print("DEBUG: metadata/items.csv not found")
        except Exception as e:
            print(f"DEBUG: Failed to reload metadata: {e}")

    if metadata is None:
        return {"tags": ["Gold Necklace", "Diamond Ring", "Silver Bracelet", "Pearl Earrings"]} # New Fallback
    
    try:
        # Get random descriptions
        if 'description' in metadata.columns:
            # Drop empty
            valid_descs = metadata['description'].dropna().tolist()
            if not valid_descs:
                 return {"tags": ["Luxury", "Elegance", "Vintage", "Modern"]}
            
            # Sample 5 random items
            
            # Sample 5 random items
            import random
            samples = random.sample(valid_descs, min(len(valid_descs), 5))
            
            cleaned_tags = []
            for desc in samples:
                # Clean up text
                text = desc.strip()
                # Remove starting articles
                for prefix in ["A ", "An ", "The ", "a ", "an ", "the "]:
                    if text.startswith(prefix):
                        text = text[len(prefix):]
                        break
                
                # Capitalize first letter
                if text:
                    text = text[0].upper() + text[1:]
                
                # Truncate if too long (ellipses)
                if len(text) > 45:
                    text = text[:42] + "..."
                    
                cleaned_tags.append(text)
                
            return {"tags": cleaned_tags}
        return {"tags": []}
    except Exception as e:
        print(f"Error fetching tags: {e}")
        return {"tags": []}

class SimilarSearchRequest(BaseModel):
    id: int
    top_k: int = 12

@app.post("/search/similar", response_model=List[SearchResponseItem])
async def search_similar(request: SimilarSearchRequest):
    # 1. Get vector for the item
    # vectors is a global numpy array
    if vectors is None:
         raise HTTPException(status_code=503, detail="Vectors not pre-loaded")
    
    if request.id >= len(vectors):
        raise HTTPException(status_code=404, detail="Item ID not found")
        
    target_vec = vectors[request.id].reshape(1, -1).astype('float32')
    faiss.normalize_L2(target_vec)
    
    # 2. Search FAISS
    # We ask for top_k + 1 because the item itself will be the first result (dist=0 or 1)
    D, I = index_std.search(target_vec, k=request.top_k + 1)
    
    # 3. Use get_hybrid_scores mostly for formatting & potential filtering if we add it later
    # We pass query_text="" so it relies purely on visual similarity for now.
    
    found_indices = I[0]
    found_scores = D[0]
    
    filtered_indices = []
    filtered_scores = []
    
    for idx, score in zip(found_indices, found_scores):
        if idx != request.id:
            filtered_indices.append(idx)
            filtered_scores.append(score)
    
    # Slice to top_k
    filtered_indices = filtered_indices[:request.top_k]
    filtered_scores = filtered_scores[:request.top_k]
    
    ranked = hybrid_searcher.get_hybrid_scores(
        "", 
        target_vec[0], 
        filtered_indices, 
        filtered_scores, 
        top_k=request.top_k
    )
    
    return format_results(ranked)

@app.get("/health")
def health_check():
    return {"status": "ok"}
