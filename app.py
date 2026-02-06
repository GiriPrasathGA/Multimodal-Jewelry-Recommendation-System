import streamlit as st
from PIL import Image
import os, faiss, base64
import pandas as pd
import numpy as np
from utils.embedder import CLIPEngine
from utils.ocr import OCRManager
from utils.hybrid import HybridSearcher

# --- UI CONFIGURATION ---
st.set_page_config(layout="wide", page_title="JewelUX Dashboard", page_icon="💎")

# --- SESSION STATE ---
if 'use_sbir' not in st.session_state: st.session_state.use_sbir = False
if 'query_emb' not in st.session_state: st.session_state.query_emb = None
if 'query_text' not in st.session_state: st.session_state.query_text = ""
if 'last_menu' not in st.session_state: st.session_state.last_menu = "🏠 Home"

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as f: return base64.b64encode(f.read()).decode()
    except: return None

@st.cache_resource
def load_resources():
    return CLIPEngine(), OCRManager(), \
           faiss.read_index("embeddings/faiss_index.bin"), \
           faiss.read_index("embeddings/faiss_sbir_index.bin"), \
           pd.read_csv("metadata/items.csv"), \
           np.load("embeddings/image_vectors.npy")

engine, ocr, index_std, index_sbir, metadata, vectors = load_resources()
hybrid_searcher = HybridSearcher(metadata)

# --- LUXURY CSS OVERHAUL ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;600&display=swap');

    .stApp { background-color: #0E1117; color: #E0E0E0; }
    
    /* Animation for results */
    @keyframes fadeIn { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }
    
    .result-block { 
        position: relative; 
        animation: fadeIn 0.6s ease-out; 
        margin-bottom: 20px;
    }

    /* Luxury Product Card */
    .item-card {
        background: #161B22;
        border-radius: 8px;
        border: 1px solid #2D333B;
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
        overflow: hidden;
        position: relative;
    }
    
    .item-card:hover {
        border-color: #D4AF37;
        transform: translateY(-8px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.6);
    }

    /* Floating Gold Badge */
    .badge {
        position: absolute;
        top: 12px;
        right: 12px;
        background: rgba(14, 17, 23, 0.9);
        color: #D4AF37;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: bold;
        border: 1px solid #D4AF37;
        z-index: 5;
    }

    .card-img {
        width: 100%;
        height: 300px;
        object-fit: cover;
        transition: transform 0.8s ease;
    }
    
    .item-card:hover .card-img { transform: scale(1.05); }

    .card-text {
        padding: 18px;
        text-align: center;
    }

    .item-title {
        font-family: 'Playfair Display', serif;
        color: #FFFFFF;
        font-size: 18px;
        margin-bottom: 6px;
    }

    .item-desc {
        font-family: 'Inter', sans-serif;
        color: #8B949E;
        font-size: 13px;
        line-height: 1.4;
        height: 38px;
        overflow: hidden;
    }

    /* OCR/LLM Box Styling */
    .ocr-box {
        background: rgba(212, 175, 55, 0.05);
        border: 1px solid #D4AF37;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
    }

    /* THE INVISIBLE BUTTON HACK */
    /* This makes the Streamlit button cover the entire card */
    div.stButton > button {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 420px; /* Covers the image and text area */
        background: transparent !important;
        border: none !important;
        color: transparent !important;
        z-index: 10;
        cursor: pointer;
    }
    div.stButton > button:hover { background: rgba(212, 175, 55, 0.03) !important; }
    div.stButton > button:active { background: transparent !important; }

    /* Fix Sidebar */
    [data-testid="stSidebar"] { border-right: 1px solid #2D333B; }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<h1 style="color:#D4AF37; font-family:Playfair Display;">JewelUX</h1>', unsafe_allow_html=True)
    menu = st.radio("SEARCH MODES", ["🏠 Home", "🔍 Text Search", "🖼️ Image Search", "✍️ Handwriting Search"], label_visibility="collapsed")
    
    if st.session_state.last_menu != menu:
        st.session_state.query_emb, st.session_state.query_text, st.session_state.use_sbir = None, "", False
        st.session_state.last_menu = menu

    st.write("---")
    top_k = st.slider("Results", 3, 21, 12, step=3)
    if st.button("Reset All", use_container_width=True):
        st.session_state.query_emb, st.session_state.use_sbir = None, False
        st.rerun()

st.markdown(f"## {menu.split(' ', 1)[1]}")

# --- INPUT MODES ---
if "Text Search" in menu:
    text_input = st.text_input("Describe your piece:", key="txt_s")
    if text_input:
        st.session_state.query_text, st.session_state.use_sbir = text_input, False
        st.session_state.query_emb = engine.get_text_embedding(text_input)

elif "Image Search" in menu:
    uploaded = st.file_uploader("Upload photo", type=['jpg', 'png', 'jpeg', 'webp'], key="img_s")
    if uploaded:
        st.session_state.use_sbir, st.session_state.query_text = False, ""
        st.session_state.query_emb = engine.get_image_embedding(Image.open(uploaded))

elif "Handwriting Search" in menu:
    hw_tabs = st.tabs(["📝 Handwritten Note", "🎨 Hand-drawn Sketch"])
    
    with hw_tabs[0]:
        uploaded_text = st.file_uploader("Upload note", type=['jpg', 'png', 'jpeg', 'webp'], key="hw_t")
        if uploaded_text:
            img = Image.open(uploaded_text)
            with st.spinner("🤖 AI reading & refining..."):
                raw_ocr, cleaned_query = ocr.extract_text(img.convert('RGB'))
            
            if cleaned_query:
                st.markdown(f"""
                <div class="ocr-box">
                    <p style='color:#8B949E; margin-bottom:4px; font-size:12px;'>Raw Text: <del>{raw_ocr}</del></p>
                    <p style='font-size:18px; color:#D4AF37;'>✨ AI Refined: <b>{cleaned_query}</b></p>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.query_text = cleaned_query
                st.session_state.query_emb = engine.get_text_embedding(cleaned_query)
                st.session_state.use_sbir = False

    with hw_tabs[1]:
        uploaded_sketch = st.file_uploader("Upload sketch", type=['jpg', 'png', 'jpeg', 'webp'], key="hw_s")
        if uploaded_sketch:
            st.session_state.use_sbir = True
            st.session_state.query_text = ""
            st.session_state.query_emb = engine.get_sketch_embedding(Image.open(uploaded_sketch))

# --- RESULTS AREA ---
if st.session_state.query_emb is not None:
    st.markdown("<p style='color:#D4AF37; font-family:Inter; font-weight:600;'>CURATED RECOMMENDATIONS</p>", unsafe_allow_html=True)
    
    q_vec = st.session_state.query_emb.reshape(1, -1).astype('float32')
    faiss.normalize_L2(q_vec)
    
    active_index = index_sbir if st.session_state.use_sbir else index_std
    D, I = active_index.search(q_vec, k=50) 
    
    ranked = hybrid_searcher.get_hybrid_scores(st.session_state.query_text, q_vec[0], I[0], D[0], top_k=top_k)

    cols = st.columns(3)
    for idx, res in enumerate(ranked):
        item = res['metadata']
        v_idx = metadata[metadata['path'] == item['path']].index[0]
        
        with cols[idx % 3]:
            img_b64 = get_base64_image(item['path'])
            
            # THE CLICKABLE CARD UI
            st.markdown(f"""
                <div class="result-block">
                    <div class="item-card">
                        <div class="badge">{int(res['score']*100)}% Match</div>
                        <img class="card-img" src="data:image/jpeg;base64,{img_b64}">
                        <div class="card-text">
                            <div class="item-title">Premium {item['category'].title()}</div>
                            <div class="item-desc">{item['description'][:75]}...</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # This button is INVISIBLE but covers the whole card due to CSS
            if st.button("", key=f"card_{idx}"):
                st.session_state.query_emb, st.session_state.use_sbir = vectors[v_idx], False
                st.session_state.query_text = item['description']
                st.rerun()

elif "Home" in menu:
    st.info("Welcome to JewelUX. Search by text, image, or sketch to explore.")
    st.image("https://images.unsplash.com/photo-1515562141207-7a88fb7ce338?q=80&w=1000", width=1000)