# 💎 JewelUX: An AI-Powered Multimodal Jewelry Recommendation and Retrieval System

JewelUX is a premium, AI-driven jewelry recommendation system that redefines how users discover luxury items. By combining state-of-the-art Computer Vision (**CLIP**) with lightning-fast vector search (**FAISS**), JewelUX enables a truly multimodal search experience—find your perfect piece through text, images, hand-drawn sketches, or even handwriting.

---

## Key Features

- Multimodal Search Engine: 
  - **Text-to-Image**: Search using natural language ("Gold necklace with rubies").
  - **Image Similarity**: Upload a photo to find visually matching jewelry from the inventory.
  - **Sketch-to-Item (SBIR)**: Draw a rough sketch in the UI and see it matched to real products.
  - **Handwriting Search**: Upload a handwritten note; the system extracts the text using OCR and performs a search.
- Real-time Market Ticker: Live simulated rates for Gold, Silver, and Diamonds.
- Liquid Gold UI: A high-end aesthetic featuring glassmorphism, holographic interactions, and custom "Aura Cursor" tracking.
- Dynamic Smart Tags: Automatically generated search suggestions based on current inventory metadata.
- Deep Insights**: Interactive product modals with similarity-based recommendations and technical specifications.

---

## How It Works (Search Logic)

JewelUX uses a sophisticated **Multi-Stage Retrieval** pipeline to ensure the most relevant jewelry is found:

1.  **Stage 1: Vector Retrieval (CLIP)** - The system converts text or images into a 512-dimensional vector and performs an approximate nearest neighbor search in FAISS to find the top 100 visual matches.
2.  **Stage 2: Semantic Filtering (BM25)** - For text queries, a BM25 keyword search is performed over the product descriptions to find exact term matches (e.g., "ruby", "18k").
3.  **Stage 3: Hybrid Scoring** - Scores from CLIP and BM25 are fused (40% Visual, 60% Textual) to rank candidates that are both visually and descriptively accurate.
4.  **Stage 4: Neural Reranking** - The top candidates are passed through a **Cross-Encoder** model that analyzes the query-description pair with deep attention, pushing the absolute best matches to the top.
5.  **Stage 5: Categorical Guardrails** - If the AI detects a specific category (like "ring"), it automatically suppresses or filters out non-matching items to prevent irrelevant noise.

---

## 🛠️ Technical Stack & AI Models

### **Backend (Python / FastAPI)**
- **Core AI Engine**: [OpenAI CLIP (ViT-B/32)](https://github.com/openai/CLIP) for generating cross-modal embeddings for text, images, and sketches.
- **Vector Database**: [Meta FAISS](https://github.com/facebookresearch/faiss) for high-performance similarity search across high-dimensional vectors.
- **Reranker**: [Cross-Encoder (ms-marco-MiniLM-L-6-v2)](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) to refine search results for maximum relevance.
- **OCR Engine**: [TrOCR (microsoft/trocr-small-handwritten)](https://huggingface.co/microsoft/trocr-small-handwritten) for high-accuracy handwriting recognition.
- **Query Refinement**: Integrated LLM logic to clean OCR noise and detect search categories automatically.
- **Hybrid Search**: A multi-stage retrieval system combining semantic vector scores, categorical filtering, and neural reranking.

### **Frontend (React / Vite)**
- **Styling**: Vanilla CSS + Tailwind CSS for a premium design system.
- **Animations**: Framer Motion for fluid transitions and micro-interactions.
- **State Management**: React Hooks (useState/useEffect) for real-time UI updates.
- **Networking**: Axios with centralized configuration for API communication.

---

## 🚀 Getting Started

### Prerequisites
- **Python**: 3.9 or higher
- **Node.js**: 18.0 or higher
- **Hardware**: 8GB+ RAM (NVIDIA GPU recommended for TrOCR/CLIP inference)

### Setup & Installation

1. **Clone the Project**
   ```bash
   git clone https://github.com/GiriPrasathGA/Multimodal-Jewelry-Recommendation-System.git
   cd Multimodal-Jewelry-Recommendation-System
   ```

2. **Setup Backend**
   ```bash
   cd backend
   python -m venv .venv
   # Windows
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Setup Frontend**
   ```bash
   cd ../frontend
   npm install
   ```

4. **Configure Environment**
   Create a `.env` file in the `backend/` directory:
   ```env
   LLM_API_KEY=your_api_key_here
   LLM_BASE_URL=https://your_provider_url_here
   ```

### Running the Application

Start both the backend and frontend with a single command from the project root:

**Option 1: Python (Cross-platform)**
```bash
python run.py
```

**Option 2: PowerShell (Windows)**
```powershell
./run_servers.ps1
```

---

## 🔌 API Endpoints (Documentation)

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/search/text` | `POST` | Semantic search using a text query. |
| `/search/image` | `POST` | Visual similarity search using an uploaded file. |
| `/search/sketch` | `POST` | Sketch-to-Image retrieval. |
| `/search/ocr` | `POST` | Handwritten query extraction and search. |
| `/search/featured` | `GET` | Retrieve a list of featured/popular items. |
| `/tags` | `GET` | Get dynamically generated smart search tags. |
| `/health` | `GET` | Check system status and resource availability. |

---

## 🗺️ Project Structure

```text
├── backend/                  # FastAPI Application
│   ├── data/                 # Raw dataset (Images & Excel)
│   ├── embeddings/           # FAISS indices (.bin) and Vectors (.npy)
│   ├── metadata/             # Processed CSV data for retrieval
│   ├── utils/                # AI Modules (Embedder, OCR, Reranker, Hybrid)
│   ├── main.py               # Core API Logic
│   └── run.py                # Uvicorn entry point
│
├── frontend/                 # React (Vite) Application
│   ├── src/
│   │   ├── components/       # Visual components (Aura, Modals, Grid)
│   │   ├── App.jsx           # Main logic and state
│   │   └── tailwind.css      # Typography and Design Tokens
│   └── vite.config.js        # Build configuration
│
├── run.py                    # Unified startup script (Multi-threaded)
└── run_servers.ps1           # Windows-native startup script
```

---
*Built as a Capstone Project for Multimodal AI and RAG Architecture.*
