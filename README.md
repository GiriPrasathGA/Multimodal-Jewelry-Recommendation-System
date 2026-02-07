# JewelUX - AI-Powered Jewelry Search

This project consists of a **FastAPI Backend** (for AI search & metadata) and a **React Frontend** (for the UI).

## 🚀 Quick Start

You need two terminal windows to run the project.

### Terminal 1: Backend (Python/FastAPI)
The backend runs on `http://localhost:8000`.

```powershell
cd backend
# Activate Virtual Environment (Windows)
.\.venv\Scripts\Activate.ps1

# Run the Server (with auto-reload)
uvicorn main:app --port 8000 --reload
```

*If you don't have the venv set up yet:*
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Terminal 2: Frontend (React/Vite)
The frontend runs on `http://localhost:5173`.

```powershell
cd frontend
# Install dependencies (only needed once)
npm install

# Start the Dev Server
npm run dev
```

## ✨ Features
- **Live Market Rates**: Real-time simulated ticker for Gold, Silver, Diamond.
- **Smart Filters**: Dynamic tags fetched from your dataset descriptions.
- **Hybrid Search**: Text, Image, and Sketch search capabilities.
- **Visuals**: "Liquid Gold" atmosphere and holographic interactions.