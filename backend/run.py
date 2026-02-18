
import uvicorn
import os
import sys

# Ensure backend directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    try:
        port = int(os.getenv("PORT", 8000))
        print(f"Starting uvicorn on port {port}...", flush=True)
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, log_level="info")
    except Exception as e:
        print(f"Failed to start uvicorn: {e}", flush=True)
        import traceback
        traceback.print_exc()
    import time
    while True:
        time.sleep(1)

