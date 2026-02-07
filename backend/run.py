
import uvicorn
import os
import sys

# Ensure backend directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    try:
        print("Starting uvicorn on port 8001...", flush=True)
        uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=False, log_level="debug")
    except Exception as e:
        print(f"Failed to start uvicorn: {e}", flush=True)
        import traceback
        traceback.print_exc()
    import time
    while True:
        time.sleep(1)

