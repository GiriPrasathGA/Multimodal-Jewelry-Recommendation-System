import subprocess
import os
import sys
import time
import signal

def run_services():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(base_dir, "backend")
    frontend_dir = os.path.join(base_dir, "frontend")
    venv_python = os.path.join(base_dir, ".venv", "Scripts", "python.exe")

    print("🚀 Starting JewelUX Project...")

    # Start Backend
    print("📦 Starting Backend (FastAPI)...")
    backend_proc = subprocess.Popen(
        [venv_python, "run.py"],
        cwd=backend_dir
    )

    # Start Frontend
    print("🎨 Starting Frontend (Vite)...")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
        shell=True
    )

    print("\n✅ Both services are starting!")
    print(f"🔗 Frontend: http://localhost:5173")
    print(f"🔗 Backend:  http://localhost:8000")
    print("\nPress Ctrl+C to stop both services.\n")

    try:
        while True:
            time.sleep(1)
            # Check if processes are still running
            if backend_proc.poll() is not None:
                print("❌ Backend process terminated unexpectedly.")
                break
            if frontend_proc.poll() is not None:
                print("❌ Frontend process terminated unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
    finally:
        # Graceful shutdown
        backend_proc.terminate()
        frontend_proc.terminate()
        print("👋 Services stopped.")

if __name__ == "__main__":
    run_services()
