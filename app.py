import uvicorn
import os
import sys

if __name__ == "__main__":
    # Add the current directory to sys.path to ensure backend can be imported
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the uvicorn server
    # working_directory is set to current directory by default
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)