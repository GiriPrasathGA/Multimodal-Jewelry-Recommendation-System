# Use Python 3.9
FROM python:3.9

# Set working directory to /code
WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from backend
COPY ./backend/requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the backend code into /code
COPY ./backend /code

# Create directories for writable data (HF Spaces - ensure permissions)
RUN mkdir -p /code/embeddings /code/metadata /code/data
RUN chmod -R 777 /code/embeddings /code/metadata /code/data

# Expose port (HF uses 7860 by default)
EXPOSE 7860

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
