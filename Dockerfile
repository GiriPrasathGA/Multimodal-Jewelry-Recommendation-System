# Use Python 3.9
FROM python:3.9

# Set working directory to /code
WORKDIR /code

# Copy requirements from backend
COPY ./backend/requirements.txt /code/requirements.txt

# Install dependencies (CPU version for speed, though HF has 16GB RAM so generic is fine too)
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the backend code into /code
COPY ./backend /code

# Create directories for writable data (HF Spaces specific permissions)
RUN mkdir -p /code/embeddings /code/metadata /code/data
RUN chmod -R 777 /code/embeddings /code/metadata /code/data

# Expose port (HF uses 7860 by default)
EXPOSE 7860

# Command to run the app
# Note: We use 7860 as the port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
