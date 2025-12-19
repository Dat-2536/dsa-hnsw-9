# 1. Use Python 3.9 slim
FROM python:3.9-slim

# 2. Install System Dependencies
# We need these to compile dlib
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# --- THE FIX IS HERE ---
# This forces dlib to use only 1 CPU core during compilation.
# It prevents the RAM explosion (OOM Error).
ENV CMAKE_BUILD_PARALLEL_LEVEL=1

# 4. Install face-recognition (and dlib)
# WARNING: This step will look like it "hangs" for 10-20 minutes. 
# DO NOT CANCEL IT. It is compiling massive C++ code on 1 core.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir face-recognition

# 5. Copy requirements
COPY backend/requirements.txt .

# 6. Install other dependencies
# (Pip will skip face-recognition here since we installed it above)
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy project files
COPY . .

# 8. Install local HNSW
RUN pip install ./backend/hnswlib

# 9. Setup Backend
WORKDIR /app/backend
EXPOSE 8000

# 10. Run App
CMD ["python", "app.py"]