# 1. Use Python 3.9 Slim as a light base image
FROM python:3.9-slim

# 2. Install system tools required for dlib and OpenCV
# (Render does not ship these by default, so install them manually)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory in the container
WORKDIR /app

# 4. Copy requirements.txt first to leverage Docker layer cache
COPY requirements.txt .

# 5. Install dependencies (building dlib the first time can take 5-10 minutes)
RUN pip install --no-cache-dir -r requirements.txt
# Install Gunicorn if it is not already in requirements
RUN pip install gunicorn

# 6. Copy application code
COPY . .

# 7. Expose port 10000 (Render commonly uses this port, or adjust as needed)
EXPOSE 10000

# 8. Start the server
# app:app means: module app.py, Flask variable named app
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:10000", "app:app"]