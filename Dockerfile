# 1. Base Python Image
FROM python:3.9-slim

# 2. Set Working Directory
WORKDIR /app

# 3. System Dependencies (Updated for latest Debian packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy and Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Project Files
COPY . .

# 6. Expose Ports
EXPOSE 8501 8000

# 7. Start Streamlit Server
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]