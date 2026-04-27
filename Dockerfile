# 1. Base Image
FROM python:3.12-slim

# 2. Setup production environment
WORKDIR /prod

# 3. Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy Code & Artifacts
# Backend
COPY api/ api/
# Frontend
COPY frontend/ frontend/
# Shared Logic (Needed for Model unpickling)
COPY src/ src/
# Models
COPY models/ models/

# 5. Default Command (Can be overridden in Cloud Run)
# To run API: uvicorn api.main:app --host 0.0.0.0 --port 8000
# To run UI:  streamlit run frontend/app.py --server.port 8080 --server.address 0.0.0.0
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
