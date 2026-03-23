# 1. Base Image
FROM python:3.12-slim

# 2. Setup production environment
WORKDIR /prod

# 3. Install dependencies first
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. SURGICAL COPY: ONLY absolute minimum files
# Backend Code
COPY api/main.py api/main.py
COPY api/__init__.py api/__init__.py

# Preprocessing Logic (Required by Joblib to unpickle the model)
COPY src/__init__.py src/__init__.py
COPY src/prep/__init__.py src/prep/__init__.py
COPY src/prep/pipeline.py src/prep/pipeline.py

# Production Artifacts
COPY models/model_prod_01.joblib models/model_prod_01.joblib
COPY models/shap_values_01.joblib models/shap_values_01.joblib
COPY models/thresholds_01.json models/thresholds_01.json
COPY models/taxonomy_01.json models/taxonomy_01.json

# 5. Start the API
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
