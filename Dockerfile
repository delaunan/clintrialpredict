# 1. Base Image: Use the slim version (Smaller & Faster than Buster)
FROM python:3.10-slim

# 2. Setup the "Production" folder inside the container
WORKDIR /prod

# 3. Copy Requirements first (For efficient caching)
COPY requirements.txt requirements.txt

# 4. Install Dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 5. Copy Only the Backend Components
# Note: We are NOT copying the 'frontend' folder here.
COPY api api
COPY models models

# 6. Start the API
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT
