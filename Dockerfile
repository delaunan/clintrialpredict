# 1. Base Image
FROM python:3.10-slim

# 2. Setup the "Production" folder inside the container
WORKDIR /prod

# 3. Copy Requirements first
COPY requirements.txt requirements.txt

# 4. Install Dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 5. Copy the Backend Components
# We MUST include 'src' because the model's preprocessing depends on it
COPY api api
COPY models models
COPY src src

# 6. Start the API
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT