# 🚀 Clintrial Predict - Command Center

## 1. Project Setup
```bash
# Activate your python environment
pyenv activate clintrialpredict

# Install dependencies if they are missing
pip install -r requirements.txt
pip install -r frontend/requirements.txt
```

## 2. Operation Modes & Access

### **Mode 1: Pure Local (Full Dev)**
*Both UI and API run on your laptop. Use this to change model code.*
- **.env** Setting: `# API_URL=...` (Commented out)
- **Terminal 1 (API):** `uvicorn api.main:app --reload`
- **Terminal 2 (UI):** `streamlit run frontend/app.py`
- **Access URL:** `http://localhost:8501`

### **Mode 2: Hybrid (Cloud Brain + Local UI)**
*UI runs on your laptop; API runs on Google Cloud. Use this to design the UI.*
- **.env** Setting: `API_URL=https://clintrialpredict-835962039082.europe-west1.run.app/predict` (Active)
- **Terminal 1 (UI):** `streamlit run frontend/app.py`
- **Access URL:** `http://localhost:8501`

### **Mode 3: Pure Cloud (Production)**
*Both UI and API run on Google Cloud. For public sharing.*
- **Access URL:** `https://clintrialpredict-835962039082.europe-west1.run.app`

---

## 3. Deployment Commands (Sync to Cloud)

### **Step 1: Sync Models to GitHub (Git LFS)**
```bash
# Add all changes including the large model pointers
git add .

# Commit the changes
git commit -m "chore: sync artifacts"

# Push to GitHub (This uploads the actual 100MB files to LFS storage)
git push origin master
```

### **Step 2: Build, Push & Deploy to Google Cloud**
```bash
# A. Build the image (using --platform linux/amd64 for cloud compatibility)
docker build --platform linux/amd64 -t europe-west1-docker.pkg.dev/clintrial-predict-2025/images/api-v01:latest .

# B. Push the image to Google Artifact Registry
docker push europe-west1-docker.pkg.dev/clintrial-predict-2025/images/api-v01:latest

# C. Deploy the image to Cloud Run
gcloud run deploy clintrialpredict \
--image europe-west1-docker.pkg.dev/clintrial-predict-2025/images/api-v01:latest \
--memory 2Gi \
--platform managed \
--region europe-west1 \
--allow-unauthenticated \
--project clintrial-predict-2025
```

---

## 4. Troubleshooting Links
- **Local URL:** `http://localhost:8501` (Only you)
- **Network URL:** `http://172.x.x.x:8501` (Anyone on your Wi-Fi)
- **API Status Check:** `https://clintrialpredict-835962039082.europe-west1.run.app` (Should show "Online")
