# Deployment Guide

This guide explains how to deploy the Clinical Trial Predictor application to Google Cloud Run using the provided helper script.

## Where to run these commands

All commands must be run from the **root** of the local `clintrialpredict` repository in your terminal.

## One-time setup / authentication

If this is your first time deploying, or if your session has expired, you need to configure your environment:

1.  **Open Docker Desktop**: Ensure Docker is running on your machine.
2.  **Run Authentication**:
    ```bash
    ./scripts/deploy.sh auth
    ```
    This command will:
    *   Log you into Google Cloud (`gcloud auth login`).
    *   Set the project to `clintrial-predict-2025`.
    *   Configure Docker to push images to the Google Artifact Registry in `europe-west1`.

## Before every deployment

Before you attempt to deploy, always run a diagnostic check to ensure everything is ready:

```bash
./scripts/deploy.sh check
```

*   **If Docker is missing/not running**: Open Docker Desktop and rerun the check.
*   **If Auth/Project is missing**: Run `./scripts/deploy.sh auth`.

## Normal UI update

If you have only made changes to the frontend (e.g., editing `frontend/app.py`), use this command:

```bash
./scripts/deploy.sh ui
```

This will build a new Docker image, push it to the registry, and update **only** the `clintrial-ui` service. It does not touch the API service.

## API update

If you have made changes to the backend or model (e.g., `api/main.py` or files in `models/`), use this command:

```bash
./scripts/deploy.sh api
```

This will build a new Docker image, push it to the registry, and update **only** the `clintrial-api` service.

## Full update: API + UI

If you have made major changes or are unsure, deploy both services at once:

```bash
./scripts/deploy.sh all
```

This builds and pushes the image once, then updates both the API and the UI services sequentially.

## What each command does

*   **`check`**: Verifies `gcloud`, `docker`, authentication, and project settings without changing anything.
*   **`auth`**: Performs the necessary logins and configurations for Google Cloud and Docker.
*   **`build`**: Locally builds the Docker image for the `linux/amd64` platform.
*   **`push`**: Uploads the locally built image to the Google Artifact Registry.
*   **`ui` / `api` / `all`**: These commands perform the full workflow: **Check -> Build -> Push -> Deploy**. Pushing the image alone is not enough; the Cloud Run service must be explicitly updated to use the new image.

## How to verify after deployment

Once the script finishes successfully, follow these steps to verify:

1.  **Open the UI URL**: The script will provide the URL, or you can find it in the Google Cloud Console.
2.  **Test Search**: Use the "Search Trials" feature to ensure the database connection and search registry are working.
3.  **Test Prediction**:
    *   Open one trial from the search results.
    *   Navigate to the "Trial Forensic" or "Predict Trial Completion" section.
    *   Click the button to run a prediction.
    *   Verify that the Success Score and SHAP explanations load correctly. This confirms the UI is successfully communicating with the API.

The UI is configured to talk to the API at:
`https://clintrial-api-835962039082.europe-west1.run.app/predict`

## Troubleshooting

*   **Docker daemon not running**: Ensure Docker Desktop is open and showing a green "running" status.
*   **Wrong Google Cloud project**: The script expects `clintrial-predict-2025`. Run `./scripts/deploy.sh auth` to fix this.
*   **Not authenticated with gcloud**: Run `./scripts/deploy.sh auth` to log in again.
*   **Permission denied**: If you cannot run the script, ensure it is executable:
    ```bash
    chmod +x scripts/deploy.sh
    ```
*   **Build failures**: Check the console output for Python dependency errors or Dockerfile issues.

## Quick command summary

| Scenario | Command |
| :--- | :--- |
| Check environment | `./scripts/deploy.sh check` |
| First-time setup | `./scripts/deploy.sh auth` |
| **Update Frontend/UI only** | `./scripts/deploy.sh ui` |
| Update Backend/API only | `./scripts/deploy.sh api` |
| Update everything | `./scripts/deploy.sh all` |
