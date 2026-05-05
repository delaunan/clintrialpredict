#!/bin/bash

# CTPredict: Cloud Run Post-Load-Test Health Check
# This script uses gcloud to search for common errors in Cloud Run logs.

PROJECT_ID="clintrial-predict-2025"
UI_SERVICE="clintrial-ui"
API_SERVICE="clintrial-api"

echo "=========================================================="
echo "Checking Cloud Run Health for Project: $PROJECT_ID"
echo "=========================================================="

echo -e "\n[1] Checking for UI Memory Errors (Memory limit exceeded)..."
gcloud logging read "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$UI_SERVICE\" AND textPayload:\"Memory limit\"" --limit 10 --format="table(timestamp, textPayload)"

echo -e "\n[2] Checking for UI Python Tracebacks (Crashes)..."
gcloud logging read "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$UI_SERVICE\" AND severity>=ERROR AND (textPayload:\"Traceback\" OR textPayload:\"NameError\" OR textPayload:\"KeyError\" OR textPayload:\"AttributeError\" OR textPayload:\"ValueError\")" --limit 10 --format="table(timestamp, textPayload)"

echo -e "\n[3] Checking for API Memory Errors..."
gcloud logging read "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$API_SERVICE\" AND textPayload:\"Memory limit\"" --limit 10 --format="table(timestamp, textPayload)"

echo -e "\n[4] Checking for API Service Errors..."
gcloud logging read "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$API_SERVICE\" AND severity>=ERROR" --limit 10 --format="table(timestamp, textPayload)"

echo -e "\n[5] Checking for Prediction Audit Failures (UI perspective)..."
gcloud logging read "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$UI_SERVICE\" AND jsonPayload.app=\"ctpredict\" AND jsonPayload.event=(\"prediction_api_error\" OR \"prediction_timeout\" OR \"prediction_request_exception\" OR \"prediction_invalid_response\" OR \"prediction_unexpected_error\")" --limit 10 --format="json"

echo -e "\nDone. If the above tables are empty, no major errors were detected in the recent logs."
echo "Visit the Google Cloud Console for detailed Metrics and Scaling charts."
