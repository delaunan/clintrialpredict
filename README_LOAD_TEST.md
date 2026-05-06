# CTPredict: UI Load Test (Playwright)

This directory contains a Playwright-based load testing suite for the CTPredict Streamlit application. It simulates multiple independent browser users performing the full search-and-prediction workflow.

## What this test proves
- **Real Browser Sessions:** Unlike standard HTTP load tests, this uses headless Chromium to simulate real user interactions with Streamlit's reactive state machine.
- **Scaling Limits:** Verifies how many concurrent Streamlit sessions the current Cloud Run configuration can handle before slowing down or crashing.
- **Search-to-Prediction Path:** Tests the full integration between the Frontend (Streamlit) and Backend (FastAPI Scoring Engine).
- **Native Dataframe Interaction:** Validates the reliability of native `st.dataframe` row selection under load.
- **Guardrail Integrity:** Confirms that Simulation Mode correctly blocks predictions and that reverting it allows normal operation.

## Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv-load-test
   source venv-load-test/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements-load-test.txt
   ```

3. Install Playwright browsers:
   ```bash
   python -m playwright install chromium
   ```

## Validation Tests (Run these first)
Before running a large load test, visually validate that the automated clicks are correct for the current UI layout.

```bash
# Validate basic click-to-trial flow
python load_test_ui.py --users 1 --headful --timeout 120000 --scenario basic

# Validate prediction flow with multiple users
python load_test_ui.py --users 3 --headful --ramp-seconds 1

# Validate simulation guardrail, then turn Simulation Mode off and validate real prediction
python load_test_ui.py --users 1 --headful --timeout 120000 --scenario simulation-block
```

## Running Launch-Readiness Tests

### Standard Scenarios
```bash
# 5 users spread across 1 second
python load_test_ui.py --users 5 --ramp-seconds 1

# 5 users spread across 1 second (Headful mode to watch)
python load_test_ui.py --users 5 --ramp-seconds 1 --headful

# 10 users spread across 1 second
python load_test_ui.py --users 10 --ramp-seconds 1

# 15 users spread across 1 second
python load_test_ui.py --users 15 --ramp-seconds 1

# 20 users spread across 1 second
python load_test_ui.py --users 20 --ramp-seconds 1

# 15 users (Sudden burst - 0s ramp)
python load_test_ui.py --users 15 --ramp-seconds 0
```

### Scenario Options
- `--scenario basic`: Stop after opening a trial.
- `--scenario prediction`: Perform full search -> open -> predict (Default).
- `--scenario simulation-block`: Turn Simulation Mode on, verify prediction is blocked, turn it off, then verify real prediction works.

## Interpreting Results

The script generates a `load_test_results/` directory containing:
- `load_test_results.json`: Detailed timing and status for every user.
- `load_test_results.csv`: Flat table for analysis in Excel/Pandas.
- `screenshots/`: Diagnostic screenshots for any failed users.

### Cloud Run Log Checks (Post-Run)

After a run, check the Google Cloud Console for these specific errors:

**UI Memory Errors:**
```sql
resource.type="cloud_run_revision"
resource.labels.service_name="clintrial-ui"
textPayload:"Memory limit"
```

**UI Python Crashes:**
```sql
resource.type="cloud_run_revision"
resource.labels.service_name="clintrial-ui"
severity>=ERROR
(textPayload:"Traceback" OR textPayload:"NameError" OR textPayload:"KeyError" OR textPayload:"AttributeError" OR textPayload:"ValueError")
```

**API Errors:**
```sql
resource.type="cloud_run_revision"
resource.labels.service_name="clintrial-api"
severity>=ERROR
```

**Prediction Audit Errors:**
```sql
resource.type="cloud_run_revision"
resource.labels.service_name="clintrial-ui"
jsonPayload.app="ctpredict"
jsonPayload.event=("prediction_api_error" OR "prediction_timeout" OR "prediction_request_exception" OR "prediction_invalid_response" OR "prediction_unexpected_error")
```
