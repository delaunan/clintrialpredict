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

### Standard Scenarios (Single Scenario for all users)
```bash
# 5 users spread across 1 second
python load_test_ui.py --users 5 --ramp-seconds 1

# 10 users spread across 1 second
python load_test_ui.py --users 10 --ramp-seconds 1
```

### Mixed & Humanized Scenarios
These simulate more realistic traffic patterns with varying user intents and human-like behaviors.

```bash
# 10-user humanized mixed test (reproducible with seed)
python load_test_ui.py --users 10 --ramp-seconds 10 --timeout 180000 \
  --scenario-mix basic=2,prediction=5,simulation-block=3 \
  --humanize --random-seed 42 --output-dir load_test_results_mix10_humanized

# 20-user humanized mixed test
python load_test_ui.py --users 20 --ramp-seconds 10 --timeout 180000 \
  --scenario-mix basic=3,prediction=11,simulation-block=6 \
  --humanize --random-seed 42 --output-dir load_test_results_mix20_humanized

# 20-user humanized mixed burst (1s ramp)
python load_test_ui.py --users 20 --ramp-seconds 1 --timeout 180000 \
  --scenario-mix basic=3,prediction=11,simulation-block=6 \
  --humanize --random-seed 42 --output-dir load_test_results_mix20_humanized_burst
```

### Scenario & Randomization Options

#### Core Scenarios
- `--scenario basic`: Stop after opening a trial.
- `--scenario prediction`: Perform full search -> open -> predict (Default).
- `--scenario simulation-block`: Turn Simulation Mode on, verify prediction is blocked, turn it off, then verify real prediction works.
- `--scenario-mix`: Specify counts for each scenario (e.g., `basic=2,prediction=5`). Total must match `--users`.

#### Humanization Flag
- `--humanize`: Convenience flag that enables randomized row selection (0-4), randomized desktop viewports, human-like think times (0.3s-1.5s), and start jitter (1.0s).

#### Granular Randomization
- `--random-seed`: Set a seed for reproducible randomized runs (all shuffling and timing).
- `--randomize-rows`: Randomly select trial index (use `--row-index-min` and `--row-index-max`).
- `--randomize-viewports`: Randomly assign desktop viewports (1366x768 up to 1920x1080).
- `--think-time-min / --think-time-max`: Random waits between key actions.
- `--start-jitter-seconds`: Random extra delay per user. **Note:** This is added on top of the `--ramp-seconds` base delay.
- `--no-shuffle-scenarios`: Keep scenario distribution sequential based on `--scenario-mix` input.

## Interpreting Results

The script generates a results directory containing:
- `load_test_results.json`: Detailed timing and status for every user, including their assigned row, viewport, and think time.
- `load_test_results.csv`: Flat table for analysis in Excel/Pandas. Includes actual per-user row, viewport dimensions, jitter, and total start delay fields.
- `screenshots/`: Diagnostic screenshots for any failed users.

The terminal output provides a **Scenario mix breakdown** and average timings for every stage of the user journey.

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
