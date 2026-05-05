import asyncio
import argparse
import json
import time
import os
import random
from datetime import datetime
from pathlib import Path
import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
DEFAULT_URL = "https://clintrial-ui-835962039082.europe-west1.run.app"
DEFAULT_USERS = 5
DEFAULT_RAMP = 1
DEFAULT_TIMEOUT = 60000  # 60 seconds
OUTPUT_DIR = Path("load_test_results")
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"

# Streamlit Dataframe Geometry Constants
DF_SELECTOR = '[data-testid="stDataFrame"]'
DF_HEADER_HEIGHT = 40
DF_ROW_HEIGHT = 34

# ==============================================================================
# LOGGING UTILITY
# ==============================================================================
def log(user_id, message):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [User {user_id:02d}] {message}")

# ==============================================================================
# UI HELPERS
# ==============================================================================

async def wait_for_any_selector(page, selectors, timeout_ms, user_id):
    """
    Waits for any of the given selectors to become visible.
    """
    start_wait = time.perf_counter()
    while (time.perf_counter() - start_wait) * 1000 < timeout_ms:
        for selector in selectors:
            try:
                if await page.is_visible(selector):
                    return selector
            except:
                continue
        await asyncio.sleep(0.5)
    return None

async def click_dataframe_row(page, row_index, user_id, timeout):
    """
    Robustly clicks a row in the Streamlit dataframe selection band.
    Tries multiple x-offsets to ensure the click hits the selection area.
    """
    log(user_id, f"Searching for dataframe to select row {row_index}...")
    
    try:
        df_element = await page.wait_for_selector(DF_SELECTOR, timeout=timeout)
        box = await df_element.bounding_box()
        if not box:
            raise Exception("Dataframe found but has no bounding box.")
            
        x_offsets = [18, 25, 35, 45]
        
        async def try_row(target_row):
            click_y = box['y'] + DF_HEADER_HEIGHT + (target_row * DF_ROW_HEIGHT) + (DF_ROW_HEIGHT / 2)
            for x_off in x_offsets:
                click_x = box['x'] + x_off
                log(user_id, f"Attempting click on Row {target_row} at x-offset {x_off}...")
                await page.mouse.click(click_x, click_y)
                
                # Success Evidence: Back button or Predict button
                success = await wait_for_any_selector(
                    page, 
                    ['text="Back to Results"', 'text="Predict Trial Completion"'], 
                    5000, 
                    user_id
                )
                if success:
                    log(user_id, f"Success! Trial page detected via {success}")
                    return True
            return False

        if await try_row(row_index):
            return True
            
        if row_index == 0:
            log(user_id, "Row 0 failed. Falling back to Row 1 retry...")
            if await try_row(1):
                return True

        raise Exception(f"Failed to open trial page after multiple click attempts on row {row_index}")
        
    except Exception as e:
        log(user_id, f"Dataframe click error: {e}")
        raise

async def wait_for_prediction_result(page, user_id, timeout):
    """
    Waits for robust evidence that the prediction result has rendered.
    Requires BOTH the Plotly chart AND a strong text indicator.
    """
    log(user_id, "Waiting for prediction result (Plotly + Completion Score text)...")
    
    text_indicators = [
        'text="Completion Score"',
        'text="Completion likelihood"',
        'text="Interactive score drivers"',
        'text="Predictive"'
    ]
    
    start_wait = time.perf_counter()
    while time.perf_counter() - start_wait < (timeout / 1000):
        if await page.is_visible('[data-testid="stPlotlyChart"]'):
            for text_sel in text_indicators:
                if await page.is_visible(text_sel):
                    log(user_id, f"Prediction result confirmed via {text_sel}")
                    return True
        await asyncio.sleep(1.0)
        
    raise Exception("Timed out waiting for robust prediction result (Plotly + Text).")

async def detect_simulation_block(page, user_id, timeout):
    """
    Detects if the simulation guardrail notice is visible.
    """
    log(user_id, "Checking for simulation block notice...")
    block_indicators = [
        "Explore Additional Capabilities",
        "not available",
        "disabled",
        "contact"
    ]
    
    start_wait = time.perf_counter()
    while time.perf_counter() - start_wait < (timeout / 1000):
        content = await page.content()
        for indicator in block_indicators:
            if indicator in content:
                log(user_id, f"Guardrail detected: '{indicator}'")
                return True
        await asyncio.sleep(1.0)
        
    return False

# ==============================================================================
# SIMULATED USER SCENARIO
# ==============================================================================

async def run_user_lifecycle(browser, user_id, args, start_delay):
    if start_delay > 0:
        await asyncio.sleep(start_delay)
        
    log(user_id, f"User session starting (Scenario: {args.scenario})...")
    
    results = {
        "user_id": user_id,
        "scenario": args.scenario,
        "success": False,
        "error": None,
        "timings": {},
        "started_at": datetime.now().isoformat(),
        "ended_at": None,
        "row_index": args.row_index,
        "screenshot": None,
        "dom_snapshot": None
    }
    
    start_time = time.perf_counter()
    context = None
    
    try:
        context = await browser.new_context(viewport={'width': 1440, 'height': 900})
        page = await context.new_page()
        
        # 1. Open App
        log(user_id, f"Opening {args.url}...")
        step_start = time.perf_counter()
        await page.goto(args.url, wait_until="domcontentloaded", timeout=args.timeout)
        await page.wait_for_selector('text="CTPredict"', timeout=args.timeout)
        results["timings"]["open_app_seconds"] = time.perf_counter() - step_start
        
        # 2. Search Trials
        log(user_id, "Clicking 'Search Trials'...")
        step_start = time.perf_counter()
        search_btn = page.get_by_role("button", name="Search Trials")
        await search_btn.click()
        results["timings"]["search_seconds"] = time.perf_counter() - step_start
        
        # 3. Grid Wait
        log(user_id, "Waiting for results grid...")
        step_start = time.perf_counter()
        await page.wait_for_selector(DF_SELECTOR, timeout=args.timeout)
        results["timings"]["grid_wait_seconds"] = time.perf_counter() - step_start
        
        await asyncio.sleep(2.0)

        # 4. Open Trial (MUST HAPPEN BEFORE TOGGLING SIMULATION MODE)
        log(user_id, f"Opening trial at row {args.row_index}...")
        step_start = time.perf_counter()
        await click_dataframe_row(page, args.row_index, user_id, args.timeout)
        results["timings"]["open_trial_seconds"] = time.perf_counter() - step_start

        # 5. Scenario: Simulation Block - Turn on Toggle (ON DETAIL PAGE)
        if args.scenario == "simulation-block":
            log(user_id, "Toggling Simulation Mode ON (on detail page)...")
            try:
                # Try primary label selector
                toggle = page.get_by_label("Simulation Mode (Editing Content)")
                await toggle.click(timeout=5000)
            except:
                try:
                    # Fallback text-based selector
                    toggle = page.locator('text="Simulation Mode"').first
                    await toggle.click(timeout=5000)
                except:
                    raise Exception("Simulation Mode toggle was not found on the trial detail page.")
            await asyncio.sleep(2.0)

        if args.scenario == "basic":
            results["success"] = True
            log(user_id, "Basic scenario complete.")
            return results

        # 6. Predict
        log(user_id, "Clicking 'Predict Trial Completion'...")
        step_start = time.perf_counter()
        predict_btn = page.get_by_role("button", name="Predict Trial Completion")
        await predict_btn.click()
        
        if args.scenario == "simulation-block":
            blocked = await detect_simulation_block(page, user_id, args.timeout)
            if blocked:
                results["success"] = True
                log(user_id, "Guardrail verification success.")
            else:
                raise Exception("Simulation mode was ON but no guardrail notice appeared after Predict.")
        else:
            # prediction scenario
            await wait_for_prediction_result(page, user_id, args.timeout)
            results["success"] = True
            log(user_id, "Prediction scenario success.")
            
        results["timings"]["prediction_seconds"] = time.perf_counter() - step_start

    except Exception as e:
        results["error"] = str(e)
        log(user_id, f"ERROR: {e}")
        
        try:
            SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
            ss_path = SCREENSHOT_DIR / f"user_{user_id:02d}_failure.png"
            await page.screenshot(path=str(ss_path))
            results["screenshot"] = str(ss_path)
            
            dom_path = SCREENSHOT_DIR / f"user_{user_id:02d}_failure.html"
            with open(dom_path, "w") as f:
                f.write(await page.content())
            results["dom_snapshot"] = str(dom_path)
        except:
            pass

    finally:
        results["ended_at"] = datetime.now().isoformat()
        results["timings"]["total_seconds"] = time.perf_counter() - start_time
        if context:
            await context.close()
            
    return results

# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

async def main():
    parser = argparse.ArgumentParser(description="CTPredict UI Load Test (Playwright)")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"UI URL")
    parser.add_argument("--users", type=int, default=DEFAULT_USERS, help=f"Simulated users")
    parser.add_argument("--ramp-seconds", type=float, default=DEFAULT_RAMP, help=f"Ramp up time")
    parser.add_argument("--headful", action="store_true", help="Run headful")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout in ms")
    parser.add_argument("--scenario", choices=["basic", "prediction", "simulation-block"], default="prediction")
    parser.add_argument("--row-index", type=int, default=0, help="Grid row index to select")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory")
    
    args = parser.parse_args()
    
    global OUTPUT_DIR, SCREENSHOT_DIR
    OUTPUT_DIR = Path(args.output_dir)
    SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print(f"CTPredict LOAD TEST")
    print(f"Scenario:     {args.scenario}")
    print(f"Users:        {args.users}")
    print(f"Ramp:         {args.ramp_seconds}s")
    print(f"Target:       {args.url}")
    print("="*70 + "\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not args.headful)

        tasks = []
        for i in range(args.users):
            user_id = i + 1
            start_delay = 0 if args.ramp_seconds == 0 else (args.ramp_seconds / args.users) * i
            
            tasks.append(asyncio.create_task(
                run_user_lifecycle(browser, user_id, args, start_delay)
            ))
            
        all_results = await asyncio.gather(*tasks, return_exceptions=False)
        await browser.close()

    # --- PROCESS RESULTS ---
    successes = [r for r in all_results if r["success"]]
    failures = [r for r in all_results if not r["success"]]
    
    def avg_t(key):
        vals = [r["timings"].get(key, 0) for r in successes if key in r["timings"]]
        return sum(vals) / len(vals) if vals else 0

    # --- TERMINAL SUMMARY ---
    print("\n" + "="*70)
    print("LOAD TEST FINAL SUMMARY")
    print("="*70)
    print(f"Users Launched:      {args.users}")
    print(f"Successes:           {len(successes)}")
    print(f"Failures:            {len(failures)}")
    print(f"Success Rate:        {(len(successes)/args.users)*100:.1f}%")
    print("-" * 35)
    print(f"Avg App Open:        {avg_t('open_app_seconds'):.2f}s")
    print(f"Avg Search Click:    {avg_t('search_seconds'):.2f}s")
    print(f"Avg Grid Wait:       {avg_t('grid_wait_seconds'):.2f}s")
    print(f"Avg Trial Open:      {avg_t('open_trial_seconds'):.2f}s")
    print(f"Avg Prediction:      {avg_t('prediction_seconds'):.2f}s")
    print(f"Avg Total Duration:  {avg_t('total_seconds'):.2f}s")
    print("-" * 70)
    
    if failures:
        print("FAILED USERS:")
        for f in failures:
            print(f"  User {f['user_id']:02d}: {f['error']}")
    else:
        print("PERFECT RUN: All users completed the flow.")
    
    with open(OUTPUT_DIR / "load_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    df_results = pd.DataFrame([
        {
            "user_id": r["user_id"],
            "scenario": r["scenario"],
            "success": r["success"],
            "total": r["timings"].get("total_seconds"),
            "app_open": r["timings"].get("open_app_seconds"),
            "search": r["timings"].get("search_seconds"),
            "grid_wait": r["timings"].get("grid_wait_seconds"),
            "trial_open": r["timings"].get("open_trial_seconds"),
            "prediction": r["timings"].get("prediction_seconds"),
            "error": r["error"]
        } for r in all_results
    ])
    df_results.to_csv(OUTPUT_DIR / "load_test_results.csv", index=False)
    
    print(f"\nFiles saved to {OUTPUT_DIR}/")
    print("="*70 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
