import asyncio
import argparse
import json
import time
import os
import random
import re
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

VALID_SCENARIOS = ["basic", "prediction", "simulation-block"]

# Desktop Viewport Presets
VIEWPORT_PRESETS = [
    {"width": 1366, "height": 768},
    {"width": 1440, "height": 900},
    {"width": 1536, "height": 864},
    {"width": 1600, "height": 900},
    {"width": 1920, "height": 1080},
]

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

async def maybe_think(user_id, user_rng, t_min, t_max, reason, results):
    """
    Simulates human think time between actions.
    """
    if t_max > 0 and t_max >= t_min:
        delay = user_rng.uniform(t_min, t_max)
        # log(user_id, f"Thinking for {delay:.2f}s ({reason})...")
        await asyncio.sleep(delay)
        results["timings"]["total_think_time_seconds"] = results["timings"].get("total_think_time_seconds", 0) + delay

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

async def any_matching_criteria_visible(page):
    """
    Safely checks if any 'matching criteria' text is visible, avoiding strict mode violations.
    """
    try:
        loc = page.get_by_text("matching criteria")
        count = await loc.count()
        for i in range(count):
            try:
                if await loc.nth(i).is_visible():
                    return True
            except:
                continue
    except:
        pass
    return False

async def is_search_button_visible(page):
    """
    Safe visibility check for the Search Trials button.
    """
    try:
        btn = page.get_by_role("button", name="Search Trials")
        return await btn.is_visible(timeout=1000)
    except:
        return False

async def do_search(page, user_id):
    """
    Robustly attempts to click the Search Trials button, handling potential DOM detachments.
    """
    last_error = None
    for attempt in range(3):
        try:
            # Re-locate every time to avoid stale elements
            btn = page.get_by_role("button", name="Search Trials")
            await btn.wait_for(state="visible", timeout=5000)
            try:
                await btn.scroll_into_view_if_needed(timeout=3000)
            except:
                pass
            await btn.click(timeout=5000)
            await asyncio.sleep(0.75)
            return True
        except Exception as e:
            last_error = e
            log(user_id, f"Search click attempt {attempt+1} failed: {e}. Retrying...")
            await asyncio.sleep(0.5)
    raise Exception(f"Search Trials button could not be clicked reliably: {last_error}")

async def wait_for_results_page_ready(page, user_id, timeout_ms):
    """
    Verifies the results page (st.dataframe) is stable, visible, and has non-trivial dimensions.
    """
    log(user_id, "Verifying results page is stable and ready...")
    start_wait = time.perf_counter()
    
    while (time.perf_counter() - start_wait) * 1000 < timeout_ms:
        df = page.locator(DF_SELECTOR)
        if not await df.is_visible():
            if await is_search_button_visible(page) and not await any_matching_criteria_visible(page):
                log(user_id, "Still on landing page, waiting for grid...")
            await asyncio.sleep(1.0)
            continue
            
        box = await df.bounding_box()
        if not box or box['width'] < 100 or box['height'] < 100:
            log(user_id, f"Dataframe box too small ({box}), waiting for layout...")
            await asyncio.sleep(0.5)
            continue
            
        await asyncio.sleep(0.5)
        box2 = await df.bounding_box()
        if not box2 or box2['x'] != box['x'] or box2['y'] != box['y'] or box2['width'] != box['width']:
            log(user_id, "Dataframe box shifted, waiting for stability...")
            continue
            
        log(user_id, f"Results page ready. Box: {box2}")
        return box2
        
    raise Exception("Results grid did not become stable/ready within timeout.")

async def _click_simulation_toggle_base(page, user_id, intent_label):
    """
    Base helper to robustly find and click the Simulation Mode toggle.
    """
    log(user_id, f"Toggling Simulation Mode {intent_label} (on detail page)...")
    
    strategies = [
        lambda p: p.get_by_label("Simulation Mode (Editing Content)"),
        lambda p: p.get_by_label("Simulation Mode"),
        lambda p: p.get_by_role("checkbox", name=re.compile("Simulation Mode", re.I)),
        lambda p: p.get_by_role("switch", name=re.compile("Simulation Mode", re.I)),
        lambda p: p.locator('div[data-testid="stCheckbox"]').filter(has_text=re.compile("Simulation Mode", re.I)),
        lambda p: p.locator('div[data-testid="stToggle"]').filter(has_text=re.compile("Simulation Mode", re.I)),
        lambda p: p.locator('label').filter(has_text=re.compile("Simulation Mode", re.I))
    ]
    
    for i, strategy in enumerate(strategies):
        try:
            element = strategy(page)
            if await element.is_visible(timeout=3000):
                log(user_id, f"Toggle found (strategy {i+1}). Clicking {intent_label}...")
                await element.click()
                await asyncio.sleep(2.0)
                return True
        except:
            continue
            
    raise Exception(f"Simulation Mode toggle was not found or could not be toggled {intent_label}.")

async def toggle_simulation_mode_on(page, user_id):
    return await _click_simulation_toggle_base(page, user_id, "ON")

async def toggle_simulation_mode_off(page, user_id):
    return await _click_simulation_toggle_base(page, user_id, "OFF")

async def click_dataframe_row(page, row_index, user_id, timeout):
    """
    Extremely robust row selection with stability checks and multi-offset retries.
    """
    log(user_id, f"Attempting robust click on Row {row_index}...")
    
    try:
        box = await wait_for_results_page_ready(page, user_id, timeout)
        x_offsets = [25, 40, 18, 45, 70, 100, 140]
        y_shifts = [0, -6, 6]
        tried_coords = []
        
        for y_shift in y_shifts:
            click_y = box['y'] + DF_HEADER_HEIGHT + (row_index * DF_ROW_HEIGHT) + (DF_ROW_HEIGHT / 2) + y_shift
            for x_off in x_offsets:
                if x_off >= box['width']: continue
                click_x = box['x'] + x_off
                tried_coords.append((click_x, click_y))
                
                log(user_id, f"Clicking at ({click_x}, {click_y}) [offset x={x_off}, y_shift={y_shift}]")
                await page.mouse.click(click_x, click_y)
                await asyncio.sleep(0.75)
                
                success = await wait_for_any_selector(
                    page, 
                    ['text="Back to Results"', 'text="Predict Trial Completion"'], 
                    5000, 
                    user_id
                )
                if success:
                    log(user_id, f"Trial opened successfully via {success}")
                    return True
                
                if await is_search_button_visible(page) and not await any_matching_criteria_visible(page):
                    log(user_id, "Returned to landing page during selection. Attempting search retry...")
                    await do_search(page, user_id)
                    box = await wait_for_results_page_ready(page, user_id, timeout)
                    
        raise Exception(f"Failed to open trial page after {len(tried_coords)} click attempts.")
    except Exception as e:
        log(user_id, f"Row click failure: {e}")
        raise

async def wait_for_prediction_result(page, user_id, timeout):
    log(user_id, "Waiting for prediction result (Plotly + Completion Score text)...")
    text_indicators = ['text="Completion Score"', 'text="Completion likelihood"', 'text="Interactive score drivers"', 'text="Predictive"']
    start_wait = time.perf_counter()
    while time.perf_counter() - start_wait < (timeout / 1000):
        if await page.is_visible('[data-testid="stPlotlyChart"]'):
            for text_sel in text_indicators:
                if await page.is_visible(text_sel):
                    log(user_id, f"Prediction result confirmed via {text_sel}")
                    return True
        await asyncio.sleep(1.0)
    raise Exception("Timed out waiting for prediction result indicators.")

async def detect_simulation_block(page, user_id, timeout):
    log(user_id, "Checking for simulation block notice...")
    block_indicators = ["Explore Additional Capabilities", "not available", "disabled", "contact"]
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

async def run_user_lifecycle(browser, user_id, user_config, args, base_delay):
    """
    Manages the full lifecycle of a single simulated user based on their specific config.
    """
    scenario = user_config["scenario"]
    row_index = user_config["row_index"]
    viewport = user_config["viewport"]
    start_jitter = user_config["start_jitter_seconds"]
    t_min = user_config["think_time_min"]
    t_max = user_config["think_time_max"]
    
    # Apply BOTH base ramp delay and per-user jitter
    total_start_delay = base_delay + start_jitter
    if total_start_delay > 0:
        await asyncio.sleep(total_start_delay)
        
    log(user_id, f"User session starting (Scenario: {scenario}, Row: {row_index}, Viewport: {viewport['width']}x{viewport['height']})...")
    
    results = {
        "user_id": user_id,
        "scenario": scenario,
        "row_index": row_index,
        "viewport_width": viewport["width"],
        "viewport_height": viewport["height"],
        "base_ramp_delay_seconds": base_delay,
        "start_jitter_seconds": start_jitter,
        "total_start_delay_seconds": total_start_delay,
        "success": False,
        "error": None,
        "timings": {"total_think_time_seconds": 0.0},
        "started_at": datetime.now().isoformat(),
        "ended_at": None,
        "screenshot": None,
        "dom_snapshot": None
    }
    
    start_time = time.perf_counter()
    context = None
    
    # Each user gets their own seeded RNG for reproducible think times
    user_rng = random.Random(args.random_seed + user_id if args.random_seed is not None else None)
    
    try:
        # Isolated context with specific viewport
        context = await browser.new_context(viewport=viewport)
        page = await context.new_page()
        
        # 1. Open App
        log(user_id, f"Opening {args.url}...")
        step_start = time.perf_counter()
        await page.goto(args.url, wait_until="domcontentloaded", timeout=args.timeout)
        await page.wait_for_selector('text="CTPredict"', timeout=args.timeout)
        results["timings"]["open_app_seconds"] = time.perf_counter() - step_start
        
        await maybe_think(user_id, user_rng, t_min, t_max, "after_app_load", results)
        
        # 2. Search Trials
        log(user_id, "Clicking 'Search Trials'...")
        step_start_search = time.perf_counter()
        await do_search(page, user_id)
        results["timings"]["search_seconds"] = time.perf_counter() - step_start_search
        
        # 3. Grid Wait
        step_start_grid = time.perf_counter()
        try:
            await wait_for_results_page_ready(page, user_id, args.timeout / 2)
        except:
            log(user_id, "Grid not ready. Waiting 5s for late transition...")
            try:
                await wait_for_results_page_ready(page, user_id, 5000)
            except:
                log(user_id, "Grid still not ready. Retrying search once...")
                if await is_search_button_visible(page):
                    await do_search(page, user_id)
                    await wait_for_results_page_ready(page, user_id, args.timeout / 2)
                else:
                    raise Exception("Neither results grid nor Search Trials button was visible after search transition.")
        results["timings"]["grid_wait_seconds"] = time.perf_counter() - step_start_grid
        
        await maybe_think(user_id, user_rng, t_min, t_max, "before_row_click", results)
        
        # 4. Open Trial
        step_start_trial = time.perf_counter()
        await click_dataframe_row(page, row_index, user_id, args.timeout)
        results["timings"]["open_trial_seconds"] = time.perf_counter() - step_start_trial

        await maybe_think(user_id, user_rng, t_min, t_max, "after_trial_open", results)

        if scenario == "basic":
            results["success"] = True
            log(user_id, "Basic scenario complete.")
            return results

        # 5. Prediction or Simulation
        if scenario == "simulation-block":
            step_start_sim = time.perf_counter()
            await maybe_think(user_id, user_rng, t_min, t_max, "before_sim_on", results)
            await toggle_simulation_mode_on(page, user_id)
            
            log(user_id, "Clicking 'Predict Trial Completion' (Guardrail Test)...")
            predict_btn = page.get_by_role("button", name="Predict Trial Completion")
            await predict_btn.click()
            if not await detect_simulation_block(page, user_id, args.timeout):
                raise Exception("Simulation mode ON but no guardrail detected.")
            
            log(user_id, "Phase 1 (Block) passed. Turning Simulation Mode OFF...")
            await maybe_think(user_id, user_rng, t_min, t_max, "before_sim_off", results)
            await toggle_simulation_mode_off(page, user_id)
            
            log(user_id, "Clicking 'Predict Trial Completion' (Real Prediction Test)...")
            predict_btn = page.get_by_role("button", name="Predict Trial Completion")
            await predict_btn.wait_for(state="visible")
            await predict_btn.click()
            
            await wait_for_prediction_result(page, user_id, args.timeout)
            log(user_id, "Phase 2 (Success) passed.")
            results["timings"]["simulation_block_seconds"] = time.perf_counter() - step_start_sim
            results["success"] = True
        else:
            log(user_id, "Triggering prediction...")
            step_start_pred = time.perf_counter()
            predict_btn = page.get_by_role("button", name="Predict Trial Completion")
            await predict_btn.click()
            await wait_for_prediction_result(page, user_id, args.timeout)
            results["timings"]["prediction_seconds"] = time.perf_counter() - step_start_pred
            results["success"] = True

    except Exception as e:
        results["error"] = str(e)
        log(user_id, f"FINAL ERROR: {e}")
        try:
            SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
            ss_path = SCREENSHOT_DIR / f"user_{user_id:02d}_failure.png"
            await page.screenshot(path=str(ss_path))
            results["screenshot"] = str(ss_path)
            dom_path = SCREENSHOT_DIR / f"user_{user_id:02d}_failure.html"
            with open(dom_path, "w") as f:
                f.write(await page.content())
            results["dom_snapshot"] = str(dom_path)
        except: pass
    finally:
        results["ended_at"] = datetime.now().isoformat()
        results["timings"]["total_seconds"] = time.perf_counter() - start_time
        if context: await context.close()
    return results

# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

async def main():
    global OUTPUT_DIR, SCREENSHOT_DIR
    parser = argparse.ArgumentParser(description="CTPredict UI Load Test")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"UI URL")
    parser.add_argument("--users", type=int, default=DEFAULT_USERS, help=f"Simulated users")
    parser.add_argument("--ramp-seconds", type=float, default=DEFAULT_RAMP, help=f"Ramp seconds")
    parser.add_argument("--headful", action="store_true", help="Run headful")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout ms")
    parser.add_argument("--scenario", choices=VALID_SCENARIOS, default="prediction", help="Scenario for all users")
    parser.add_argument("--scenario-mix", help="Mixed scenarios: basic=2,prediction=5,simulation-block=3")
    parser.add_argument("--no-shuffle-scenarios", action="store_true", help="Disable scenario shuffling")
    parser.add_argument("--random-seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--row-index", type=int, default=0, help="Fixed row index to select")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output dir")
    
    # Humanization / Randomization CLI Options
    parser.add_argument("--humanize", action="store_true", help="Enable randomized row, viewport, think time, and jitter")
    parser.add_argument("--randomize-rows", action="store_true", help="Enable randomized row selection")
    parser.add_argument("--row-index-min", type=int, default=0)
    parser.add_argument("--row-index-max", type=int, default=4)
    parser.add_argument("--randomize-viewports", action="store_true", help="Enable randomized desktop viewports")
    parser.add_argument("--think-time-min", type=float, default=0.0)
    parser.add_argument("--think-time-max", type=float, default=0.0)
    parser.add_argument("--start-jitter-seconds", type=float, default=0.0)
    
    args = parser.parse_args()
    
    # Apply humanize defaults
    if args.humanize:
        if not args.randomize_rows: args.randomize_rows = True
        if not args.randomize_viewports: args.randomize_viewports = True
        if args.think_time_max == 0.0:
            args.think_time_min = 0.3
            args.think_time_max = 1.5
        if args.start_jitter_seconds == 0.0:
            args.start_jitter_seconds = 1.0

    # Validate row range
    if args.row_index_min < 0:
        print(f"Error: --row-index-min must be >= 0 (got {args.row_index_min})")
        return
    if args.row_index_max < args.row_index_min:
        print(f"Error: --row-index-max ({args.row_index_max}) must be >= --row-index-min ({args.row_index_min})")
        return

    OUTPUT_DIR = Path(args.output_dir)
    SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Reproducible Random State
    master_rng = random.Random(args.random_seed)
    
    # 2. Parse Scenarios
    user_scenarios = []
    if args.scenario_mix:
        try:
            mix_parts = args.scenario_mix.split(",")
            for part in mix_parts:
                s_name, s_count = part.split("=")
                s_name = s_name.strip()
                if s_name not in VALID_SCENARIOS:
                    raise ValueError(f"Invalid scenario name: {s_name}. Valid names are: {VALID_SCENARIOS}")
                user_scenarios.extend([s_name] * int(s_count))
            if len(user_scenarios) != args.users:
                raise ValueError(f"Scenario mix total ({len(user_scenarios)}) does not match --users ({args.users})")
            if not args.no_shuffle_scenarios:
                master_rng.shuffle(user_scenarios)
        except Exception as e:
            print(f"Error parsing --scenario-mix: {e}")
            return
    else:
        user_scenarios = [args.scenario] * args.users

    # 3. Precompute per-user configurations
    user_configs = []
    for i in range(args.users):
        user_id = i + 1
        
        # Row selection
        row_idx = args.row_index
        if args.randomize_rows:
            row_idx = master_rng.randint(args.row_index_min, args.row_index_max)
            
        # Viewport
        vport = {"width": 1440, "height": 900}
        if args.randomize_viewports:
            vport = master_rng.choice(VIEWPORT_PRESETS)
            
        # Jitter
        jitter = 0.0
        if args.start_jitter_seconds > 0:
            jitter = master_rng.uniform(0, args.start_jitter_seconds)
            
        user_configs.append({
            "scenario": user_scenarios[i],
            "row_index": row_idx,
            "viewport": vport,
            "start_jitter_seconds": jitter,
            "think_time_min": args.think_time_min,
            "think_time_max": args.think_time_max
        })

    print("\n" + "="*70)
    print(f"CTPredict LOAD TEST: {args.users} users")
    if args.humanize: print("MODE:         HUMANIZED (Randomized row, viewport, think, jitter)")
    if args.scenario_mix: print(f"Scenario Mix: {args.scenario_mix}")
    else: print(f"Scenario:     {args.scenario}")
    print(f"Ramp:         {args.ramp_seconds}s")
    print(f"Target URL:   {args.url}")
    if args.random_seed is not None: print(f"Random Seed:  {args.random_seed}")
    print("="*70 + "\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not args.headful)
        tasks = []
        for i in range(args.users):
            user_id = i + 1
            # Base ramp delay
            base_delay = 0 if args.ramp_seconds == 0 else (args.ramp_seconds / args.users) * i
            tasks.append(asyncio.create_task(run_user_lifecycle(browser, user_id, user_configs[i], args, base_delay)))
        all_results = await asyncio.gather(*tasks)
        await browser.close()

    # SUMMARY
    successes = [r for r in all_results if r["success"]]
    failures = [r for r in all_results if not r["success"]]
    def avg_t(key):
        vals = [r["timings"].get(key, 0) for r in successes if key in r["timings"]]
        return sum(vals) / len(vals) if vals else 0

    print("\n" + "="*70)
    print("LOAD TEST FINAL SUMMARY")
    print("="*70)
    print(f"Users Launched:      {args.users}")
    print(f"Successes:           {len(successes)}")
    print(f"Failures:            {len(failures)}")
    print(f"Success Rate:        {(len(successes)/args.users)*100:.1f}%")
    print("-" * 35)
    
    print("Scenario mix breakdown:")
    for s_name in sorted(set(user_scenarios)):
        s_launched = user_scenarios.count(s_name)
        s_success = len([r for r in successes if r["scenario"] == s_name])
        s_failed = len([r for r in failures if r["scenario"] == s_name])
        print(f"  {s_name:17}: {s_launched} launched, {s_success} success, {s_failed} failed")
    
    print("-" * 35)
    if args.randomize_rows: print(f"Rows Picked:         Range {args.row_index_min} to {args.row_index_max}")
    if args.randomize_viewports: print(f"Viewports:           Randomized desktop presets")
    if args.start_jitter_seconds > 0: print(f"Start Jitter:        Enabled (max {args.start_jitter_seconds}s)")
    if args.think_time_max > 0: print(f"Avg Think Time:      {avg_t('total_think_time_seconds'):.2f}s")
    print("-" * 35)
    print(f"Avg App Open:        {avg_t('open_app_seconds'):.2f}s")
    print(f"Avg Search Click:    {avg_t('search_seconds'):.2f}s")
    print(f"Avg Grid Wait:       {avg_t('grid_wait_seconds'):.2f}s")
    print(f"Avg Trial Open:      {avg_t('open_trial_seconds'):.2f}s")
    print(f"Avg Prediction:      {avg_t('prediction_seconds'):.2f}s")
    if "simulation-block" in user_scenarios:
        print(f"Avg Sim Block:       {avg_t('simulation_block_seconds'):.2f}s")
    print(f"Avg Total Duration:  {avg_t('total_seconds'):.2f}s")
    print("-" * 70)
    
    if failures:
        print("FAILED USERS:")
        for f in failures:
            print(f"  User {f['user_id']:02d} ({f['scenario']}): {f['error']}")
    else:
        print("PERFECT RUN: All users completed the flow.")
    
    with open(OUTPUT_DIR / "load_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    df_results = pd.DataFrame([
        {
            "user_id": r["user_id"], "scenario": r["scenario"], "success": r["success"], 
            "row_index": r["row_index"], 
            "viewport": f"{r['viewport_width']}x{r['viewport_height']}",
            "viewport_width": r["viewport_width"],
            "viewport_height": r["viewport_height"],
            "base_ramp_delay_seconds": r["base_ramp_delay_seconds"],
            "start_jitter_seconds": r["start_jitter_seconds"],
            "total_start_delay_seconds": r["total_start_delay_seconds"],
            "think_time": r["timings"].get("total_think_time_seconds", 0),
            "total": r["timings"].get("total_seconds"),
            "app_open": r["timings"].get("open_app_seconds"),
            "search": r["timings"].get("search_seconds"),
            "grid_wait": r["timings"].get("grid_wait_seconds"),
            "trial_open": r["timings"].get("open_trial_seconds"),
            "prediction": r["timings"].get("prediction_seconds"),
            "sim_block": r["timings"].get("simulation_block_seconds"),
            "error": r["error"]
        } for r in all_results
    ])
    df_results.to_csv(OUTPUT_DIR / "load_test_results.csv", index=False)
    print(f"\nFiles saved to {OUTPUT_DIR}/")
    print("="*70 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
