"""Streamlit Community Cloud Keep-Awake Script (Selenium).

Opens the deployed Streamlit dashboard in headless Chrome to maintain an active
session and prevent Streamlit Community Cloud from putting the app to sleep after
inactivity (~12 hours).

If the app is already asleep, it detects and clicks the 'Yes, get this app back up'
wake button across both top-level DOM and iframes, waits for container boot and
rendering, verifies the dashboard content, and holds the connection open to record
active session telemetry.

Environment Variables:
  APP_URL      URL of the deployed app (default: https://rajasthanroyals.streamlit.app/)
  EXPECT_TEXT  Text identifying rendered dashboard (default: 'Rinku Singh')
  TIMEOUT_S    Maximum seconds to wait for boot & render (default: 240)
"""
from __future__ import annotations

import os
import sys
import time

APP_URL = os.environ.get("APP_URL", "https://rajasthanroyals.streamlit.app/").strip()
EXPECT_TEXT = os.environ.get("EXPECT_TEXT", "Rinku Singh").strip()
TIMEOUT_S = int(os.environ.get("TIMEOUT_S", "240"))

# Keywords to match the wake-up button on Streamlit Community Cloud
WAKE_BUTTON_KEYWORDS = [
    "get this app back up",
    "wake up",
    "wake",
    "yes, get this app back up",
]

# Multiple indicators that confirm the dashboard has successfully loaded
CORE_INDICATORS = [
    EXPECT_TEXT.lower(),
    "rinku singh",
    "who delivers when it matters",
    "superr selector",
    "josh hazlewood",
    "the final rankings",
    "clean matches",
]


def init_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    # 1. Prefer native Selenium Manager (built into Selenium 4.10+)
    try:
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(120)
        return driver
    except Exception as e:
        print(f"[Keep-Awake] Native Chrome init note: {e}. Trying ChromeDriverManager fallback...")

    # 2. Fallback to webdriver_manager if needed
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(120)
        return driver
    except Exception as e2:
        print(f"[Keep-Awake] Fallback Chrome init failed: {e2}")
        raise


def find_and_click_wake_button(driver) -> bool:
    from selenium.webdriver.common.by import By

    def check_context():
        # Search all button, link, and role=button elements
        candidates = driver.find_elements(By.XPATH, "//button | //a | //*[@role='button']")
        for el in candidates:
            try:
                text = (el.text or "").strip().lower()
                aria = (el.get_attribute("aria-label") or "").strip().lower()
                for kw in WAKE_BUTTON_KEYWORDS:
                    if kw in text or kw in aria:
                        driver.execute_script("arguments[0].scrollIntoView(true);", el)
                        time.sleep(0.5)
                        try:
                            el.click()
                        except Exception:
                            driver.execute_script("arguments[0].click();", el)
                        return True
            except Exception:
                continue

        # Case-insensitive XPath fallback
        for kw in WAKE_BUTTON_KEYWORDS:
            xpath = (
                f"//*[(self::button or self::a or @role='button') and "
                f"contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{kw}')]"
            )
            elements = driver.find_elements(By.XPATH, xpath)
            if elements:
                try:
                    driver.execute_script("arguments[0].click();", elements[0])
                    return True
                except Exception:
                    continue
        return False

    # Check top-level window
    driver.switch_to.default_content()
    if check_context():
        return True

    # Check inside each iframe (Streamlit viewer runs in an iframe)
    iframes = driver.find_elements(By.TAG_NAME, "iframe")
    for frame in iframes:
        try:
            driver.switch_to.frame(frame)
            if check_context():
                driver.switch_to.default_content()
                return True
        except Exception:
            pass
        finally:
            driver.switch_to.default_content()

    return False


def is_app_rendered(driver) -> tuple[bool, str]:
    from selenium.webdriver.common.by import By

    # 1. Check top-level page
    driver.switch_to.default_content()
    source = (driver.page_source or "").lower()
    for ind in CORE_INDICATORS:
        if ind in source:
            return True, ind

    # 2. Check Streamlit container markers
    markers = ["stAppViewContainer", "stApp", "stHeader"]
    for marker in markers:
        try:
            els = driver.find_elements(By.CLASS_NAME, marker)
            for el in els:
                t = (el.text or "").lower()
                for ind in CORE_INDICATORS:
                    if ind in t:
                        return True, f"{marker}:{ind}"
        except Exception:
            pass

    # 3. Check inside each iframe
    iframes = driver.find_elements(By.TAG_NAME, "iframe")
    for frame in iframes:
        try:
            driver.switch_to.frame(frame)
            frame_source = (driver.page_source or "").lower()
            for ind in CORE_INDICATORS:
                if ind in frame_source:
                    driver.switch_to.default_content()
                    return True, f"iframe:{ind}"
        except Exception:
            pass
        finally:
            driver.switch_to.default_content()

    return False, ""


def main() -> int:
    if not APP_URL.startswith("http"):
        print(f"[Keep-Awake] Fatal Error: Invalid APP_URL: '{APP_URL}'")
        return 1

    print("=" * 60)
    print("Streamlit Keep-Awake Monitor")
    print(f"Target URL:    {APP_URL}")
    print(f"Expected Text: '{EXPECT_TEXT}'")
    print(f"Timeout:       {TIMEOUT_S}s")
    print("=" * 60)

    driver = None
    try:
        driver = init_driver()
        print(f"[Keep-Awake] Headless Chrome initialized. Navigating to {APP_URL}...")
        driver.get(APP_URL)
        time.sleep(5)

        start_time = time.time()
        deadline = start_time + TIMEOUT_S
        woke = False

        while time.time() < deadline:
            rendered, matched_marker = is_app_rendered(driver)
            if rendered:
                elapsed = int(time.time() - start_time)
                print(f"[Keep-Awake] SUCCESS: App verified live (matched '{matched_marker}') in {elapsed}s.")
                break

            # If not yet awake, look for the wake button
            if not woke and find_and_click_wake_button(driver):
                woke = True
                print("[Keep-Awake] App was asleep: clicked 'Wake up' button. Booting container...")

            time.sleep(5)
        else:
            elapsed = int(time.time() - start_time)
            print(f"[Keep-Awake] TIMEOUT: App did not render within {elapsed}s.")
            screenshot_path = "keep_awake_failure.png"
            try:
                driver.save_screenshot(screenshot_path)
                print(f"[Keep-Awake] Diagnostic screenshot saved to {screenshot_path}")
            except Exception as e:
                print(f"[Keep-Awake] Could not save screenshot: {e}")
            
            # Print page title and snippet for logs
            try:
                print(f"[Keep-Awake] Current Title: {driver.title}")
                driver.switch_to.default_content()
                snippet = (driver.page_source or "")[:500]
                print(f"[Keep-Awake] Page snippet:\n{snippet}")
            except Exception:
                pass
            return 1

        # Keep session active briefly to ensure Streamlit records active traffic
        print("[Keep-Awake] Holding session open for 15s to establish active connection...")
        time.sleep(15)
        status = "App was asleep and successfully woken up." if woke else "App was already awake."
        print(f"[Keep-Awake] COMPLETED: {status}")
        return 0

    except Exception as exc:
        print(f"[Keep-Awake] Unexpected runtime exception: {exc}")
        if driver:
            try:
                driver.save_screenshot("keep_awake_failure.png")
            except Exception:
                pass
        return 1
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
