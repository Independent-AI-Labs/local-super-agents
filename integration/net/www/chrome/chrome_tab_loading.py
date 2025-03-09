import time
import threading
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, WebDriverException
from selenium.webdriver.common.by import By

from integration.data.config import DEFAULT_WEB_PAGE_TIMEOUT


def wait_for_page_load(driver, timeout=DEFAULT_WEB_PAGE_TIMEOUT):
    """
    Wait for page to complete loading using document.readyState.

    Args:
        driver: WebDriver instance
        timeout: Maximum time to wait in seconds

    Returns:
        bool: True if page loaded successfully, False otherwise
    """
    try:
        # First wait for the document to reach 'interactive' or 'complete' state
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script("return document.readyState") in ["interactive", "complete"]
        )

        # Then wait for the document to fully complete loading
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )

        # Also wait for any jQuery or AJAX requests to complete if present
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script(
                "return (typeof jQuery === 'undefined' || jQuery.active === 0) && "
                "(typeof Angular === 'undefined' || Angular.element(document).injector() === undefined || "
                "!Angular.element(document).injector().get('$http').pendingRequests.length)"
            )
        )

        return True
    except TimeoutException:
        print("Page load timed out, continuing with partial content")
        return False
    except Exception as e:
        print(f"Error while waiting for page load: {str(e)}")
        return False


def check_for_loading_indicators(driver):
    """
    Check for common loading indicators on the page.

    Args:
        driver: WebDriver instance

    Returns:
        bool: True if loading indicators are no longer present
    """
    try:
        # Common loading indicator selectors
        loading_selectors = [
            "//div[contains(@class, 'loading')]",
            "//div[contains(@class, 'spinner')]",
            "//div[contains(@class, 'loader')]",
            "//div[contains(@id, 'loading')]",
            "//div[contains(@id, 'spinner')]"
        ]

        # Check if any loading indicators are still visible
        for selector in loading_selectors:
            elements = driver.find_elements(By.XPATH, selector)
            for element in elements:
                if element.is_displayed():
                    return False

        return True
    except (StaleElementReferenceException, WebDriverException):
        # If we can't check, assume page is loaded
        return True


def wait_for_page_load_with_circuit_breaker(driver, max_wait_time=30, check_interval=0.5):
    """
    Wait for page load with circuit breaker to prevent hanging.

    Args:
        driver: WebDriver instance
        max_wait_time: Maximum wait time in seconds
        check_interval: How often to check loading status

    Returns:
        bool: True if page loaded successfully, False if timed out
    """
    start_time = time.time()

    # First wait for basic document.readyState
    initial_load_success = wait_for_page_load(driver, min(10, max_wait_time / 2))

    # If initial load successful, continue with more detailed checks
    if initial_load_success:
        # Track consecutive stable states
        consecutive_stable_counts = 0
        required_stable_counts = 3  # Require multiple consecutive stable checks

        while time.time() - start_time < max_wait_time:
            # Check if page appears stable
            dom_size_before = len(driver.page_source)
            time.sleep(check_interval)

            # Check if DOM size has stabilized (no significant changes)
            dom_size_after = len(driver.page_source)
            dom_change_percent = abs(dom_size_after - dom_size_before) / max(dom_size_before, 1) * 100

            # Check loading indicators
            no_loading_indicators = check_for_loading_indicators(driver)

            # If DOM is stable and no loading indicators
            if dom_change_percent < 1 and no_loading_indicators:
                consecutive_stable_counts += 1
                if consecutive_stable_counts >= required_stable_counts:
                    print(f"Page load detected after {time.time() - start_time:.2f} seconds")
                    return True
            else:
                consecutive_stable_counts = 0

            # Circuit breaker: check if we're spending too much time
            elapsed = time.time() - start_time
            if elapsed > max_wait_time * 0.8:  # 80% of max time
                print(f"Circuit breaker triggered after {elapsed:.2f} seconds")
                return False

    print(f"Page load timed out after {time.time() - start_time:.2f} seconds")
    return False


def scroll_with_monitoring(driver, max_scroll_time, state_dict):
    """
    Scroll the page with monitoring and adaptive pausing.

    Args:
        driver: WebDriver instance
        max_scroll_time: Maximum time to spend scrolling
        state_dict: Shared state dictionary for circuit breaker
    """
    start_time = time.time()
    last_height = driver.execute_script("return document.body.scrollHeight")

    # Track scroll progress
    scroll_count = 0
    max_scroll_count = 10  # Maximum number of scroll attempts
    consecutive_unchanged = 0

    try:
        while time.time() - start_time < max_scroll_time and scroll_count < max_scroll_count:
            # Check if we should abort processing
            if state_dict.get("abort_processing", False):
                print("Aborting scroll due to circuit breaker")
                break

            # Scroll down
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            scroll_count += 1

            # Adaptive wait based on site responsiveness
            wait_time = 0.2  # Start with a short wait

            # Wait for page to stabilize or continue after max wait
            stability_start = time.time()
            while time.time() - stability_start < 1.0:  # Max 1 second per scroll
                # Check if we should abort processing
                if state_dict.get("abort_processing", False):
                    return

                # Check if new content has loaded by comparing heights
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    # No new content, so we can move on
                    break

                last_height = new_height
                time.sleep(wait_time)
                wait_time = min(wait_time * 1.5, 0.5)  # Increase wait time up to 0.5s

            # Check if we should continue scrolling
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                # If we've reached the bottom or content isn't changing
                consecutive_unchanged += 1
                if consecutive_unchanged >= 2:
                    print(f"Scroll complete: No new content after {scroll_count} scrolls")
                    break
            else:
                consecutive_unchanged = 0
                last_height = new_height

    except Exception as e:
        print(f"Error during scrolling: {e}")


def handle_cookie_prompts(driver):
    """
    Attempt to automatically accept cookie prompts.

    Args:
        driver: WebDriver instance
    """
    # Common cookie acceptance button patterns
    cookie_button_patterns = [
        "//button[contains(., 'Accept')]",
        "//button[contains(., 'Accept All')]",
        "//button[contains(., 'I Accept')]",
        "//button[contains(., 'Allow')]",
        "//button[contains(., 'Allow All')]",
        "//a[contains(., 'Accept')]",
        "//div[contains(., 'Accept')][contains(@class, 'button')]",
        "//div[contains(., 'Accept')][contains(@role, 'button')]",
        "//button[contains(@id, 'accept')]",
        "//button[contains(@class, 'accept')]",
        "//button[contains(@class, 'cookie-accept')]"
    ]

    for pattern in cookie_button_patterns:
        try:
            buttons = driver.find_elements(By.XPATH, pattern)
            for button in buttons:
                if button.is_displayed():
                    button.click()
                    time.sleep(0.5)  # Wait briefly for the prompt to disappear
                    return
        except Exception:
            continue