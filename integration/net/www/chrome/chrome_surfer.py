import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from urllib.parse import urlparse

from selenium import webdriver
from selenium.common.exceptions import TimeoutException

from integration.data.config import SEARXNG_BASE_URL, CHROME_PATH, EXTENSIONS_PATH, TOP_N_WEB_SEARCH_RESULTS, MAX_WEB_SCRAPE_WORKERS_PER_SEARCH
from integration.net.util.semanting_filtering import digest_scraped_data, extract_web_content_with_semantics, format_semantic_content
from integration.net.util.web_util import get_request, extract_urls_and_common_words, categorize_links
from integration.net.www.chrome.chrome_driver_pool import ChromeDriverPool
# Import the improved tab loading utilities
from integration.net.www.chrome.chrome_tab_loading import (
    wait_for_page_load_with_circuit_breaker,
    scroll_with_monitoring,
    handle_cookie_prompts
)
from integration.util.misc_util import get_indexed_search_results_path

# Initialize the pool once
CHROME_DRIVER_POOL = ChromeDriverPool()

# Default maximum wait time for tabs in seconds
DEFAULT_MAX_TAB_WAIT_TIME = 30

# Configure circuit breaker thresholds
MAX_RETRIES = 2
CIRCUIT_BREAKER_RESET_TIME = 300  # 5 minutes
FAILURE_THRESHOLD = 3

# Global circuit breaker state
_CIRCUIT_BREAKER = {
    "failures": {},  # Tracks failures by domain
    "open_circuits": set(),  # Set of domains with open circuit breakers
    "last_reset_time": time.time()
}
_CIRCUIT_BREAKER_LOCK = threading.RLock()


def init_web_driver():
    # [Existing code remains unchanged]
    options = webdriver.ChromeOptions()
    options.binary_location = CHROME_PATH

    # Commented, so agent activity can be monitored / visualized.
    # options.add_argument('--headless')  # Run in headless mode
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-infobars')
    options.add_argument('--disable-gpu')
    options.add_argument(
        '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36')

    # DANGER ZONE
    # Access all sites, ignore all TLS / HTTPS configuration errors and warnings.
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--allow-insecure-localhost')
    options.add_argument('--ignore-urlfetcher-cert-requests')
    options.add_argument('--disable-net-security')

    # Configure experimental options to disable webdriver detection.
    options.add_experimental_option('useAutomationExtension', False)
    options.add_experimental_option('excludeSwitches', ['enable-automation'])

    # Load the NopeCHA extension.
    options.add_extension(os.path.join(EXTENSIONS_PATH, 'chrome', 'nopecha.crx'))
    # Cookies are good!
    options.add_extension(os.path.join(EXTENSIONS_PATH, 'chrome', 'accept-all-cookies.crx'))

    driver = webdriver.Chrome(options=options)

    # driver.get(f"https://nopecha.com/setup#{NOPECHA_KEY}")

    return driver


def search_web(
        search_terms: list,
        semantic_patterns: List[str] | None = None,
        instructions: str = None,
        max_workers: int = 8,
        transient: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Perform web searches for the given search terms and filter the results.
    Indexed search results are kept for this query by default. Set transient to 'True' to disable this behaviour.

    Args:
        search_terms: List of search terms to query
        semantic_patterns: Optional list of semantic patterns to filter results
        instructions: Optional specific natural language instructions
        max_workers: Maximum number of parallel worker threads
        transient: Whether to keep indexed data for this search
    Returns:
        Tuple[List[str], List[str]]: Filtered data and discovered patterns
    """
    # [Existing code remains unchanged]
    # Generate hierarchical directory path based on vectorized inputs
    results_dir = get_indexed_search_results_path(search_terms, semantic_patterns, instructions, transient)
    os.makedirs(results_dir, exist_ok=True)

    filtered_data = []
    discovered_patterns = []

    try:
        # Perform web searches in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(single_search, term, TOP_N_WEB_SEARCH_RESULTS) for term in search_terms]
            results = [future.result() for future in as_completed(futures)]

        # Extract all search results and common words
        all_search_results = []
        for result in results:
            search_result, top_common_words = result
            all_search_results.extend(search_result)

            # Add unique common words to discovered patterns
            for word in top_common_words:
                if semantic_patterns is None or word not in semantic_patterns:
                    if word not in discovered_patterns:
                        discovered_patterns.append(word)

        if not transient:
            for idx, search_result in enumerate(all_search_results):
                file_path = os.path.join(results_dir, f'search_result_{idx}_raw.txt')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(search_result)

        # If no semantic patterns are provided, use top discovered patterns
        patterns_to_use = semantic_patterns if semantic_patterns else discovered_patterns

        filtered_data = digest_scraped_data(
            all_search_results,
            patterns_to_use,
            instructions
        )

        if not transient:
            for idx, search_result in enumerate(filtered_data):
                file_path = os.path.join(results_dir, f'search_result_{idx}_digested.txt')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(search_result)

    except Exception as e:
        # Log the error and return empty results
        print(f"Error in search_web: {str(e)}")
        filtered_data = [f"Error during web search: {str(e)}"]

    return filtered_data, discovered_patterns


def single_search(search_term: str, max_results: int = 3, max_workers: int = 2) -> Tuple[List[str], List[str]]:
    # [Existing code remains unchanged]
    results = get_request(f"{SEARXNG_BASE_URL}/search?q={search_term}", {}, {})
    urls_to_scrape, most_common_words = extract_urls_and_common_words(results)
    return scrape_urls(urls_to_scrape[:max_results], max_workers), most_common_words


def check_circuit_breaker(url: str) -> bool:
    """
    Check if a circuit breaker is active for the given URL's domain.

    Args:
        url: URL to check

    Returns:
        bool: True if circuit is closed (requests allowed), False if open (blocked)
    """
    domain = urlparse(url).netloc

    with _CIRCUIT_BREAKER_LOCK:
        # Reset circuit breakers if enough time has passed
        current_time = time.time()
        if current_time - _CIRCUIT_BREAKER["last_reset_time"] > CIRCUIT_BREAKER_RESET_TIME:
            _CIRCUIT_BREAKER["open_circuits"].clear()
            _CIRCUIT_BREAKER["failures"] = {}
            _CIRCUIT_BREAKER["last_reset_time"] = current_time

        # Check if circuit is open for this domain
        if domain in _CIRCUIT_BREAKER["open_circuits"]:
            print(f"Circuit breaker open for {domain}, skipping request")
            return False

    return True


def record_failure(url: str):
    """
    Record a failure for the given URL's domain and potentially open circuit breaker.

    Args:
        url: URL that failed
    """
    # Extract domain from URL
    domain = urlparse(url).netloc

    with _CIRCUIT_BREAKER_LOCK:
        # Increment failure count
        if domain not in _CIRCUIT_BREAKER["failures"]:
            _CIRCUIT_BREAKER["failures"][domain] = 1
        else:
            _CIRCUIT_BREAKER["failures"][domain] += 1

        # Open circuit if threshold reached
        if _CIRCUIT_BREAKER["failures"][domain] >= FAILURE_THRESHOLD:
            _CIRCUIT_BREAKER["open_circuits"].add(domain)
            print(f"Circuit breaker tripped for {domain} after {FAILURE_THRESHOLD} failures")


def scrape_urls(urls, max_tab_wait_time: int = DEFAULT_MAX_TAB_WAIT_TIME,
                max_workers: int = MAX_WEB_SCRAPE_WORKERS_PER_SEARCH):
    """
    Scrape multiple URLs in batches using a Chrome driver from the pool.
    Processes URLs in batches limited by max_workers to avoid opening too many tabs.

    Args:
        urls: List of URLs to scrape
        max_tab_wait_time: Maximum time to wait for a tab to load and process in seconds
        max_workers: Maximum number of tabs to open at once

    Returns:
        List[str]: Scraped content from each URL
    """
    if not urls:
        return []

    # Filter URLs through circuit breaker
    filtered_urls = [url for url in urls if check_circuit_breaker(url)]
    if not filtered_urls:
        return ["🠶 All URLs were blocked by circuit breaker due to previous failures"]

    all_results = []

    # Process URLs in batches
    for i in range(0, len(filtered_urls), max_workers):
        batch_urls = filtered_urls[i:i + max_workers]
        print(f"Processing batch {i // max_workers + 1} with {len(batch_urls)} URLs")

        # Get a driver from the pool
        driver = CHROME_DRIVER_POOL.get_driver()

        try:
            # Process this batch
            batch_results = process_url_batch(driver, batch_urls, max_tab_wait_time)
            all_results.extend(batch_results)

        finally:
            # Always return the driver to the pool
            CHROME_DRIVER_POOL.return_driver(driver)

    return all_results


def process_url_batch(driver, urls, max_tab_wait_time):
    """
    Process a batch of URLs with a single driver instance.

    Args:
        driver: WebDriver instance
        urls: List of URLs to process in this batch
        max_tab_wait_time: Maximum time to wait for each tab

    Returns:
        List[str]: Results for this batch of URLs
    """
    # Remember the initial tab/window handle
    initial_handle = driver.current_window_handle
    initial_handles = set(driver.window_handles)

    results = []
    tab_info = {}  # Will store {handle: url} mapping

    # Use a lock for thread safety
    driver_lock = threading.RLock()

    # Track overall process timing
    batch_start_time = time.time()
    batch_timeout = max_tab_wait_time * len(urls) * 0.8  # Scale with batch size but limit

    try:
        # PHASE 1: Open all tabs for this batch
        print(f"Opening {len(urls)} tabs in this batch")
        for index, url in enumerate(urls):
            try:
                tab_name = f"tab{index}"
                print(f"Opening tab for {url}")
                with driver_lock:
                    driver.execute_script(f"window.open('{url}', '{tab_name}');")
            except Exception as e:
                print(f"Error opening tab for {url}: {e}")
                record_failure(url)
                results.append(f"🠶 ERROR: Failed to open tab for URL={url}: {str(e)}\n\n")

        # Allow a short time for tabs to start loading
        time.sleep(0.5)

        # Get all new tab handles
        with driver_lock:
            all_handles = set(driver.window_handles)
            new_handles = list(all_handles - initial_handles)

        # Associate handles with URLs
        for i, handle in enumerate(new_handles):
            if i < len(urls):
                tab_info[handle] = urls[i]

        # PHASE 2: Process each tab
        for handle in new_handles[:]:  # Create a copy to safely modify during iteration
            # Check for batch timeout
            if time.time() - batch_start_time > batch_timeout:
                remaining = [h for h in new_handles if h not in [initial_handle] and h in driver.window_handles]
                results.append(f"🠶 BATCH TIMEOUT: Skipped {len(remaining)} URLs due to batch timeout\n\n")
                break

            # Process this tab
            try:
                with driver_lock:
                    # Check if handle is still valid
                    if handle not in driver.window_handles:
                        continue

                    driver.switch_to.window(handle)
                    current_url = tab_info.get(handle, driver.current_url)

                print(f"Processing {current_url}")

                # Process tab with proper monitoring
                try:
                    from integration.net.util.semanting_filtering import extract_web_content_with_semantics, format_semantic_content
                    from integration.net.util.web_util import categorize_links

                    # Track processing state
                    state = {
                        "loading_complete": False,
                        "scrolling_complete": False,
                        "content_extracted": False,
                        "abort_processing": False
                    }

                    # Set up circuit breaker
                    timeout_timer = threading.Timer(max_tab_wait_time * 0.9,
                                                    lambda: state.update({"abort_processing": True}))
                    timeout_timer.daemon = True
                    timeout_timer.start()

                    try:
                        # Wait for page to load
                        load_success = wait_for_page_load_with_circuit_breaker(driver, max_tab_wait_time * 0.6)
                        state["loading_complete"] = True

                        # Check for abort signal
                        if state["abort_processing"]:
                            results.append(f"🠶 TIMEOUT: Processing of URL={current_url} aborted by circuit breaker\n\n")
                            continue

                        # Scroll to load lazy content
                        scroll_thread = threading.Thread(target=scroll_with_monitoring,
                                                         args=(driver, max_tab_wait_time * 0.3, state))
                        scroll_thread.daemon = True
                        scroll_thread.start()
                        scroll_thread.join(max_tab_wait_time * 0.3)
                        state["scrolling_complete"] = True

                        # Get page source
                        page_source = driver.page_source

                        # Handle cookie/CAPTCHA prompts
                        if ("verify" in page_source.lower() and "human" in page_source.lower()) or (
                                "accept" in page_source.lower() and "cookies" in page_source.lower()):
                            handle_cookie_prompts(driver)
                            page_source = driver.page_source

                        # Extract links
                        links = categorize_links(current_url, page_source)

                        # Extract content with fallbacks
                        try:
                            semantic_data = extract_web_content_with_semantics(page_source, current_url)
                            formatted_content = format_semantic_content(semantic_data)
                        except Exception as extraction_error:
                            print(f"Falling back to basic extraction: {type(extraction_error).__name__}")
                            from integration.net.util.web_util import extract_relevant_content
                            readable_text = extract_relevant_content(page_source)
                            sparse_text = re.sub(r'\n+', '\n', readable_text).strip()
                            formatted_content = sparse_text

                        # Mark extraction as complete
                        state["content_extracted"] = True

                        # Add result
                        tab_result = (
                            f"🠶 WEBSITE CONTENTS FOR URL={current_url}:\n\n"
                            f"{formatted_content}\n\n"
                            f"🠶 LINKS FROM WEBSITE:\n{links}\n\n"
                            f"🠶 END OF WEBSITE CONTENTS FOR URL={current_url}\n\n"
                        )
                        results.append(tab_result)

                    finally:
                        # Clean up timer
                        if timeout_timer.is_alive():
                            timeout_timer.cancel()

                except TimeoutException:
                    print(f"Timeout processing {current_url}")
                    record_failure(current_url)
                    results.append(f"🠶 TIMEOUT: Processing URL={current_url} exceeded {max_tab_wait_time} seconds\n\n")
                except Exception as e:
                    print(f"Error processing {current_url}: {str(e)}")
                    record_failure(current_url)
                    results.append(f"🠶 ERROR: Processing URL={current_url} failed: {str(e)}\n\n")

            except Exception as tab_error:
                print(f"Error handling tab: {str(tab_error)}")
                url = tab_info.get(handle, "unknown")
                results.append(f"🠶 ERROR: Tab handling failed for URL={url}: {str(tab_error)}\n\n")

            finally:
                # Always try to close the tab
                try:
                    with driver_lock:
                        if handle in driver.window_handles:
                            driver.switch_to.window(handle)
                            driver.close()
                except Exception as close_error:
                    print(f"Error closing tab: {str(close_error)}")

        # Switch back to the initial tab
        try:
            with driver_lock:
                if initial_handle in driver.window_handles:
                    driver.switch_to.window(initial_handle)
        except Exception as e:
            print(f"Error switching back to initial tab: {e}")

        return results

    except Exception as batch_error:
        print(f"Batch processing error: {str(batch_error)}")
        return [f"🠶 BATCH ERROR: Failed to process batch: {str(batch_error)}\n\n"]


def process_single_tab_with_monitoring(driver, url, max_wait_time):
    """
    Process a single tab with advanced monitoring and timeout handling.

    Args:
        driver: WebDriver instance
        url: URL being processed
        max_wait_time: Maximum wait time in seconds

    Returns:
        str: Extracted content from the tab
    """
    start_time = time.time()
    driver.set_page_load_timeout(max_wait_time)

    # Track processing stages
    state = {
        "loading_complete": False,
        "scrolling_complete": False,
        "content_extracted": False,
        "abort_processing": False
    }

    # Create a circuit breaker timer
    def circuit_breaker():
        state["abort_processing"] = True
        print(f"Circuit breaker triggered for {url} after {time.time() - start_time:.2f} seconds")
        try:
            driver.execute_script("window.stop();")
        except:
            pass

    circuit_timer = threading.Timer(max_wait_time * 0.9, circuit_breaker)
    circuit_timer.daemon = True
    circuit_timer.start()

    try:
        # Wait for the page to load with detailed progress checks
        load_success = wait_for_page_load_with_circuit_breaker(driver, max_wait_time * 0.6)
        state["loading_complete"] = True

        if state["abort_processing"]:
            return f"🠶 TIMEOUT: Circuit breaker stopped processing URL={url} after {time.time() - start_time:.2f} seconds\n\n"

        # Scroll the page to trigger lazy-loading
        scroll_thread = threading.Thread(target=scroll_with_monitoring,
                                         args=(driver, max_wait_time * 0.3, state))
        scroll_thread.daemon = True
        scroll_thread.start()
        scroll_thread.join(max_wait_time * 0.3)  # Wait with timeout

        state["scrolling_complete"] = True

        if state["abort_processing"]:
            return f"🠶 TIMEOUT: Circuit breaker stopped processing URL={url} after {time.time() - start_time:.2f} seconds\n\n"

        # Check for CAPTCHA and cookie prompts
        try:
            page_source = driver.page_source
            if ("verify" in page_source.lower() and "human" in page_source.lower()) or (
                    "accept" in page_source.lower() and "cookies" in page_source.lower()):
                handle_cookie_prompts(driver)
                # Get updated page source after handling prompts
                page_source = driver.page_source

            # Extract links
            links = categorize_links(url, page_source)

            # Use enhanced semantic extraction with fallbacks
            try:
                # First attempt with fallback for NLTK errors
                try:
                    semantic_data = extract_web_content_with_semantics(page_source, url)
                    formatted_content = format_semantic_content(semantic_data)
                except AttributeError as nltk_error:
                    # This is likely the NLTK WordListCorpusReader error
                    if "_LazyCorpusLoader__args" in str(nltk_error):
                        # Handle the specific NLTK corpus error silently
                        raise Exception("NLTK corpus not properly initialized")
                    else:
                        # Re-raise if it's a different AttributeError
                        raise
            except Exception as extraction_error:
                print(f"Falling back to basic extraction: {type(extraction_error).__name__}")
                # Fall back to basic extraction
                from integration.net.util.web_util import extract_relevant_content
                readable_text = extract_relevant_content(page_source)
                sparse_text = re.sub(r'\n+', '\n', readable_text).strip()
                formatted_content = sparse_text

            state["content_extracted"] = True

        except Exception as e:
            print(f"Error extracting content: {e}")
            formatted_content = f"! ERROR ACCESSING WEBSITE: {str(e)}"
            links = ""

        return (
            f"🠶 WEBSITE CONTENTS FOR URL={url}:\n\n"
            f"{formatted_content}\n\n"
            f"🠶 LINKS FROM WEBSITE:\n{links}\n\n"
            f"🠶 END OF WEBSITE CONTENTS FOR URL={url}\n\n"
        )

    except TimeoutException:
        print(f"Page load timeout for {url}")
        return f"🠶 TIMEOUT: URL={url} exceeded page load timeout of {max_wait_time} seconds\n\n"
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return f"🠶 ERROR: Processing of URL={url} failed with error: {str(e)}\n\n"
    finally:
        # Cancel the circuit breaker timer
        if circuit_timer.is_alive():
            circuit_timer.cancel()


def quit_all_web_drivers():
    """Close all web drivers in the pool."""
    CHROME_DRIVER_POOL.close_all()
