import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from selenium import webdriver
from selenium.common.exceptions import TimeoutException

from integration.data.config import SEARXNG_BASE_URL, CHROME_PATH, EXTENSIONS_PATH, TOP_N_WEB_SEARCH_RESULTS
from integration.net.util.semanting_filtering import digest_scraped_data, extract_web_content_with_semantics, format_semantic_content
from integration.net.util.web_util import get_request, extract_urls_and_common_words, categorize_links, scroll_in_background, scroll_to_bottom
from integration.net.www.chrome_driver_pool import ChromeDriverPool
from integration.util.misc_util import get_indexed_search_results_path

# Initialize the pool once
CHROME_DRIVER_POOL = ChromeDriverPool()

# Default maximum wait time for tabs in seconds
DEFAULT_MAX_TAB_WAIT_TIME = 30


def init_web_driver():
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


def search_web(search_terms: list, semantic_patterns: List[str] | None = None, instructions: str = None, max_workers: int = 8, transient: bool = False) -> Tuple[
    List[str], List[str]]:
    """
    Perform web searches for the given search terms and filter the results.

    Args:
        search_terms: List of search terms to query
        semantic_patterns: Optional list of semantic patterns to filter results
        instructions: Optional specific filtering instructions
        max_workers: Maximum number of parallel worker threads
        transient: Indexed search results are kept for this query by default. Set transient to 'True' to disable this behaviour.
    Returns:
        Tuple[List[str], List[str]]: Filtered data and discovered patterns
    """
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


def single_search(search_term: str, max_results: int = 3) -> Tuple[List[str], List[str]]:
    results = get_request(f"{SEARXNG_BASE_URL}/search?q={search_term}", {}, {})
    urls_to_scrape, most_common_words = extract_urls_and_common_words(results)
    return scrape_urls(urls_to_scrape[:max_results]), most_common_words


def scrape_urls(urls, max_tab_wait_time: int = DEFAULT_MAX_TAB_WAIT_TIME):
    """
    Scrape multiple URLs in parallel using a Chrome driver from the pool.
    First opens all tabs concurrently, then processes them separately for maximum efficiency.
    Uses enhanced content extraction and semantic HTML parsing for better results.

    Args:
        urls: List of URLs to scrape
        max_tab_wait_time: Maximum time to wait for a tab to load and process in seconds

    Returns:
        List[str]: Scraped content from each URL
    """
    if not urls:
        return []

    driver = CHROME_DRIVER_POOL.get_driver()

    try:
        # Remember the initial tab/window handle
        initial_handle = driver.current_window_handle
        initial_handles = set(driver.window_handles)

        results = []
        tab_info = {}  # Will store {handle: url} mapping

        # Use a lock for synchronizing access to the WebDriver
        driver_lock = threading.RLock()

        # PHASE 1: Open all tabs in parallel
        print(f"Opening {len(urls)} tabs in parallel")
        for index, url in enumerate(urls):
            try:
                tab_name = f"tab{index}"
                print(f"Opening tab for {url}")
                with driver_lock:
                    driver.execute_script(f"window.open('{url}', '{tab_name}');")
            except Exception as e:
                print(f"Error opening tab for {url}: {e}")

        # Get all new tab handles
        with driver_lock:
            all_handles = driver.window_handles
            new_handles = [h for h in all_handles if h not in initial_handles]

        # Process each tab sequentially to avoid race conditions
        for handle in new_handles[:]:  # Create a copy of the list to safely modify during iteration
            try:
                with driver_lock:
                    # Verify the handle is still valid
                    if handle not in driver.window_handles:
                        print(f"Window handle {handle} no longer exists, skipping")
                        continue

                    driver.switch_to.window(handle)
                    current_url = driver.current_url
                    tab_info[handle] = current_url

                print(f"Processing {current_url}")

                # Process the tab with a timeout
                start_time = time.time()
                try:
                    with driver_lock:
                        if handle in driver.window_handles:
                            content = process_single_tab(driver, url, max_tab_wait_time)
                            results.append(content)
                except Exception as e:
                    print(f"Error processing {current_url}: {str(e)}")
                    results.append(f"🠶 ERROR: Processing URL={current_url} failed: {str(e)}\n\n")
                finally:
                    # Try to close the tab after processing
                    try:
                        with driver_lock:
                            if handle in driver.window_handles:
                                driver.switch_to.window(handle)
                                driver.close()
                    except Exception as close_error:
                        print(f"Error closing tab: {str(close_error)}")

            except Exception as e:
                print(f"Error handling tab: {str(e)}")

        # Switch back to the initial tab
        try:
            with driver_lock:
                if initial_handle in driver.window_handles:
                    driver.switch_to.window(initial_handle)
        except Exception as e:
            print(f"Error switching back to initial tab: {e}")

        return results

    finally:
        # Return the driver to the pool
        CHROME_DRIVER_POOL.return_driver(driver)


def process_single_tab(driver, url, max_wait_time):
    """
    Process a single tab without threading to avoid race conditions.

    Args:
        driver: WebDriver instance
        url: URL being processed
        max_wait_time: Maximum wait time in seconds

    Returns:
        str: Extracted content from the tab
    """
    start_time = time.time()
    driver.set_page_load_timeout(max_wait_time)

    try:
        # Scroll the page to trigger lazy-loading
        scroll_to_bottom(driver, max_heights_scrolled=4)

        # Allow some time for the tab to load content
        wait_time = min(1.5, max_wait_time / 3)
        time.sleep(wait_time)

        # Check if we've exceeded our time limit
        if time.time() - start_time > max_wait_time:
            return f"🠶 TIMEOUT: Processing of URL={url} exceeded {max_wait_time} seconds\n\n"

        # Extract content
        try:
            page_source = driver.page_source

            # Check for CAPTCHA and cookie prompts
            if ("verify" in page_source.lower() and "human" in page_source.lower()) or (
                    "accept" in page_source.lower() and "cookies" in page_source.lower()):
                time.sleep(.5)
                page_source = driver.page_source

            # Extract links
            links = categorize_links(url, page_source)

            # Use enhanced semantic extraction
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
        # Return the driver to the pool
        CHROME_DRIVER_POOL.return_driver(driver)


def process_tab(driver, handle, url, max_wait_time):
    """
    Process a single tab to extract content with timeout handling.

    Args:
        driver: WebDriver instance
        handle: Window handle
        url: URL being processed
        max_wait_time: Maximum wait time in seconds

    Returns:
        str: Extracted content from the tab
    """
    start_time = time.time()
    driver.set_page_load_timeout(max_wait_time)
    local_driver = driver  # Use a local reference to avoid race conditions

    try:
        # Check if window still exists before switching
        try:
            if handle not in local_driver.window_handles:
                return f"🠶 ERROR: Window handle for URL={url} no longer exists\n\n"
            local_driver.switch_to.window(handle)
        except Exception as window_error:
            return f"🠶 ERROR: Could not switch to window for URL={url}: {str(window_error)}\n\n"

        # Start scrolling in this tab to trigger lazy-loaded content
        scroll_thread = threading.Thread(target=scroll_in_background, args=(local_driver,))
        scroll_thread.daemon = True
        scroll_thread.start()

        # Allow some time for the tab to load content
        wait_time = min(1.5, max_wait_time / 2)
        time.sleep(wait_time)

        # Check if we've exceeded our time limit
        if time.time() - start_time > max_wait_time:
            return f"🠶 TIMEOUT: Processing of URL={url} exceeded {max_wait_time} seconds\n\n"

        # Verify window is still valid before continuing
        try:
            if handle not in local_driver.window_handles:
                return f"🠶 ERROR: Window handle for URL={url} was closed during processing\n\n"
        except Exception:
            return f"🠶 ERROR: Window handle verification failed for URL={url}\n\n"

        # Extract content
        try:
            page_source = driver.page_source

            # Check for CAPTCHA and cookie prompts
            if ("verify" in page_source.lower() and "human" in page_source.lower()) or (
                    "accept" in page_source.lower() and "cookies" in page_source.lower()):
                time.sleep(.5)
                page_source = driver.page_source

            # Extract links
            links = categorize_links(url, page_source)

            # Use enhanced semantic extraction
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

        except Exception as e:
            print(f"Error extracting content: {e}")
            formatted_content = f"! ERROR ACCESSING WEBSITE: {str(e)}"
            links = ""

        # Close this tab after processing
        driver.close()

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
        print(f"Error processing tab {handle}: {e}")
        try:
            # Try to close the tab if it's still open
            driver.close()
        except:
            pass
        return f"🠶 ERROR: Processing of URL={url} failed with error: {str(e)}\n\n"


def quit_all_web_drivers():
    CHROME_DRIVER_POOL.close_all()
