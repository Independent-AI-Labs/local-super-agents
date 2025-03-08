import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from selenium import webdriver

from integration.data.config import SEARXNG_BASE_URL, CHROME_PATH, EXTENSIONS_PATH
from integration.net.util.semanting_filtering import digest_scraped_data, extract_web_content_with_semantics, format_semantic_content
from integration.net.util.web_util import get_request, extract_urls_and_common_words, categorize_links
from integration.net.www.chrome_driver_pool import ChromeDriverPool
from knowledge.retrieval.hype.search.file_search import bulk_search_files

# Initialize the pool once
CHROME_DRIVER_POOL = ChromeDriverPool()


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


def search_web(search_terms: list, semantic_patterns: List[str] | None = None, instructions: str = None, max_workers: int = 8) -> Tuple[
    List[str], List[str]]:
    """
    Perform web searches for the given search terms and filter the results.

    Args:
        search_terms: List of search terms to query
        semantic_patterns: Optional list of semantic patterns to filter results
        instructions: Optional specific filtering instructions
        max_workers: Maximum number of parallel worker threads
        use_hype: Whether to use the HYPE system for filtering (legacy method)

    Returns:
        Tuple[List[str], List[str]]: Filtered data and discovered patterns
    """
    # Create a temporary directory to store files (used for HYPE method)
    temp_dir = "results"
    os.makedirs(temp_dir, exist_ok=True)

    filtered_data = []
    discovered_patterns = []

    try:
        # Perform web searches in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(single_search, term) for term in search_terms]
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

        for idx, search_result in enumerate(all_search_results):
            file_path = os.path.join(temp_dir, f'search_result_{idx}_raw.txt')
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(search_result)

        # If no semantic patterns are provided, use top discovered patterns
        patterns_to_use = semantic_patterns if semantic_patterns else discovered_patterns

        filtered_data = digest_scraped_data(
            all_search_results,
            patterns_to_use,
            instructions
        )

        for idx, search_result in enumerate(filtered_data):
            file_path = os.path.join(temp_dir, f'search_result_{idx}_digested.txt')
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(search_result)

    except Exception as e:
        # Log the error and return empty results
        print(f"Error in search_web: {str(e)}")
        filtered_data = [f"Error during web search: {str(e)}"]

    finally:
        # Clean up temporary directory if needed
        # Commented out for debugging purposes
        # if os.path.exists(temp_dir):
        #     for file_name in os.listdir(temp_dir):
        #         file_path = os.path.join(temp_dir, file_name)
        #         if os.path.isfile(file_path):
        #             os.remove(file_path)
        #     os.rmdir(temp_dir)
        pass

    return filtered_data, discovered_patterns


def single_search(search_term: str, max_results: int = 3) -> Tuple[List[str], List[str]]:
    results = get_request(f"{SEARXNG_BASE_URL}/search?q={search_term}", {}, {})
    urls_to_scrape, most_common_words = extract_urls_and_common_words(results)
    return scrape_urls(urls_to_scrape[:max_results]), most_common_words


def filter_scraped_data_hype(temp_dir: str, semantic_patterns: List[str] | None = None, instructions: str = None) -> List[str]:
    """
    Filters the scraped data from files stored in the temp directory based on semantic patterns and instructions.

    :param temp_dir: The directory containing the files with scraped data.
    :param semantic_patterns: A list of semantic patterns to filter the results.
    :param instructions: Additional instructions to guide the filtering process.
    :return: Filtered data as a string.
    """
    # Extract the search terms from the semantic patterns if provided
    search_term_strings = semantic_patterns if semantic_patterns else []

    # Run a bulk search on all the files in the temp directory
    search_results = bulk_search_files(
        root_dir=temp_dir,
        search_term_strings=search_term_strings,
        context_size_lines=8,  # Adjust the context size if necessary
        large_file_size_threshold=512 * 1024 * 1024,  # 512MB
        min_score=1,  # Minimum score threshold
        and_search=False,  # Allow OR searches by default
        exact_matches_only=True  # Allow inexact matches
    )

    # Collect the filtered data from the search results
    filtered_data = []
    for result in search_results:
        # You can adjust the filtering logic here based on instructions
        if result.common.score >= 1:  # You can modify the scoring threshold
            filtered_data.append(f"URI: {result.common.uri}\nScore: {result.common.score}\n")

            # File matches
            if result.file_matches:
                filtered_data.append(f"File: {result.file_matches.title}\n")
                for line_number, match in zip(result.file_matches.line_numbers, result.file_matches.matches_with_context):
                    filtered_data.append(f"Line {line_number}: {match}\n")

    # Join all the filtered results into a string
    return filtered_data


def scroll_to_bottom(driver, max_heights_scrolled: int = 6):
    last_height = driver.execute_script("return document.body.scrollHeight")
    step = 0

    # No doom-scrolling!
    while step < max_heights_scrolled:
        # Scroll down to the bottom of the page
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        time.sleep(.5)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height >= last_height:
            break

        last_height = new_height
        step += 1

# TODO Max tab wait time.
def scrape_urls(urls):
    """
    Scrape multiple URLs in parallel using a Chrome driver from the pool.
    First opens all tabs concurrently, then processes them separately for maximum efficiency.
    Uses enhanced content extraction and semantic HTML parsing for better results.
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

        # PHASE 1: Open all tabs in parallel
        print(f"Opening {len(urls)} tabs in parallel")
        for index, url in enumerate(urls):
            try:
                tab_name = f"tab{index}"
                print(f"Opening tab for {url}")
                driver.execute_script(f"window.open('{url}', '{tab_name}');")
            except Exception as e:
                print(f"Error opening tab for {url}: {e}")

        # Get all new tab handles
        all_handles = driver.window_handles
        new_handles = [h for h in all_handles if h not in initial_handles]

        # PHASE 2: Let all tabs load while scrolling through each
        # Map handles to their URLs and begin scrolling in each
        for handle in new_handles:
            try:
                driver.switch_to.window(handle)
                current_url = driver.current_url
                tab_info[handle] = current_url
                print(f"Starting scroll on {current_url}")

                # Start scrolling in this tab to trigger lazy-loaded content
                scroll_thread = threading.Thread(target=scroll_in_background, args=(driver,))
                scroll_thread.daemon = True
                scroll_thread.start()

            except Exception as e:
                print(f"Error switching to tab: {e}")

        # Allow some time for all tabs to load their content
        time.sleep(1.5)  # Adjust this value based on your needs

        # PHASE 3: Process each tab to extract content
        for handle in new_handles:
            try:
                driver.switch_to.window(handle)
                url = tab_info.get(handle, driver.current_url)

                print(f"Processing content from {url}")

                try:
                    # Use enhanced content extraction instead of the basic method
                    page_source = driver.page_source

                    # Check for CAPTCHA and cookie prompts
                    if ("verify" in page_source.lower() and "human" in page_source.lower()) or (
                            "accept" in page_source.lower() and "cookies" in page_source.lower()):
                        time.sleep(.5)
                        page_source = driver.page_source

                    # Extract links (we'll keep this function as it's not related to content extraction)
                    links = categorize_links(url, page_source)

                    # Use enhanced semantic extraction instead of basic extraction
                    try:
                        semantic_data = extract_web_content_with_semantics(page_source, url)
                        formatted_content = format_semantic_content(semantic_data)
                    except Exception as extraction_error:
                        print(f"Error with enhanced extraction: {extraction_error}")
                        # Fall back to basic extraction if enhanced extraction fails
                        from integration.net.util.web_util import extract_relevant_content
                        readable_text = extract_relevant_content(page_source)
                        sparse_text = re.sub(r'\n+', '\n', readable_text).strip()
                        formatted_content = sparse_text

                except Exception as e:
                    print(f"Error extracting content: {e}")
                    formatted_content = f"! ERROR ACCESSING WEBSITE: {str(e)}"
                    links = ""

                # Add the result
                results.append(
                    f"🠶 WEBSITE CONTENTS FOR URL={url}:\n\n"
                    f"{formatted_content}\n\n"
                    f"🠶 LINKS FROM WEBSITE:\n{links}\n\n"
                    f"🠶 END OF WEBSITE CONTENTS FOR URL={url}\n\n"
                )

                # Close this tab after processing
                driver.close()

            except Exception as e:
                print(f"Error processing tab {handle}: {e}")
                try:
                    # Try to close the tab if it's still open
                    driver.close()
                except:
                    pass

        # Switch back to the initial tab
        try:
            driver.switch_to.window(initial_handle)
        except Exception as e:
            print(f"Error switching back to initial tab: {e}")

        return results

    finally:
        # Return the driver to the pool
        CHROME_DRIVER_POOL.return_driver(driver)


def scroll_in_background(driver, max_heights_scrolled=6, scroll_interval=0.5):
    """
    Scroll the page in the background to trigger lazy-loading of content.
    This function is meant to be run in a separate thread.
    """
    try:
        last_height = driver.execute_script("return document.body.scrollHeight")
        step = 0

        # No doom-scrolling!
        while step < max_heights_scrolled:
            # Scroll down to the bottom of the page
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            time.sleep(scroll_interval)

            # Calculate new scroll height and compare with last scroll height
            new_height = driver.execute_script("return document.body.scrollHeight")

            if new_height >= last_height:
                break

            last_height = new_height
            step += 1
    except Exception as e:
        # Silently fail as this is a background task
        print(f"Background scrolling error (non-critical): {e}")


def quit_all_web_drivers():
    CHROME_DRIVER_POOL.close_all()
