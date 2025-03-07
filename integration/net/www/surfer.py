import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from selenium import webdriver

from integration.data.config import SEARXNG_BASE_URL, CHROME_PATH, EXTENSIONS_PATH
from integration.net.util.web_util import get_request, extract_relevant_content, extract_urls_and_common_words, categorize_links
from knowledge.retrieval.hype.search.file_search import bulk_search_files

DRIVER_REGISTRY = []


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


def search_web(search_terms: list, semantic_patterns: List[str] | None = None, instructions: str = None, max_workers: int = 8) -> Tuple[List[str], List[str]]:
    # Create a temporary directory to store files
    temp_dir = tempfile.mkdtemp()
    discovered_patterns = []

    try:
        # Perform web searches in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(single_search, term) for term in search_terms]
            results = [future.result() for future in as_completed(futures)]

        # Save search results to the temporary directory
        for idx, result in enumerate(results):
            search_result, top_common_words = result
            for word in top_common_words:
                if word not in semantic_patterns and word not in discovered_patterns:
                    semantic_patterns.append(word)
                    discovered_patterns.append(word)

            file_path = os.path.join(temp_dir, f'search_result_{idx}.txt')
            with open(file_path, 'w', encoding='utf-8') as f:
                for website in search_result:
                    f.write(website + "\n")

        # Call filter_scraped_data to process the data
        filtered_data = filter_scraped_data(temp_dir, semantic_patterns, instructions)

    finally:
        # Delete the temporary directory after processing
        if os.path.exists(temp_dir):
            for file_name in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(temp_dir)

    return filtered_data, discovered_patterns


def single_search(search_term: str, max_results: int = 3) -> Tuple[List[str], List[str]]:
    results = get_request(f"{SEARXNG_BASE_URL}/search?q={search_term}", {}, {})
    urls_to_scrape, most_common_words = extract_urls_and_common_words(results)
    return scrape_urls(urls_to_scrape[:max_results]), most_common_words


def filter_scraped_data(temp_dir: str, semantic_patterns: List[str] | None = None, instructions: str = None) -> List[str]:
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


def scrape_urls(urls):
    driver = init_web_driver()
    DRIVER_REGISTRY.append(driver)

    initial_window = driver.current_window_handle

    results = []

    # Phase 1: Load all URLs in new tabs simultaneously
    for index, url in enumerate(urls):
        # Open a new tab
        try:
            print(f"Loading {url} in Chrome Driver {len(DRIVER_REGISTRY)} Tab {index}")
            driver.execute_script(f"window.open('{url}', 'tab{index}');")
        except Exception as e:
            print(f"Loading {url} in Chrome Driver {len(DRIVER_REGISTRY)} Tab {index} failed:\n{e}")

    # Phase 2: Process each tab
    for window in driver.window_handles:
        if window == initial_window:
            continue

        # Switch to the new tab
        driver.switch_to.window(window)

        # Start a new thread for scrolling the page to the bottom while it loads
        scroll_to_bottom(driver)

        url = driver.current_url
        print(f"Processing {url}")

        try:
            readable_text = extract_relevant_content(driver.page_source)

            # TODO Return links for citations later.
            links = categorize_links(url, driver.page_source)

            if ("verify" in readable_text.lower() and "human" in readable_text.lower()) or (
                    "accept" in readable_text.lower() and "cookies" in readable_text.lower()):
                time.sleep(.5)  # Allow more time for CAPTCHAs and cookies
                readable_text = extract_relevant_content(driver.page_source)

            sparse_text = re.sub(r'\n+', '\n', readable_text).strip()
            deduplicated_text = '\n'.join(sorted(set(sparse_text.splitlines()), key=sparse_text.splitlines().index))

        except Exception as e:
            print(e)
            deduplicated_text = "! ERROR ACCESSING WEBSITE!"
            links = ""

        results.append(
            f"!# WEBSITE CONTENTS FOR URL={url}:\n\n"
            f"{deduplicated_text}\n\n"
            f"!# END OF WEBSITE CONTENTS FOR URL={url}\n\n"
        )

        driver.close()

    driver.quit()
    DRIVER_REGISTRY.remove(driver)

    return results


def quit_all_web_drivers():
    for driver in DRIVER_REGISTRY:
        driver.close()
        driver.quit()
