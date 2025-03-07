import re
import json

from collections import Counter
from urllib.parse import urlparse, urljoin

import requests

from typing import Dict, Any, Tuple, List

from bs4 import BeautifulSoup


def get_request(url: str, headers: Dict[str, str], params: Dict[str, Any]) -> str:
    """
    Sends a GET request to the specified URL with the provided headers and parameters.

    Args:
        url (str): The URL to send the GET request to.
        headers (Dict[str, str]): A dictionary of HTTP headers to send with the request.
        params (Dict[str, Any]): A dictionary of URL parameters to send with the request.

    Returns:
        str: The response text if the request is successful, or a JSON-formatted error message.
    """
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.HTTPError as e:
        return json.dumps({"error": str(e), "content": ""})
    except Exception as e:
        return json.dumps({"error": str(e), "content": ""})


def post_request(url: str, headers: Dict[str, str], json_data: Dict[str, Any]) -> str:
    """
    Sends a POST request to the specified URL with the provided headers and JSON data.

    Args:
        url (str): The URL to send the POST request to.
        headers (Dict[str, str]): A dictionary of HTTP headers to send with the request.
        json_data (Dict[str, Any]): A dictionary of JSON data to send in the body of the request.

    Returns:
        str: The response text if the request is successful, or a JSON-formatted error message.
    """
    try:
        response = requests.post(url, headers=headers, json=json_data)
        response.raise_for_status()
        return response.text
    except requests.exceptions.HTTPError as e:
        return json.dumps({"error": str(e), "content": ""})
    except Exception as e:
        return json.dumps({"error": str(e), "content": ""})


def top_common_words(text: str, n: int = 10, min_len: int = 4) -> list:
    # Normalize the text: convert to lowercase and remove non-alphabetic characters
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)  # Extract words using regex

    # Count the frequency of each word
    word_counts = Counter(words)

    # Get the 10 most common words
    most_common_words = word_counts.most_common(n)

    return [word for word, _ in most_common_words if len(word) >= min_len and word not in ["http", "https", "cached"]]


def extract_urls_and_common_words(html_doc: str) -> Tuple[List[str], List[str]]:
    """
    Extracts URLs from anchor tags with class 'url_wrapper' in an HTML document.

    Args:
        html_doc (str): The HTML document.

    Returns:
        List[str]: A list of extracted URLs.
    """
    soup = BeautifulSoup(html_doc, 'html.parser')
    urls = [a_tag.get('href') for a_tag in soup.find_all('a', class_='url_header') if a_tag.get('href')]
    return urls, top_common_words(soup.text)


def remove_unwanted_elements(soup: BeautifulSoup) -> BeautifulSoup:
    """
    Removes unwanted elements from a BeautifulSoup object.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object to clean.

    Returns:
        BeautifulSoup: The cleaned BeautifulSoup object.
    """
    for tag in ['head', 'nav', 'footer', 'header', 'aside', 'script', 'style', 'iframe', 'noscript', 'button', 'input', 'meta']:
        for element in soup.find_all(tag):
            element.decompose()

    return soup


def extract_relevant_content(html_doc: str) -> str:
    """
    Extracts relevant content from an HTML document, excluding UI elements.

    Args:
        html_doc (str): The HTML document.

    Returns:
        str: The extracted relevant content.
    """
    soup = remove_unwanted_elements(BeautifulSoup(html_doc, 'html.parser'))
    main_content = soup.find_all(['article', 'main', 'div'])
    relevant_text = '\n'.join(content.get_text(strip=True) for content in main_content)
    return relevant_text


def categorize_links(url: str, page_source: str) -> str:
    """
    Categorizes links found on a webpage into website navigation, external links, and file links.

    Args:
        url (str): The base URL of the page.
        page_source (str): The HTML source of the page.

    Returns:
        str: A formatted string categorizing the links.
    """
    soup = BeautifulSoup(page_source, 'html.parser')
    links = soup.find_all('a', href=True)

    website_navigation = []
    external_links = []
    file_links = []

    domain = urlparse(url).netloc

    file_extensions = re.compile(r'\.(pdf|docx?|xlsx?|pptx?|zip|rar|tar\.gz|mp3|mp4|avi|mkv)$', re.IGNORECASE)

    for link in links:
        href = link['href']
        full_url = urljoin(url, href)
        parsed_href = urlparse(full_url)

        # Extract link text
        link_text = link.text.strip() if link.text else ""

        # Extract title attribute
        title = link.get('title', '')
        if title:
            title = f" ({title})"

        link_info = f"{link_text}{title} - {full_url}"

        if file_extensions.search(parsed_href.path):
            file_links.append(link_info)
        elif parsed_href.netloc == domain:
            website_navigation.append(link_info)
        else:
            external_links.append(link_info)

    # Format the output into a human-readable hierarchy string
    result = "# Website Navigation Links:\n\n"
    result += "\n".join(f"    * {item}\n" for item in website_navigation)
    result += "\n\n# External Links:\n\n"
    result += "\n".join(f"    * {item}\n" for item in external_links)
    result += "\n\n# File Links / Downloads:\n\n"
    result += "\n".join(f"    * {item}\n" for item in file_links)

    return result
