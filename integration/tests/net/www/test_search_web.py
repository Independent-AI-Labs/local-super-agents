import pytest

from integration.net.www.surfer import search_web

# Test Configuration
TEST_BASE_URL = "http://127.0.0.1:8888"  # Replace with a test environment URL if needed
NOPECHA_KEY = "your_nopecha_key_here"  # Replace with your actual NopeCHA key
SEARCH_TERMS = ["machine learning", "artificial intelligence"]
SEMANTIC_PATTERNS = ["machine learning", "artificial intelligence", "AI", "ML"]
INSTRUCTIONS = "Filter the results based on machine learning and AI topics."


@pytest.fixture
def setup_env(monkeypatch):
    """
    Fixture to set up the environment variables required for the tests.
    """
    from integration.data import config
    config.SEARXNG_BASE_URL = TEST_BASE_URL


def test_search_web(setup_env):
    """
    Test the search_web function by verifying it returns a valid result.
    """
    filtered_data, discovered_patterns = search_web(SEARCH_TERMS, SEMANTIC_PATTERNS, INSTRUCTIONS)

    assert isinstance(filtered_data, list), "Filtered data should be a list."
    assert len(filtered_data) > 0, "Filtered data should not be empty."

    for item in filtered_data:
        assert isinstance(item, str), "Each item in filtered data should be a string."

    assert isinstance(discovered_patterns, list), "Discovered patterns should be a list."
    assert len(discovered_patterns) > 0, "Discovered patterns should not be empty."

    for pattern in discovered_patterns:
        assert isinstance(pattern, str), "Each item in discovered patterns should be a string."
