import logging
import os
import time

import pytest

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Set up GitHub test credentials from environment variables
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
TEST_USER = os.getenv("GITHUB_USER")
TEST_ORG = os.getenv("GITHUB_ORG")


# Add a hook to delay between tests to avoid rate limiting
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Add a small delay between tests to avoid rate limiting."""
    time.sleep(1)  # 1 second delay between tests


# Define a command-line option to run write tests
def pytest_addoption(parser):
    parser.addoption(
        "--run-write-tests",
        action="store_true",
        default=False,
        help="Run tests that modify data (create, update, delete operations)"
    )


# Skip write tests unless explicitly enabled
def pytest_configure(config):
    if not config.option.run_write_tests:
        skip_write = pytest.mark.skip(reason="Need --run-write-tests option to run")
        for marker in ["write", "create", "update", "delete"]:
            config.addinivalue_line("markers", f"{marker}: mark test as modifying data")
        pytest.mark.write = skip_write
        pytest.mark.create = skip_write
        pytest.mark.update = skip_write
        pytest.mark.delete = skip_write


# Define custom markers
def pytest_collection_modifyitems(config, items):
    """Define custom markers for different test types."""
    # If run-write-tests was specified, don't skip them
    if config.getoption("--run-write-tests"):
        return

    skip_write = pytest.mark.skip(reason="Need --run-write-tests option to run")
    for item in items:
        # Skip tests that modify data unless explicitly enabled
        if any(marker in item.keywords for marker in ["write", "create", "update", "delete"]):
            item.add_marker(skip_write)
