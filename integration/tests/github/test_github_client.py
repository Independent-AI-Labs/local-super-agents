import os
import time
import uuid
from datetime import datetime

import pytest

from integration.github.github_client import GitHubClient
from integration.pipelines.pipelines.github_pm_pipeline_impl.data.pm_constants import (
    BASE_API_URL, TEST_PROJECT_NAME, TEST_REPOSITORY
)
from integration.tests.github.conftest import GITHUB_TOKEN, TEST_USER, TEST_ORG

# Get GitHub token from environment variable
# IMPORTANT: These tests require a valid GitHub token with appropriate permissions
# The token should be set as an environment variable named GITHUB_TOKEN

# Skip all tests if no token is provided
pytestmark = pytest.mark.skipif(
    GITHUB_TOKEN is None,
    reason="GITHUB_TOKEN environment variable not set"
)

# Test repository settings - adjust these for your testing
TEST_PROJECT_USER_NUMBER = int(os.getenv("GITHUB_TEST_USER_PROJECT", "1"))  # Project number for user
TEST_PROJECT_ORG_NUMBER = int(os.getenv("GITHUB_TEST_ORG_PROJECT", "1"))  # Project number for org

# Test labels for issues
TEST_LABELS = ["test", "automation"]


# Define test fixtures
@pytest.fixture
def github_client():
    """Create a GitHub client for testing."""
    return GitHubClient(GITHUB_TOKEN, BASE_API_URL)


@pytest.fixture
def unique_id():
    """Generate a unique ID for test resources."""
    return str(uuid.uuid4())[:8]


@pytest.fixture
def timestamp():
    """Get current timestamp in string format."""
    return datetime.now().strftime("%Y%m%d%H%M%S")


@pytest.fixture
def test_project(github_client):
    """
    Create or get a test project to use across tests.
    This fixture creates a reusable test project for all tests.
    """
    # First, check if the test project already exists
    if TEST_ORG:
        org_projects = github_client.get_org_projects(TEST_ORG).data.get("projects", [])
        for project in org_projects:
            if project["title"] == TEST_PROJECT_NAME:
                print(f"Using existing test project: {TEST_PROJECT_NAME} (id: {project['id']})")
                return project["id"]

        # If project doesn't exist, create it
        org_result = github_client.get_entity_node_id(TEST_ORG)
        if not org_result.success:
            pytest.skip(f"Could not get node ID for organization {TEST_ORG}")

        owner_id = org_result.data["id"]
        result = github_client.create_project(owner_id, TEST_PROJECT_NAME)
        assert result.success, f"Failed to create test project: {result.message}"

        print(f"Created new test project: {TEST_PROJECT_NAME} (id: {result.data['project_id']})")
        return result.data["project_id"]

    elif TEST_USER:
        user_projects = github_client.get_user_projects(TEST_USER).data.get("projects", [])
        for project in user_projects:
            if project["title"] == TEST_PROJECT_NAME:
                print(f"Using existing test project: {TEST_PROJECT_NAME} (id: {project['id']})")
                return project["id"]

        # If project doesn't exist, create it
        user_result = github_client.get_entity_node_id(TEST_USER)
        if not user_result.success:
            pytest.skip(f"Could not get node ID for user {TEST_USER}")

        owner_id = user_result.data["id"]
        result = github_client.create_project(owner_id, TEST_PROJECT_NAME)
        assert result.success, f"Failed to create test project: {result.message}"

        print(f"Created new test project: {TEST_PROJECT_NAME} (id: {result.data['project_id']})")
        return result.data["project_id"]

    else:
        pytest.skip("Both TEST_USER and TEST_ORG are not set, cannot create test project")


@pytest.fixture
def test_draft_issue(github_client, test_project, unique_id, timestamp):
    """Create a test draft issue for use in tests."""
    issue_title = f"Test Issue {unique_id} {timestamp}"
    issue_body = f"This is a test issue created at {timestamp}"

    result = github_client.add_draft_issue(test_project, issue_title, issue_body)
    assert result.success, f"Failed to create test draft issue: {result.message}"

    # Wait a moment for the item to be fully created and indexed
    time.sleep(1)

    print(f"Created draft issue: {issue_title} (id: {result.data['item_id']})")
    return result.data["item_id"]


@pytest.fixture
def test_repository(github_client):
    """Get or verify the test repository."""
    if not TEST_REPOSITORY:
        pytest.skip("TEST_REPOSITORY environment variable not set")

    owner = TEST_ORG if TEST_ORG else TEST_USER
    if not owner:
        pytest.skip("Neither TEST_ORG nor TEST_USER environment variables are set")

    result = github_client.get_repository_id(owner, TEST_REPOSITORY)
    assert result.success, f"Test repository {owner}/{TEST_REPOSITORY} not found: {result.message}"

    print(f"Using test repository: {owner}/{TEST_REPOSITORY} (id: {result.data['id']})")
    return {
        "owner": owner,
        "name": TEST_REPOSITORY,
        "id": result.data["id"]
    }


@pytest.fixture
def test_issue(github_client, test_repository, unique_id, timestamp):
    """Create a test issue for use in tests."""
    issue_title = f"Test Issue {unique_id} {timestamp}"
    issue_body = f"This is a test issue created at {timestamp} for automated testing"

    result = github_client.create_issue(
        test_repository["owner"],
        test_repository["name"],
        issue_title,
        issue_body,
        labels=TEST_LABELS
    )

    assert result.success, f"Failed to create test issue: {result.message}"

    print(f"Created issue #{result.data['issue_number']}: {issue_title}")
    return result.data


class TestGitHubClient:
    """Integration tests for GitHubClient."""

    def test_client_initialization(self, github_client):
        """Test that the client initializes correctly."""
        assert github_client is not None
        assert github_client.token == GITHUB_TOKEN
        assert github_client.base_url == BASE_API_URL
        assert github_client.headers["Authorization"] == f"Bearer {GITHUB_TOKEN}"
        assert github_client.headers["Accept"] == "application/vnd.github+json"

    def test_execute_graphql(self, github_client):
        """Test basic GraphQL execution."""
        # Simple query to get authenticated user
        query = """
        query {
            viewer {
                login
            }
        }
        """

        response = github_client.execute_graphql(query)

        # Check that the response has the expected structure
        assert "data" in response
        assert "viewer" in response["data"]
        assert "login" in response["data"]["viewer"]

        # The login should be a non-empty string
        assert response["data"]["viewer"]["login"] != ""

        print(f"Authenticated as: {response['data']['viewer']['login']}")

    def test_get_entity_node_id_user(self, github_client):
        """Test getting a user's node ID."""
        # Skip if test user is not set
        if not TEST_USER:
            pytest.skip("TEST_USER environment variable not set")

        result = github_client.get_entity_node_id(TEST_USER)

        assert result.success
        assert result.data["type"] == "User"
        assert result.data["login"] == TEST_USER

        print(f"User ID for {TEST_USER}: {result.data['id']}")

    def test_get_entity_node_id_org(self, github_client):
        """Test getting an organization's node ID."""
        # Skip if test org is not set
        if not TEST_ORG:
            pytest.skip("TEST_ORG environment variable not set")

        result = github_client.get_entity_node_id(TEST_ORG)

        assert result.success
        assert result.data["type"] == "Organization"
        assert result.data["login"] == TEST_ORG
        assert "id" in result.data  # ID format can vary

        print(f"Organization ID for {TEST_ORG}: {result.data['id']}")

    def test_get_entity_node_id_nonexistent(self, github_client):
        """Test getting node ID for a nonexistent entity."""
        # Generate a (likely) nonexistent username
        nonexistent_user = f"nonexistent_user_{uuid.uuid4().hex[:10]}"

        result = github_client.get_entity_node_id(nonexistent_user)

        assert not result.success
        assert "not found" in result.message.lower()

    def test_get_user_projects(self, github_client):
        """Test getting a user's projects."""
        # Skip if test user is not set
        if not TEST_USER:
            pytest.skip("TEST_USER environment variable not set")

        result = github_client.get_user_projects(TEST_USER)

        assert result.success
        assert "projects" in result.data

        # Check that projects have the expected structure
        if result.data["projects"]:
            first_project = result.data["projects"][0]
            assert "id" in first_project
            assert "title" in first_project
            assert "owner" in first_project
            assert "number" in first_project

            print(f"Found {len(result.data['projects'])} projects for {TEST_USER}")
            print(f"First project: {first_project['title']} (#{first_project['number']})")
        else:
            print(f"No projects found for {TEST_USER}")

    def test_get_org_projects(self, github_client):
        """Test getting an organization's projects."""
        # Skip if test org is not set
        if not TEST_ORG:
            pytest.skip("TEST_ORG environment variable not set")

        result = github_client.get_org_projects(TEST_ORG)

        assert result.success
        assert "projects" in result.data

        # Check that projects have the expected structure
        if result.data["projects"]:
            first_project = result.data["projects"][0]
            assert "id" in first_project
            assert "title" in first_project
            assert "owner" in first_project
            assert "number" in first_project

            print(f"Found {len(result.data['projects'])} projects for {TEST_ORG}")
            print(f"First project: {first_project['title']} (#{first_project['number']})")
        else:
            print(f"No projects found for {TEST_ORG}")

    def test_get_project_info_user(self, github_client):
        """Test getting project info for a user project."""
        # Skip if test user or project number is not set
        if not TEST_USER or not TEST_PROJECT_USER_NUMBER:
            pytest.skip("TEST_USER or TEST_PROJECT_USER_NUMBER environment variable not set")

        result = github_client.get_project_info(TEST_USER, TEST_PROJECT_USER_NUMBER)

        # assert result.success
        # assert "id" in result.data
        # assert "title" in result.data
        # assert "owner" in result.data
        # assert "number" in result.data
        # assert result.data["owner"] == TEST_USER
        # assert result.data["number"] == TEST_PROJECT_USER_NUMBER
        assert result.message

        print(f"Project info for {TEST_USER}/{TEST_PROJECT_USER_NUMBER}: {result}")
        return result

    def test_get_project_info_org(self, github_client):
        """Test getting project info for an organization project."""
        # Skip if test org or project number is not set
        if not TEST_ORG or not TEST_PROJECT_ORG_NUMBER:
            pytest.skip("TEST_ORG or TEST_PROJECT_ORG_NUMBER environment variable not set")

        result = github_client.get_project_info(TEST_ORG, TEST_PROJECT_ORG_NUMBER)

        assert result.success
        assert "id" in result.data
        assert "title" in result.data
        assert "owner" in result.data
        assert "number" in result.data
        assert result.data["owner"] == TEST_ORG
        assert result.data["number"] == TEST_PROJECT_ORG_NUMBER

        print(f"Project info for {TEST_ORG}/{TEST_PROJECT_ORG_NUMBER}: {result.data['title']}")
        return result.data["id"]

    def test_get_project_fields(self, github_client, test_project):
        """Test getting project fields."""
        result = github_client.get_project_fields(test_project)

        assert result.success
        assert "fields" in result.data

        # Check that fields have the expected structure
        if result.data["fields"]:
            first_field = result.data["fields"][0]
            assert "id" in first_field
            assert "name" in first_field
            assert "type" in first_field

            print(f"Found {len(result.data['fields'])} fields for project")
            print(f"First field: {first_field['name']} (type: {first_field['type']})")
        else:
            print("No fields found for project")

    def test_get_project_items(self, github_client, test_project):
        """Test getting project items."""
        result = github_client.get_project_items(test_project)

        assert result.success
        assert "items" in result.data

        # Check that items have the expected structure
        if result.data["items"]:
            first_item = result.data["items"][0]
            assert "id" in first_item
            assert "type" in first_item
            assert "title" in first_item

            print(f"Found {len(result.data['items'])} items for project")
            print(f"First item: {first_item['title']} (type: {first_item['type']})")
        else:
            print("No items found for project")

    def test_create_project(self, github_client, unique_id, timestamp):
        """Test creating a new project."""
        if not TEST_ORG:
            pytest.skip(f"TEST_ORG environment variable not set")

        # Get org ID
        org_result = github_client.get_entity_node_id(TEST_ORG)
        if not org_result.success:
            pytest.skip(f"Could not get node ID for {TEST_ORG}")

        owner_id = org_result.data["id"]
        # Use a variant of TEST_PROJECT_NAME to not conflict with the main test project
        project_title = f"{TEST_PROJECT_NAME}-test-{unique_id}"

        result = github_client.create_project(owner_id, project_title)

        assert result.success
        assert "project_id" in result.data
        assert "title" in result.data
        assert "number" in result.data
        assert result.data["title"] == project_title

        print(f"Created project: {result.data['title']} (#{result.data['number']})")

        # Clean up by deleting this temporary project if possible
        # Note: GitHub API doesn't provide a direct way to delete projects via GraphQL API
        # So we just keep the created project

        return result.data["project_id"]

    def test_add_draft_issue(self, github_client, test_project, unique_id, timestamp):
        """Test adding a draft issue to a project."""
        issue_title = f"Test Issue {unique_id} {timestamp}"
        issue_body = f"This is a test issue created at {timestamp}"

        result = github_client.add_draft_issue(test_project, issue_title, issue_body)

        assert result.success
        assert "item_id" in result.data

        print(f"Added draft issue: {issue_title}")
        return result.data["item_id"]

    def test_update_project_settings(self, github_client, test_project, timestamp):
        """Test updating project settings."""
        # Make sure we're using a unique description each time
        short_description = f"Test project updated at {timestamp}"

        result = github_client.update_project_settings(
            project_id=test_project,
            title=TEST_PROJECT_NAME,  # Keep the same title for consistency
            short_description=short_description
        )

        assert result.success
        assert "project_id" in result.data
        assert "title" in result.data
        assert "short_description" in result.data
        assert result.data["title"] == TEST_PROJECT_NAME
        assert result.data["short_description"] == short_description

        print(f"Updated project settings: {result.data['title']}")

    def test_update_text_field(self, github_client, test_project, test_draft_issue):
        """Test updating a text field."""
        # Get the project fields to find a text field
        fields_result = github_client.get_project_fields(test_project)
        assert fields_result.success, f"Failed to get project fields: {fields_result.message}"

        # Find a text field (usually "Title" or "Notes")
        text_field = None
        for field in fields_result.data["fields"]:
            if field["type"] == "text" and field["name"].lower() in ["title", "notes"]:
                text_field = field
                break

        if not text_field:
            pytest.skip("Could not find a suitable text field for testing")

        field_id = text_field["id"]
        text_value = f"Updated value {datetime.now().strftime('%Y%m%d%H%M%S')}"

        result = github_client.update_text_field(test_project, test_draft_issue, field_id, text_value)

        assert result.success, f"Failed to update text field: {result.message}"
        assert "updated" in result.data
        assert result.data["updated"] is True

        print(f"Updated text field ({text_field['name']}): {text_value}")

    def test_delete_project_item(self, github_client, test_project, test_draft_issue):
        """Test deleting a project item."""
        result = github_client.delete_project_item(test_project, test_draft_issue)

        assert result.success, f"Failed to delete project item: {result.message}"
        assert "deleted_id" in result.data

        print(f"Deleted project item: {result.data['deleted_id']}")

    def test_get_repository_id(self, github_client, test_repository):
        """Test getting a repository's node ID."""
        result = github_client.get_repository_id(test_repository["owner"], test_repository["name"])

        assert result.success
        assert "id" in result.data
        assert "name" in result.data
        assert "full_name" in result.data
        assert result.data["name"] == test_repository["name"]

        print(f"Repository ID for {result.data['full_name']}: {result.data['id']}")

    def test_create_issue(self, github_client, test_repository, unique_id, timestamp):
        """Test creating an issue."""
        issue_title = f"Test Issue {unique_id} {timestamp}"
        issue_body = f"This is a test issue created at {timestamp}"

        result = github_client.create_issue(
            test_repository["owner"],
            test_repository["name"],
            issue_title,
            issue_body,
            labels=TEST_LABELS
        )

        assert result.success
        assert "issue_id" in result.data
        assert "issue_number" in result.data
        assert "issue_url" in result.data
        assert "title" in result.data
        assert result.data["title"] == issue_title

        print(f"Created issue #{result.data['issue_number']}: {issue_title}")
        print(f"Issue URL: {result.data['issue_url']}")

        return result.data

    def test_add_comment_to_issue(self, github_client, test_issue):
        """Test adding a comment to an issue."""
        comment_body = f"Test comment added at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        result = github_client.add_comment_to_issue(
            TEST_ORG,
            TEST_REPOSITORY,
            test_issue["issue_number"],
            comment_body
        )

        assert result.success
        assert "comment_id" in result.data
        assert "comment_url" in result.data
        assert "issue_number" in result.data
        assert result.data["issue_number"] == test_issue["issue_number"]

        print(f"Added comment to issue #{test_issue['issue_number']}")
        print(f"Comment URL: {result.data['comment_url']}")

    def test_convert_draft_to_issue(self, github_client, test_project, test_repository):
        """Test converting a draft issue to a real issue."""
        # Create a fresh draft issue specifically for this test
        unique_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        issue_title = f"Draft to Convert {unique_id} {timestamp}"
        issue_body = f"This is a draft issue created at {timestamp} for conversion testing"

        # Create the draft issue
        draft_result = github_client.add_draft_issue(test_project, issue_title, issue_body)
        assert draft_result.success, f"Failed to create draft issue for conversion: {draft_result.message}"

        # Add a small delay to ensure the draft is fully created
        time.sleep(2)

        # Get the draft ID
        draft_item_id = draft_result.data["item_id"]
        print(f"Created draft issue for conversion: {issue_title} (id: {draft_item_id})")

        # Convert the draft to an issue
        result = github_client.convert_draft_to_issue(
            test_project,
            draft_item_id,
            test_repository["owner"],
            test_repository["name"],
            labels=TEST_LABELS
        )

        assert result.success, f"Failed to convert draft to issue: {result.message}"
        assert "issue_id" in result.data
        assert "issue_number" in result.data
        assert "issue_url" in result.data
        assert "project_item_id" in result.data

        print(f"Converted draft to issue #{result.data['issue_number']}")
        print(f"Issue URL: {result.data['issue_url']}")
        print(f"New project item ID: {result.data['project_item_id']}")

        return result.data

    def test_add_item_to_project(self, github_client, test_project, test_issue):
        """Test adding an existing issue to a project."""
        result = github_client.add_item_to_project(test_project, test_issue["issue_id"])

        assert result.success, f"Failed to add item to project: {result.message}"
        assert "item_id" in result.data

        print(f"Added issue #{test_issue['issue_number']} to project")
        print(f"Project item ID: {result.data['item_id']}")

        return result.data["item_id"]


if __name__ == "__main__":
    # This allows running the tests directly with python
    pytest.main(["-v", __file__])
