import logging
from typing import Dict, Any, Optional, List

import requests

from integration.github.models.project_models import ActionResult, ProjectInfo, ProjectField, ProjectItem


class GitHubClient:
    """Client for interacting with GitHub API."""

    def __init__(self, token: str, base_url: str = "https://api.github.com"):
        """
        Initialize the GitHub client.

        Args:
            token: GitHub API token
            base_url: Base URL for GitHub API
        """
        self.token = token
        self.base_url = base_url
        self.logger = logging.getLogger(self.__class__.__name__)
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }

    def execute_graphql(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a GraphQL query against the GitHub API.

        Args:
            query: GraphQL query string
            variables: Variables for the query

        Returns:
            Dict: Response from the API
        """
        graphql_url = f"{self.base_url}/graphql"

        payload = {
            "query": query
        }

        if variables:
            payload["variables"] = variables

        try:
            response = requests.post(
                graphql_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"GraphQL request failed: {str(e)}")
            if hasattr(e, 'response') and e.response:
                self.logger.error(f"Response: {e.response.text}")
            raise

    def create_issue(self, owner: str, repo: str, title: str, body: str,
                     labels: Optional[List[str]] = None) -> ActionResult:
        """
        Create a new issue in a repository.

        Args:
            owner: Repository owner (user or organization)
            repo: Repository name
            title: Issue title
            body: Issue body
            labels: List of labels to apply (optional)

        Returns:
            ActionResult: Result of the action
        """
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/issues"

            payload = {
                "title": title,
                "body": body
            }

            if labels:
                payload["labels"] = labels

            response = requests.post(
                url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()

            return ActionResult(
                success=True,
                message=f"Successfully created issue #{data.get('number')}",
                data={
                    "issue_id": data.get("node_id"),
                    "issue_number": data.get("number"),
                    "issue_url": data.get("html_url"),
                    "title": data.get("title")
                }
            )
        except requests.RequestException as e:
            self.logger.error(f"Error creating issue: {str(e)}")
            if hasattr(e, 'response') and e.response:
                self.logger.error(f"Response: {e.response.text}")
            return ActionResult(
                success=False,
                message=f"Failed to create issue: {str(e)}",
                data={}
            )

    def convert_draft_to_issue(self, project_id: str, draft_item_id: str,
                               owner: str, repo: str, labels: Optional[List[str]] = None) -> ActionResult:
        """
        Convert a draft issue in a project to a real issue in a repository.

        Args:
            project_id: Project ID
            draft_item_id: Draft item ID
            owner: Repository owner (user or organization)
            repo: Repository name
            labels: List of labels to apply (optional)

        Returns:
            ActionResult: Result of the action
        """
        try:
            # Get the repository ID
            repo_result = self.get_repository_id(owner, repo)
            if not repo_result.success:
                return ActionResult(
                    success=False,
                    message=f"Repository {owner}/{repo} not found: {repo_result.message}",
                    data={}
                )

            repository_id = repo_result.data.get("id")

            # Use the correct mutation to convert the draft issue to a real issue
            query = """
            mutation($input: ConvertProjectV2DraftIssueItemToIssueInput!) {
                convertProjectV2DraftIssueItemToIssue(input: $input) {
                    item {
                        id
                        content {
                            ... on Issue {
                                id
                                number
                                url
                                title
                            }
                        }
                    }
                }
            }
            """

            variables = {
                "input": {
                    "itemId": draft_item_id,
                    "repositoryId": repository_id
                }
            }

            response = self.execute_graphql(query, variables)

            if "errors" in response:
                return ActionResult(
                    success=False,
                    message=f"Failed to convert draft to issue: {response['errors'][0]['message']}",
                    data={}
                )

            item_data = response.get("data", {}).get("convertProjectV2DraftIssueItemToIssue", {}).get("item", {})
            issue_data = item_data.get("content", {})

            if not issue_data:
                return ActionResult(
                    success=False,
                    message="Failed to convert draft to issue: No issue data returned",
                    data={}
                )

            # If labels are provided, add them to the issue
            if labels and issue_data.get("number"):
                issue_number = issue_data.get("number")
                url = f"{self.base_url}/repos/{owner}/{repo}/issues/{issue_number}/labels"

                payload = {
                    "labels": labels
                }

                label_response = requests.post(
                    url,
                    headers=self.headers,
                    json=payload
                )
                label_response.raise_for_status()

            return ActionResult(
                success=True,
                message="Successfully converted draft to issue",
                data={
                    "issue_id": issue_data.get("id", ""),
                    "issue_number": issue_data.get("number"),
                    "issue_url": issue_data.get("url"),
                    "project_item_id": item_data.get("id")
                }
            )

        except Exception as e:
            self.logger.error(f"Error converting draft to issue: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Failed to convert draft to issue: {str(e)}",
                data={}
            )

    def add_comment_to_issue(self, owner: str, repo: str, issue_number: int, body: str) -> ActionResult:
        """
        Add a comment to an existing issue.

        Args:
            owner: Repository owner (user or organization)
            repo: Repository name
            issue_number: Issue number
            body: Comment body

        Returns:
            ActionResult: Result of the action
        """
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/issues/{issue_number}/comments"

            payload = {
                "body": body
            }

            response = requests.post(
                url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()

            return ActionResult(
                success=True,
                message=f"Successfully added comment to issue #{issue_number}",
                data={
                    "comment_id": data.get("node_id"),
                    "comment_url": data.get("html_url"),
                    "issue_number": issue_number
                }
            )
        except requests.RequestException as e:
            self.logger.error(f"Error adding comment to issue: {str(e)}")
            if hasattr(e, 'response') and e.response:
                self.logger.error(f"Response: {e.response.text}")
            return ActionResult(
                success=False,
                message=f"Failed to add comment to issue: {str(e)}",
                data={}
            )

    def get_repository_id(self, owner: str, repo: str) -> ActionResult:
        """
        Get the node ID of a repository.

        Args:
            owner: Repository owner (user or organization)
            repo: Repository name

        Returns:
            ActionResult: Result of the action with the node ID
        """
        try:
            query = """
            query($owner: String!, $name: String!) {
                repository(owner: $owner, name: $name) {
                    id
                    name
                    nameWithOwner
                }
            }
            """

            variables = {
                "owner": owner,
                "name": repo
            }

            response = self.execute_graphql(query, variables)
            repo_data = response.get("data", {}).get("repository", {})

            if not repo_data:
                return ActionResult(
                    success=False,
                    message=f"Repository {owner}/{repo} not found",
                    data={}
                )

            return ActionResult(
                success=True,
                message="Successfully retrieved repository ID",
                data={
                    "id": repo_data.get("id", ""),
                    "name": repo_data.get("name", ""),
                    "full_name": repo_data.get("nameWithOwner", "")
                }
            )

        except Exception as e:
            self.logger.error(f"Error getting repository ID: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Failed to get repository ID: {str(e)}",
                data={}
            )

    def get_project_info(self, owner: str, project_number: int) -> ActionResult:
        """
        Get information about a project.

        Args:
            owner: Username or organization name
            project_number: Project number

        Returns:
            ActionResult: Result of the action
        """
        # First determine if owner is user or organization
        try:
            user_info = requests.get(
                f"{self.base_url}/users/{owner}",
                headers=self.headers
            ).json()

            is_org = user_info.get("type") == "Organization"

            # Query differs based on whether it's a user or org project
            if is_org:
                query = """
                query($login: String!, $number: Int!) {
                    organization(login: $login) {
                        projectV2(number: $number) {
                            id
                            title
                            number
                            public
                            shortDescription
                        }
                    }
                }
                """
                variables = {
                    "login": owner,
                    "number": project_number
                }

                response = self.execute_graphql(query, variables)
                project_data = response.get("data", {}).get("organization", {}).get("projectV2", {})
            else:
                query = """
                query($login: String!, $number: Int!) {
                    user(login: $login) {
                        projectV2(number: $number) {
                            id
                            title
                            number
                            public
                            shortDescription
                        }
                    }
                }
                """
                variables = {
                    "login": owner,
                    "number": project_number
                }

                response = self.execute_graphql(query, variables)
                project_data = response.get("data", {}).get("user", {}).get("projectV2", {})

            if not project_data:
                return ActionResult(
                    success=False,
                    message=f"Project {project_number} not found for {owner}",
                    data={}
                )

            project_info = ProjectInfo(
                id=project_data.get("id", ""),
                title=project_data.get("title", ""),
                owner=owner,
                public=project_data.get("public", False),
                number=project_data.get("number", 0),
                description=project_data.get("shortDescription", "")
            )

            return ActionResult(
                success=True,
                message=f"Successfully retrieved project information",
                data=project_info.dict()
            )

        except Exception as e:
            self.logger.error(f"Error getting project info: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Failed to get project information: {str(e)}",
                data={}
            )

    def get_user_projects(self, username: str, limit: int = 20) -> ActionResult:
        """
        Get projects for a user.

        Args:
            username: GitHub username
            limit: Maximum number of projects to return

        Returns:
            ActionResult: Result of the action
        """
        query = """
        query($login: String!, $first: Int!) {
            user(login: $login) {
                projectsV2(first: $first) {
                    nodes {
                        id
                        title
                        number
                        public
                        shortDescription
                    }
                }
            }
        }
        """

        variables = {
            "login": username,
            "first": limit
        }

        try:
            response = self.execute_graphql(query, variables)
            projects_data = response.get("data", {}).get("user", {}).get("projectsV2", {}).get("nodes", [])

            projects = []
            for project in projects_data:
                projects.append(ProjectInfo(
                    id=project.get("id", ""),
                    title=project.get("title", ""),
                    owner=username,
                    public=project.get("public", False),
                    number=project.get("number", 0),
                    description=project.get("shortDescription", "")
                ).dict())

            return ActionResult(
                success=True,
                message=f"Successfully retrieved {len(projects)} projects for user {username}",
                data={"projects": projects}
            )

        except Exception as e:
            self.logger.error(f"Error getting user projects: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Failed to get projects for user {username}: {str(e)}",
                data={}
            )

    def get_org_projects(self, org_name: str, limit: int = 20) -> ActionResult:
        """
        Get projects for an organization.

        Args:
            org_name: GitHub organization name
            limit: Maximum number of projects to return

        Returns:
            ActionResult: Result of the action
        """
        query = """
        query($login: String!, $first: Int!) {
            organization(login: $login) {
                projectsV2(first: $first) {
                    nodes {
                        id
                        title
                        number
                        public
                        shortDescription
                    }
                }
            }
        }
        """

        variables = {
            "login": org_name,
            "first": limit
        }

        try:
            response = self.execute_graphql(query, variables)
            projects_data = response.get("data", {}).get("organization", {}).get("projectsV2", {}).get("nodes", [])

            projects = []
            for project in projects_data:
                projects.append(ProjectInfo(
                    id=project.get("id", ""),
                    title=project.get("title", ""),
                    owner=org_name,
                    public=project.get("public", False),
                    number=project.get("number", 0),
                    description=project.get("shortDescription", "")
                ).dict())

            return ActionResult(
                success=True,
                message=f"Successfully retrieved {len(projects)} projects for organization {org_name}",
                data={"projects": projects}
            )

        except Exception as e:
            self.logger.error(f"Error getting organization projects: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Failed to get projects for organization {org_name}: {str(e)}",
                data={}
            )

    def get_project_fields(self, project_id: str) -> ActionResult:
        """
        Get fields for a project.

        Args:
            project_id: Project ID

        Returns:
            ActionResult: Result of the action
        """
        query = """
        query($projectId: ID!) {
            node(id: $projectId) {
                ... on ProjectV2 {
                    fields(first: 20) {
                        nodes {
                            ... on ProjectV2Field {
                                id
                                name
                            }
                            ... on ProjectV2IterationField {
                                id
                                name
                                configuration {
                                    iterations {
                                        startDate
                                        id
                                    }
                                }
                            }
                            ... on ProjectV2SingleSelectField {
                                id
                                name
                                options {
                                    id
                                    name
                                }
                            }
                        }
                    }
                }
            }
        }
        """

        variables = {
            "projectId": project_id
        }

        try:
            response = self.execute_graphql(query, variables)
            fields_data = response.get("data", {}).get("node", {}).get("fields", {}).get("nodes", [])

            fields = []
            for field in fields_data:
                field_type = "text"  # Default type
                options = None
                iterations = None

                if "options" in field:
                    field_type = "single_select"
                    options = {opt["id"]: opt["name"] for opt in field.get("options", [])}

                if "configuration" in field:
                    field_type = "iteration"
                    iterations = {it["id"]: it["startDate"] for it in field.get("configuration", {}).get("iterations", [])}

                fields.append(ProjectField(
                    id=field.get("id", ""),
                    name=field.get("name", ""),
                    type=field_type,
                    options=options,
                    iterations=iterations
                ).dict())

            return ActionResult(
                success=True,
                message=f"Successfully retrieved {len(fields)} fields for project",
                data={"fields": fields}
            )

        except Exception as e:
            self.logger.error(f"Error getting project fields: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Failed to get fields for project: {str(e)}",
                data={}
            )

    def add_item_to_project(self, project_id: str, content_id: str) -> ActionResult:
        """
        Add an item to a project.

        Args:
            project_id: Project ID
            content_id: Content ID (issue or PR)

        Returns:
            ActionResult: Result of the action
        """
        query = """
        mutation($projectId: ID!, $contentId: ID!) {
            addProjectV2ItemById(input: {
                projectId: $projectId
                contentId: $contentId
            }) {
                item {
                    id
                }
            }
        }
        """

        variables = {
            "projectId": project_id,
            "contentId": content_id
        }

        try:
            response = self.execute_graphql(query, variables)

            if "errors" in response:
                return ActionResult(
                    success=False,
                    message=f"Failed to add item to project: {response['errors'][0]['message']}",
                    data={}
                )

            item_id = response.get("data", {}).get("addProjectV2ItemById", {}).get("item", {}).get("id", "")

            return ActionResult(
                success=True,
                message=f"Successfully added item to project",
                data={"item_id": item_id}
            )

        except Exception as e:
            self.logger.error(f"Error adding item to project: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Failed to add item to project: {str(e)}",
                data={}
            )

    def add_draft_issue(self, project_id: str, title: str, body: str) -> ActionResult:
        """
        Add a draft issue to a project.

        Args:
            project_id: Project ID
            title: Issue title
            body: Issue body

        Returns:
            ActionResult: Result of the action
        """
        query = """
        mutation($projectId: ID!, $title: String!, $body: String!) {
            addProjectV2DraftIssue(input: {
                projectId: $projectId
                title: $title
                body: $body
            }) {
                projectItem {
                    id
                }
            }
        }
        """

        variables = {
            "projectId": project_id,
            "title": title,
            "body": body
        }

        try:
            response = self.execute_graphql(query, variables)

            if "errors" in response:
                return ActionResult(
                    success=False,
                    message=f"Failed to add draft issue: {response['errors'][0]['message']}",
                    data={}
                )

            item_id = response.get("data", {}).get("addProjectV2DraftIssue", {}).get("projectItem", {}).get("id", "")

            return ActionResult(
                success=True,
                message=f"Successfully added draft issue to project",
                data={"item_id": item_id}
            )

        except Exception as e:
            self.logger.error(f"Error adding draft issue: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Failed to add draft issue: {str(e)}",
                data={}
            )

    def update_text_field(self, project_id: str, item_id: str, field_id: str, text_value: str) -> ActionResult:
        """
        Update a text field value for an item.

        Args:
            project_id: Project ID
            item_id: Item ID
            field_id: Field ID
            text_value: New text value

        Returns:
            ActionResult: Result of the action
        """
        query = """
        mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $text: String!) {
            updateProjectV2ItemFieldValue(input: {
                projectId: $projectId
                itemId: $itemId
                fieldId: $fieldId
                value: { text: $text }
            }) {
                projectV2Item {
                    id
                }
            }
        }
        """

        variables = {
            "projectId": project_id,
            "itemId": item_id,
            "fieldId": field_id,
            "text": text_value
        }

        try:
            response = self.execute_graphql(query, variables)

            if "errors" in response:
                return ActionResult(
                    success=False,
                    message=f"Failed to update text field: {response['errors'][0]['message']}",
                    data={}
                )

            return ActionResult(
                success=True,
                message=f"Successfully updated text field",
                data={"updated": True}
            )

        except Exception as e:
            self.logger.error(f"Error updating text field: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Failed to update text field: {str(e)}",
                data={}
            )

    def update_select_field(self, project_id: str, item_id: str, field_id: str, option_id: str) -> ActionResult:
        """
        Update a single select field value for an item.

        Args:
            project_id: Project ID
            item_id: Item ID
            field_id: Field ID
            option_id: ID of the selected option

        Returns:
            ActionResult: Result of the action
        """
        query = """
        mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
            updateProjectV2ItemFieldValue(input: {
                projectId: $projectId
                itemId: $itemId
                fieldId: $fieldId
                value: { singleSelectOptionId: $optionId }
            }) {
                projectV2Item {
                    id
                }
            }
        }
        """

        variables = {
            "projectId": project_id,
            "itemId": item_id,
            "fieldId": field_id,
            "optionId": option_id
        }

        try:
            response = self.execute_graphql(query, variables)

            if "errors" in response:
                return ActionResult(
                    success=False,
                    message=f"Failed to update select field: {response['errors'][0]['message']}",
                    data={}
                )

            return ActionResult(
                success=True,
                message=f"Successfully updated select field",
                data={"updated": True}
            )

        except Exception as e:
            self.logger.error(f"Error updating select field: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Failed to update select field: {str(e)}",
                data={}
            )

    def update_iteration_field(self, project_id: str, item_id: str, field_id: str, iteration_id: str) -> ActionResult:
        """
        Update an iteration field value for an item.

        Args:
            project_id: Project ID
            item_id: Item ID
            field_id: Field ID
            iteration_id: ID of the iteration

        Returns:
            ActionResult: Result of the action
        """
        query = """
        mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $iterationId: String!) {
            updateProjectV2ItemFieldValue(input: {
                projectId: $projectId
                itemId: $itemId
                fieldId: $fieldId
                value: { iterationId: $iterationId }
            }) {
                projectV2Item {
                    id
                }
            }
        }
        """

        variables = {
            "projectId": project_id,
            "itemId": item_id,
            "fieldId": field_id,
            "iterationId": iteration_id
        }

        try:
            response = self.execute_graphql(query, variables)

            if "errors" in response:
                return ActionResult(
                    success=False,
                    message=f"Failed to update iteration field: {response['errors'][0]['message']}",
                    data={}
                )

            return ActionResult(
                success=True,
                message=f"Successfully updated iteration field",
                data={"updated": True}
            )

        except Exception as e:
            self.logger.error(f"Error updating iteration field: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Failed to update iteration field: {str(e)}",
                data={}
            )

    def delete_project_item(self, project_id: str, item_id: str) -> ActionResult:
        """
        Delete an item from a project.

        Args:
            project_id: Project ID
            item_id: Item ID

        Returns:
            ActionResult: Result of the action
        """
        query = """
        mutation($projectId: ID!, $itemId: ID!) {
            deleteProjectV2Item(input: {
                projectId: $projectId
                itemId: $itemId
            }) {
                deletedItemId
            }
        }
        """

        variables = {
            "projectId": project_id,
            "itemId": item_id
        }

        try:
            response = self.execute_graphql(query, variables)

            if "errors" in response:
                return ActionResult(
                    success=False,
                    message=f"Failed to delete item: {response['errors'][0]['message']}",
                    data={}
                )

            deleted_id = response.get("data", {}).get("deleteProjectV2Item", {}).get("deletedItemId", "")

            return ActionResult(
                success=True,
                message=f"Successfully deleted item from project",
                data={"deleted_id": deleted_id}
            )

        except Exception as e:
            self.logger.error(f"Error deleting project item: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Failed to delete item: {str(e)}",
                data={}
            )

    def create_project(self, owner_id: str, title: str) -> ActionResult:
        """
        Create a new project.

        Args:
            owner_id: Node ID of the owner (user or organization)
            title: Project title

        Returns:
            ActionResult: Result of the action
        """
        query = """
        mutation($ownerId: ID!, $title: String!) {
            createProjectV2(input: {
                ownerId: $ownerId
                title: $title
            }) {
                projectV2 {
                    id
                    title
                    number
                }
            }
        }
        """

        variables = {
            "ownerId": owner_id,
            "title": title
        }

        try:
            response = self.execute_graphql(query, variables)

            if "errors" in response:
                return ActionResult(
                    success=False,
                    message=f"Failed to create project: {response['errors'][0]['message']}",
                    data={}
                )

            project_data = response.get("data", {}).get("createProjectV2", {}).get("projectV2", {})

            return ActionResult(
                success=True,
                message=f"Successfully created project: {project_data.get('title')}",
                data={
                    "project_id": project_data.get("id", ""),
                    "title": project_data.get("title", ""),
                    "number": project_data.get("number", 0)
                }
            )

        except Exception as e:
            self.logger.error(f"Error creating project: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Failed to create project: {str(e)}",
                data={}
            )

    def update_project_settings(self, project_id: str, title: str = None,
                                public: bool = None, readme: str = None,
                                short_description: str = None) -> ActionResult:
        """
        Update project settings.

        Args:
            project_id: Project ID
            title: New project title
            public: Whether the project is public
            readme: Project README content
            short_description: Short description

        Returns:
            ActionResult: Result of the action
        """
        query = """
        mutation($projectId: ID!, $title: String, $public: Boolean, $readme: String, $shortDescription: String) {
            updateProjectV2(input: {
                projectId: $projectId
                title: $title
                public: $public
                readme: $readme
                shortDescription: $shortDescription
            }) {
                projectV2 {
                    id
                    title
                    public
                    readme
                    shortDescription
                }
            }
        }
        """

        variables = {
            "projectId": project_id
        }

        # Only include parameters that are provided
        if title is not None:
            variables["title"] = title
        if public is not None:
            variables["public"] = public
        if readme is not None:
            variables["readme"] = readme
        if short_description is not None:
            variables["shortDescription"] = short_description

        try:
            response = self.execute_graphql(query, variables)

            if "errors" in response:
                return ActionResult(
                    success=False,
                    message=f"Failed to update project settings: {response['errors'][0]['message']}",
                    data={}
                )

            project_data = response.get("data", {}).get("updateProjectV2", {}).get("projectV2", {})

            return ActionResult(
                success=True,
                message=f"Successfully updated project settings",
                data={
                    "project_id": project_data.get("id", ""),
                    "title": project_data.get("title", ""),
                    "public": project_data.get("public", False),
                    "readme": project_data.get("readme", ""),
                    "short_description": project_data.get("shortDescription", "")
                }
            )

        except Exception as e:
            self.logger.error(f"Error updating project settings: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Failed to update project settings: {str(e)}",
                data={}
            )

    def get_project_items(self, project_id: str, limit: int = 20) -> ActionResult:
        """
        Get items in a project.

        Args:
            project_id: Project ID
            limit: Maximum number of items to return

        Returns:
            ActionResult: Result of the action
        """
        query = """
        query($projectId: ID!, $first: Int!) {
            node(id: $projectId) {
                ... on ProjectV2 {
                    items(first: $first) {
                        nodes {
                            id
                            fieldValues(first: 8) {
                                nodes {
                                    ... on ProjectV2ItemFieldTextValue {
                                        text
                                        field {
                                            ... on ProjectV2FieldCommon {
                                                name
                                            }
                                        }
                                    }
                                    ... on ProjectV2ItemFieldDateValue {
                                        date
                                        field {
                                            ... on ProjectV2FieldCommon {
                                                name
                                            }
                                        }
                                    }
                                    ... on ProjectV2ItemFieldSingleSelectValue {
                                        name
                                        field {
                                            ... on ProjectV2FieldCommon {
                                                name
                                            }
                                        }
                                    }
                                }
                            }
                            content {
                                ... on DraftIssue {
                                    title
                                    body
                                }
                                ... on Issue {
                                    title
                                    url
                                }
                                ... on PullRequest {
                                    title
                                    url
                                }
                            }
                        }
                    }
                }
            }
        }
        """

        variables = {
            "projectId": project_id,
            "first": limit
        }

        try:
            response = self.execute_graphql(query, variables)
            items_data = response.get("data", {}).get("node", {}).get("items", {}).get("nodes", [])

            items = []
            for item in items_data:
                content = item.get("content", {})

                # Determine the item type
                item_type = "unknown"
                title = ""
                if "body" in content:
                    item_type = "draft_issue"
                    title = content.get("title", "")
                elif "url" in content:
                    if "Issue" in str(content):
                        item_type = "issue"
                        title = content.get("title", "")
                    else:
                        item_type = "pull_request"
                        title = content.get("title", "")

                # Extract field values
                field_values = {}
                for field_value in item.get("fieldValues", {}).get("nodes", []):
                    if "field" in field_value and "name" in field_value["field"]:
                        field_name = field_value["field"]["name"]

                        if "text" in field_value:
                            field_values[field_name] = field_value["text"]
                        elif "date" in field_value:
                            field_values[field_name] = field_value["date"]
                        elif "name" in field_value and "field" in field_value:
                            field_values[field_name] = field_value["name"]

                items.append(ProjectItem(
                    id=item.get("id", ""),
                    type=item_type,
                    title=title,
                    field_values=field_values,
                    content_id=content.get("id", "")
                ).dict())

            return ActionResult(
                success=True,
                message=f"Successfully retrieved {len(items)} items from project",
                data={"items": items}
            )

        except Exception as e:
            self.logger.error(f"Error getting project items: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Failed to get project items: {str(e)}",
                data={}
            )

    def get_entity_node_id(self, entity_name: str) -> ActionResult:
        """
        Get the node ID of a GitHub user or organization.

        Args:
            entity_name: Username or organization name

        Returns:
            ActionResult: Result of the action with the node ID
        """
        try:
            # First try as user
            user_query = """
            query($login: String!) {
                user(login: $login) {
                    id
                    login
                    __typename
                }
            }
            """

            variables = {
                "login": entity_name
            }

            response = self.execute_graphql(user_query, variables)
            user_data = response.get("data", {}).get("user", {})

            if user_data:
                return ActionResult(
                    success=True,
                    message=f"Found entity as user",
                    data={
                        "id": user_data.get("id", ""),
                        "login": user_data.get("login", ""),
                        "type": "User"
                    }
                )

            # Try as organization
            org_query = """
            query($login: String!) {
                organization(login: $login) {
                    id
                    login
                    __typename
                }
            }
            """

            response = self.execute_graphql(org_query, variables)
            org_data = response.get("data", {}).get("organization", {})

            if org_data:
                return ActionResult(
                    success=True,
                    message=f"Found entity as organization",
                    data={
                        "id": org_data.get("id", ""),
                        "login": org_data.get("login", ""),
                        "type": "Organization"
                    }
                )

            return ActionResult(
                success=False,
                message=f"Entity {entity_name} not found",
                data={}
            )

        except Exception as e:
            self.logger.error(f"Error getting entity node ID: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Failed to get node ID for {entity_name}: {str(e)}",
                data={}
            )
