from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Enum for possible action types."""
    REPLY_TO_USER = "REPLY_TO_USER"
    GET_PROJECT_INFO = "GET_PROJECT_INFO"
    GET_USER_PROJECTS = "GET_USER_PROJECTS"
    GET_ORG_PROJECTS = "GET_ORG_PROJECTS"
    GET_PROJECT_FIELDS = "GET_PROJECT_FIELDS"
    GET_PROJECT_ITEMS = "GET_PROJECT_ITEMS"
    ADD_ITEM_TO_PROJECT = "ADD_ITEM_TO_PROJECT"
    ADD_DRAFT_ISSUE = "ADD_DRAFT_ISSUE"
    UPDATE_PROJECT_SETTINGS = "UPDATE_PROJECT_SETTINGS"
    UPDATE_TEXT_FIELD = "UPDATE_TEXT_FIELD"
    UPDATE_SELECT_FIELD = "UPDATE_SELECT_FIELD"
    UPDATE_ITERATION_FIELD = "UPDATE_ITERATION_FIELD"
    DELETE_PROJECT_ITEM = "DELETE_PROJECT_ITEM"
    CREATE_PROJECT = "CREATE_PROJECT"
    CONVERT_DRAFT_TO_ISSUE = "CONVERT_DRAFT_TO_ISSUE"
    ADD_COMMENT_TO_ISSUE = "ADD_COMMENT_TO_ISSUE"
    CREATE_ISSUE = "CREATE_ISSUE"
    GET_REPOSITORY_ID = "GET_REPOSITORY_ID"


class Action(BaseModel):
    """Model for an action to execute."""
    type: ActionType
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ActionResult(BaseModel):
    """Model for an action execution result."""
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    message: str


class ProjectInfo(BaseModel):
    """Model for project information."""
    id: str
    title: str
    owner: str
    public: bool
    number: int
    description: Optional[str] = None


class ProjectField(BaseModel):
    """Model for a project field."""
    id: str
    name: str
    type: str
    options: Optional[Dict[str, str]] = None
    iterations: Optional[Dict[str, str]] = None


class ProjectItem(BaseModel):
    """Model for a project item."""
    id: str
    type: str
    title: str
    field_values: Dict[str, Any] = Field(default_factory=dict)
    content_id: Optional[str] = None


class GraphQLQuery(BaseModel):
    """Model for GraphQL queries."""
    query: str
    variables: Optional[Dict[str, Any]] = None
