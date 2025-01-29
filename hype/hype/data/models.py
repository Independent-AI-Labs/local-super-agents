from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


# Base model allowing arbitrary types for flexibility
class BaseModelWithArbitraryTypes(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class FileNode:
    """
    Represents a node in a file system tree with the ability to serialize and deserialize.
    """

    def __init__(self, path: str, size: int):
        self.path = path
        self.size = size
        self.children: Dict[str, 'FileNode'] = {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the FileNode object to a dictionary.
        """
        return {
            'path': self.path,
            'size': self.size,
            'children': {k: v.to_dict() for k, v in self.children.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileNode':
        """
        Deserializes a dictionary to a FileNode object.
        """
        node = cls(data['path'], data['size'])
        node.children = {k: cls.from_dict(v) for k, v in data['children'].items()}
        return node


class SearchTerm(BaseModelWithArbitraryTypes):
    """
    Represents a search term and its corresponding metadata.
    """
    order: int = Field(0, description="The greater the order the lesser the scoring significance of this term")
    text: str = Field(..., description="The text of the search term")
    ref_ids: Optional[List[str]] = Field(None, description="Reference IDs for tracking purposes such as in batch searches")


class StructuredMatch(BaseModelWithArbitraryTypes):
    """
    Represents a match found in structured data.
    """
    item_index: int = Field(..., description="Index of the item in the structured data")
    item_content: str = Field(..., description="Content of the matched item")


class FileMatch(BaseModelWithArbitraryTypes):
    """
    Represents matches found in a file, including line numbers and contextual information.
    """
    line_numbers: List[int] = Field(..., description="List of line numbers where matches were found")
    title: str = Field(..., description="Title of the file or the match context")
    matches_with_context: List[str] = Field(..., description="List of matched strings with surrounding context")


class CommonData(BaseModelWithArbitraryTypes):
    """
    Represents common data associated with a search result, including URI and scoring information.
    """
    uri: str = Field(..., description="Uniform Resource Identifier of the file or resource")
    search_term_match_counts: Dict[str, int] = Field({}, description="Dictionary of matched patterns and their counts")
    matched_search_terms: List[SearchTerm] = Field([], description="All matches for the search result")
    score: float = Field(0, description="Score calculated based on the search results")


class UnifiedSearchResult(BaseModelWithArbitraryTypes):
    """
    Represents a unified search result, including both structured and file matches.
    """
    structured_matches: StructuredMatch = Field(default=None, description="Match found in structured data")
    file_matches: FileMatch = Field(default=None, description="Match found in a file")
    common: CommonData = Field(..., description="Common data associated with the search result")
