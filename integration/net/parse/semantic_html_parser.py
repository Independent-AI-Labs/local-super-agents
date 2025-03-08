import json

import nltk
from bs4 import BeautifulSoup

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


class SemanticHTMLParser:
    """
    Parser that understands semantic HTML5 tags and their importance
    for extracting structurally meaningful content.
    """

    def __init__(self):
        # Define semantic HTML5 tags and their importance
        self.semantic_tags = {
            'article': 10,  # Highest importance - self-contained composition
            'main': 9,  # Main content of the document
            'section': 8,  # Thematic grouping of content
            'h1': 7,  # Primary heading
            'h2': 6,  # Secondary heading
            'h3': 5,  # Tertiary heading
            'p': 4,  # Paragraph
            'ul': 3,  # Unordered list
            'ol': 3,  # Ordered list
            'li': 2,  # List item
            'blockquote': 4,  # Quotation
            'figure': 3,  # Self-contained content
            'figcaption': 2,  # Caption for figure
            'details': 3,  # Details/summary widget
            'summary': 3,  # Summary part of details
            'time': 2,  # Time designation
            'mark': 2,  # Marked/highlighted text
            'dl': 3,  # Description list
            'dt': 2,  # Term in description list
            'dd': 2,  # Description in description list
            'table': 3,  # Table
            'th': 2,  # Table header
            'tr': 1,  # Table row
            'td': 1  # Table cell
        }

    def parse(self, html_doc):
        """
        Parse HTML content with awareness of semantic tags.

        Args:
            html_doc: HTML document as string

        Returns:
            dict: Structured content with semantic information
        """
        soup = BeautifulSoup(html_doc, 'html.parser')

        # Extract semantic structure
        semantic_structure = self._extract_semantic_structure(soup)

        # Extract main content based on semantic structure
        main_content = self._extract_main_content(semantic_structure)

        # Extract headings hierarchy
        headings = self._extract_headings(soup)

        # Extract metadata
        metadata = self._extract_metadata(soup)

        return {
            'main_content': main_content,
            'headings': headings,
            'metadata': metadata,
            'semantic_structure': semantic_structure
        }

    def _extract_semantic_structure(self, soup):
        """Extract the semantic structure of the document."""
        structure = []

        # Process body or root if no body
        body = soup.body or soup

        # Process direct children of body
        for child in body.children:
            if hasattr(child, 'name') and child.name in self.semantic_tags:
                processed = self._process_semantic_element(child)
                if processed:
                    structure.append(processed)

        return structure

    def _process_semantic_element(self, element, depth=0):
        """Process a semantic element and its children recursively."""
        if not element.name:
            return None

        importance = self.semantic_tags.get(element.name, 0)
        text_content = element.get_text(strip=True)

        # Skip elements with no text
        if not text_content:
            return None

        result = {
            'tag': element.name,
            'importance': importance,
            'depth': depth,
            'text': text_content,
            'attrs': dict(element.attrs) if element.attrs else {},
            'children': []
        }

        # Process children recursively
        for child in element.children:
            if hasattr(child, 'name') and child.name in self.semantic_tags:
                child_data = self._process_semantic_element(child, depth + 1)
                if child_data:
                    result['children'].append(child_data)

        return result

    def _extract_main_content(self, semantic_structure):
        """Extract main content based on semantic importance."""
        main_content = []

        # Helper function to traverse the structure and find important content
        def extract_content(structure, min_importance=3):
            for item in structure:
                if item and 'importance' in item and item['importance'] >= min_importance:
                    main_content.append(item['text'])

                if item and 'children' in item and item['children']:
                    extract_content(item['children'], min_importance)

        extract_content(semantic_structure)
        return '\n\n'.join(main_content)

    def _extract_headings(self, soup):
        """Extract the hierarchy of headings."""
        headings = []
        for i in range(1, 7):  # h1 through h6
            for heading in soup.find_all(f'h{i}'):
                headings.append({
                    'level': i,
                    'text': heading.get_text(strip=True)
                })

        return headings

    def _extract_metadata(self, soup):
        """Extract metadata from the document."""
        metadata = {}

        # Extract title
        title_tag = soup.title
        if title_tag:
            metadata['title'] = title_tag.get_text(strip=True)

        # Extract meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', meta.get('property', ''))
            content = meta.get('content', '')
            if name and content:
                metadata[name] = content

        # Extract structured data
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                json_data = json.loads(script.string)
                metadata['structured_data'] = json_data
            except (json.JSONDecodeError, TypeError):
                pass

        return metadata
