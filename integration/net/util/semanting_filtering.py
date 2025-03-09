"""
Optimized semantic filtering module for processing web content.
Integrates enhanced content extraction with natural language processing
and semantic HTML parsing techniques.
"""

import re
from typing import List, Dict
from urllib.parse import urlparse

import nltk
from bs4 import BeautifulSoup
# Import NLP libraries
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from integration.data.config import WEB_SEARCH_MODEL
from integration.net.ollama.ollama_api import prompt_model
from integration.net.parse.semantic_html_parser import SemanticHTMLParser

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Global labels for consistent formatting
LABEL_TITLE = "🠶 TITLE:"
LABEL_HEADINGS = "🠶 HEADINGS:"
LABEL_CONTENT = "🠶 CONTENT:"
LABEL_METADATA = "🠶 METADATA:"
LABEL_WEBSITE_CONTENTS = "🠶 WEBSITE CONTENTS FOR URL="
LABEL_LINKS = "🠶 LINKS FROM WEBSITE:"
LABEL_END_WEBSITE = "🠶 END OF WEBSITE CONTENTS FOR URL="
LABEL_EXTRACTED_LINKS = "🠶 LINKS:"

# Maximum token size for LLM processing
MAX_TOKEN_SIZE = 16000
# Approximate characters per token (for estimation)
CHARS_PER_TOKEN = 4


class EnhancedContentExtractor:
    """
    Enhanced content extractor that uses NLP techniques to identify
    and extract the most relevant content from HTML documents.
    """

    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_df=0.8,  # Ignore terms that appear in more than 80% of documents
            min_df=2  # Ignore terms that appear in fewer than 2 documents
        )

    def extract_content(self, html_doc, url):
        """
        Extract relevant content from HTML using NLP techniques.

        Args:
            html_doc: HTML document as string
            url: URL of the document for context

        Returns:
            str: Extracted content
        """
        # First use basic extraction to get candidate content
        soup = self._remove_unwanted_elements(BeautifulSoup(html_doc, 'html.parser'))

        # Extract all text blocks (paragraphs, divs, etc.)
        blocks = []
        block_elements = soup.find_all(['p', 'div', 'article', 'section', 'main'])

        for block in block_elements:
            text = block.get_text(strip=True)
            if len(text) > 100:  # Only consider substantial blocks
                blocks.append(text)

        if not blocks:
            # Fallback to basic extraction if no substantial blocks found
            return self._basic_extraction(soup)

        # Use NLP to identify the main content
        return self._extract_main_content(blocks, url)

    def _remove_unwanted_elements(self, soup):
        """Remove navigation, headers, footers, scripts, etc."""
        for tag in ['head', 'nav', 'footer', 'header', 'aside', 'script', 'style', 'iframe',
                    'noscript', 'button', 'input', 'meta', 'form', 'svg', 'path']:
            for element in soup.find_all(tag):
                element.decompose()

        # Remove elements with common navigation or sidebar classes/IDs
        nav_patterns = ['nav', 'menu', 'sidebar', 'footer', 'header', 'banner', 'ad', 'widget']
        for pattern in nav_patterns:
            for element in soup.find_all(class_=lambda c: c and pattern in c.lower()):
                element.decompose()
            for element in soup.find_all(id=lambda i: i and pattern in i.lower()):
                element.decompose()

        return soup

    def _basic_extraction(self, soup):
        """Basic content extraction fallback method."""
        main_content = soup.find_all(['article', 'main', 'div'])
        relevant_text = '\n'.join(content.get_text(strip=True) for content in main_content)
        return relevant_text

    def _extract_main_content(self, blocks, url):
        """
        Use NLP techniques to identify the main content.

        1. Calculate TF-IDF scores for each block
        2. Use sentence density and length as features
        3. Score blocks based on semantic relevance to the URL
        """
        if len(blocks) == 1:
            return blocks[0]

        # Compute TF-IDF matrix
        try:
            tfidf_matrix = self.vectorizer.fit_transform(blocks)
            tfidf_scores = tfidf_matrix.toarray().sum(axis=1)
        except ValueError:
            # If vectorization fails, fall back to length-based scoring
            tfidf_scores = [len(block) for block in blocks]

        # Calculate sentence density (sentences per character)
        sentence_density = []
        for block in blocks:
            sentences = sent_tokenize(block)
            density = len(sentences) / (len(block) + 1)  # Add 1 to avoid division by zero
            sentence_density.append(density)

        # Extract domain from URL for relevance scoring
        try:
            domain = urlparse(url).netloc
            domain_terms = set(domain.lower().split('.')[:-1])  # Ignore TLD
        except:
            domain_terms = set()

        # Score blocks based on domain relevance
        domain_relevance = []
        for block in blocks:
            words = set(w.lower() for w in block.split() if w.lower() not in self.stop_words)
            relevance = len(words.intersection(domain_terms)) / (len(words) + 1)
            domain_relevance.append(relevance)

        # Combine scores with appropriate weights
        combined_scores = [
            (0.5 * tfidf + 0.3 * density + 0.2 * relevance + 0.1 * len(block))
            for tfidf, density, relevance, block in zip(tfidf_scores, sentence_density, domain_relevance, blocks)
        ]

        # Sort blocks by score in descending order
        ranked_blocks = [block for _, block in sorted(
            zip(combined_scores, blocks), key=lambda x: x[0], reverse=True
        )]

        # Return top blocks that together constitute a substantial amount of content
        output = []
        total_length = 0
        target_length = sum(len(block) for block in blocks) * 0.7  # Target 70% of total content

        for block in ranked_blocks:
            output.append(block)
            total_length += len(block)
            if total_length >= target_length:
                break

        return '\n\n'.join(output)


def extract_web_content_with_semantics(html_doc, url):
    """
    Extract web content using a combination of semantic HTML parsing
    and enhanced NLP-based content extraction.

    Args:
        html_doc: HTML document as string
        url: URL of the document

    Returns:
        dict: Structured content with semantic information
    """
    # Basic cleanup first
    soup = BeautifulSoup(html_doc, 'html.parser')

    # Apply semantic parsing
    parser = SemanticHTMLParser()
    semantic_data = parser.parse(html_doc)

    # Enhanced content extraction
    extractor = EnhancedContentExtractor()
    enhanced_content = extractor.extract_content(html_doc, url)

    # Add enhanced content to semantic data
    semantic_data['enhanced_content'] = enhanced_content

    # Add URL to metadata
    if 'metadata' not in semantic_data:
        semantic_data['metadata'] = {}
    semantic_data['metadata']['url'] = url

    return semantic_data


def format_semantic_content(semantic_data):
    """
    Format semantic data into a searchable and displayable string.

    Args:
        semantic_data: Structured semantic data

    Returns:
        str: Formatted content
    """
    parts = []

    # Add title
    if 'metadata' in semantic_data and 'title' in semantic_data['metadata']:
        parts.append(f"{LABEL_TITLE} {semantic_data['metadata']['title']}")

    # Add headings in hierarchy
    if 'headings' in semantic_data and semantic_data['headings']:
        parts.append(LABEL_HEADINGS)
        for heading in semantic_data['headings']:
            indent = "  " * (heading['level'] - 1)
            parts.append(f"{indent}{heading['text']}")

    # Add enhanced content first (preferred)
    if 'enhanced_content' in semantic_data and semantic_data['enhanced_content']:
        parts.append(LABEL_CONTENT)
        parts.append(semantic_data['enhanced_content'])
    # Fallback to main_content if enhanced_content is not available or empty
    elif 'main_content' in semantic_data and semantic_data['main_content']:
        parts.append(LABEL_CONTENT)
        parts.append(semantic_data['main_content'])

    # Add metadata
    if 'metadata' in semantic_data:
        parts.append(LABEL_METADATA)
        for key, value in semantic_data['metadata'].items():
            if key != 'structured_data' and not isinstance(value, dict):
                parts.append(f"  {key}: {value}")

    return "\n\n".join(parts)


def _extract_sections(content: str) -> Dict[str, str]:
    """
    Helper function to extract specific sections from content.

    Args:
        content (str): The content to extract sections from

    Returns:
        Dict[str, str]: A dictionary with section labels as keys and content as values
    """
    sections = {}

    # Extract URL
    url_match = re.search(fr'{LABEL_WEBSITE_CONTENTS}(.+?):', content)
    if url_match:
        sections['url'] = url_match.group(1)

    # Extract Title
    title_start = content.find(LABEL_TITLE)
    headings_start = content.find(LABEL_HEADINGS)

    if title_start != -1:
        title_end = headings_start if headings_start != -1 else content.find(LABEL_CONTENT)
        if title_end != -1:
            title_content = content[title_start + len(LABEL_TITLE):title_end].strip()
            sections[LABEL_TITLE] = title_content

    # Extract Headings
    content_start = content.find(LABEL_CONTENT)

    if headings_start != -1 and content_start != -1:
        headings_content = content[headings_start + len(LABEL_HEADINGS):content_start].strip()
        sections[LABEL_HEADINGS] = headings_content

    # Extract Content
    links_start = content.find(LABEL_LINKS)
    end_website = content.find(LABEL_END_WEBSITE)

    content_end = links_start if links_start != -1 else end_website
    if content_end == -1:
        content_end = len(content)

    if content_start != -1:
        main_content = content[content_start + len(LABEL_CONTENT):content_end].strip()
        sections[LABEL_CONTENT] = main_content

    return sections


def _is_similar_content(content1: str, content2: str, threshold: float = 0.8) -> bool:
    """
    Check if two content strings are similar based on shared words and phrases.

    Args:
        content1 (str): First content string
        content2 (str): Second content string
        threshold (float): Similarity threshold (0.0 to 1.0)

    Returns:
        bool: True if contents are similar, False otherwise
    """
    # First, check for exact duplication or near-exact duplication
    if content1 == content2:
        return True

    # Also check if either string is mostly contained within the other
    # This helps catch cases where one extract is a subset of another
    if len(content1) > 3 * len(content2) or len(content2) > 3 * len(content1):
        shorter = content1 if len(content1) < len(content2) else content2
        longer = content2 if len(content1) < len(content2) else content1

        # If 90% of the shorter content appears in the longer content, consider it similar
        if shorter and longer and shorter in longer:
            return True

    # Extract words from both contents (lowercase for case-insensitive comparison)
    words1 = list(re.findall(r'\b\w+\b', content1.lower()))
    words2 = list(re.findall(r'\b\w+\b', content2.lower()))

    # If either has very few words, use a more lenient approach
    if len(words1) < 10 or len(words2) < 10:
        # For very short content, check if most words from the shorter appear in the longer
        shorter_words = words1 if len(words1) < len(words2) else words2
        longer_words = words2 if len(words1) < len(words2) else words1

        common_words = sum(1 for word in shorter_words if word in longer_words)
        return (common_words / len(shorter_words)) >= threshold if shorter_words else False

    # For longer content, use n-gram comparison for better phrase matching
    def get_ngrams(words, n=3):
        return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]

    # Get 3-grams (phrases of 3 consecutive words)
    ngrams1 = set(get_ngrams(words1))
    ngrams2 = set(get_ngrams(words2))

    # Calculate Jaccard similarity on n-grams
    if ngrams1 and ngrams2:
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        ngram_similarity = intersection / union if union > 0 else 0

        # If n-gram similarity is high, content is similar
        if ngram_similarity >= threshold:
            return True

    # Fall back to word-level Jaccard similarity for additional check
    words1_set = set(words1)
    words2_set = set(words2)

    # Filter out very common words (expanded list)
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'is', 'are',
                    'this', 'that', 'these', 'those', 'it', 'they', 'them', 'their', 'we', 'us', 'our', 'you', 'your',
                    'he', 'she', 'his', 'her', 'has', 'have', 'had', 'was', 'were', 'been', 'be', 'as', 'from', 'will'}

    words1_set = words1_set - common_words
    words2_set = words2_set - common_words

    # Calculate Jaccard similarity if there are words to compare
    if not words1_set or not words2_set:
        return False

    intersection = len(words1_set.intersection(words2_set))
    union = len(words1_set.union(words2_set))

    word_similarity = intersection / union if union > 0 else 0

    return word_similarity >= threshold


def _deduplicate_results(results: List[str]) -> List[str]:
    """
    Deduplicate results based on content similarity.

    Args:
        results (List[str]): List of content strings

    Returns:
        List[str]: Deduplicated results
    """
    if not results:
        return []

    # First, handle internal duplication within each result
    deduped_individual_results = []
    for content in results:
        deduped_content = _deduplicate_internal_content(content)
        deduped_individual_results.append(deduped_content)

    # Then, handle duplication across different results
    deduplicated = []
    seen_urls = set()
    seen_content_signatures = set()

    for content in deduped_individual_results:
        # Extract URL and check if we've seen it before
        sections = _extract_sections(content)

        # Skip if we've seen this URL before
        if 'url' in sections and sections['url'] in seen_urls:
            continue

        # Add URL to seen URLs if it exists
        if 'url' in sections:
            seen_urls.add(sections['url'])

        # Generate a content signature for similarity detection
        content_signature = ""
        if LABEL_CONTENT in sections:
            # Create a signature from the first 100 words and last 100 words
            words = re.findall(r'\b\w+\b', sections[LABEL_CONTENT].lower())
            if len(words) > 200:
                content_signature = " ".join(words[:100]) + " " + " ".join(words[-100:])
            else:
                content_signature = " ".join(words)

        # Check if we've seen similar content before
        if content_signature and content_signature in seen_content_signatures:
            continue

        # Check for content similarity with existing results
        is_duplicate = False
        for existing in deduplicated:
            existing_sections = _extract_sections(existing)

            # Compare main content sections if they exist
            if (LABEL_CONTENT in existing_sections and
                    LABEL_CONTENT in sections and
                    _is_similar_content(existing_sections[LABEL_CONTENT], sections[LABEL_CONTENT])):
                is_duplicate = True
                break

        if not is_duplicate:
            # Add to deduplicated results and track the content signature
            deduplicated.append(content)
            if content_signature:
                seen_content_signatures.add(content_signature)

    return deduplicated


def _deduplicate_internal_content(content: str) -> str:
    """
    Remove duplicate blocks of content within a single result.

    Args:
        content (str): The content string to deduplicate internally

    Returns:
        str: Content with internal duplications removed
    """
    # Handle the case of repeated blocks like in the MIT news example
    sections = _extract_sections(content)

    if LABEL_CONTENT in sections:
        main_content = sections[LABEL_CONTENT]

        # Split content into paragraphs or natural blocks
        blocks = re.split(r'\n\s*\n', main_content)

        # Check for repeating patterns of blocks
        if len(blocks) > 5:  # Only check if there are enough blocks to potentially have repetition
            # Look for repetitive patterns by comparing chunks of the content
            chunk_size = len(blocks) // 2

            if chunk_size >= 3:  # Need at least a few blocks to identify a pattern
                first_chunk = blocks[:chunk_size]
                second_chunk = blocks[chunk_size:2 * chunk_size]

                # Check if the chunks are similar
                similarity = sum(1 for a, b in zip(first_chunk, second_chunk) if _is_similar_content(a, b, 0.7))

                if similarity / chunk_size > 0.7:  # If 70% of the blocks are similar, we have a repeating pattern
                    # Keep only the first chunk
                    sections[LABEL_CONTENT] = "\n\n".join(first_chunk)

        # Also detect and remove exact duplicate paragraphs
        unique_blocks = []
        seen_block_signatures = set()

        for block in blocks:
            # Create a signature for this block (first 10 words)
            words = re.findall(r'\b\w+\b', block.lower())
            block_signature = " ".join(words[:min(10, len(words))])

            # Check if we've seen this signature before
            if block_signature in seen_block_signatures:
                # Check if this block is very similar to any block we've already kept
                is_duplicate = any(_is_similar_content(block, existing_block, 0.8) for existing_block in unique_blocks)
                if is_duplicate:
                    continue

            # Add this block and its signature
            unique_blocks.append(block)
            seen_block_signatures.add(block_signature)

        # Rebuild the content with unique blocks only
        sections[LABEL_CONTENT] = "\n\n".join(unique_blocks)

    # Reconstruct the content with the deduplicated sections
    result_parts = []

    if 'url' in sections:
        result_parts.append(f"URL: {sections['url']}")

    for label in [LABEL_TITLE, LABEL_HEADINGS, LABEL_CONTENT]:
        if label in sections and sections[label]:
            result_parts.append(f"{label}\n{sections[label]}")

    return "\n\n".join(result_parts)


def _chunk_content(content: List[str], max_tokens: int = MAX_TOKEN_SIZE) -> List[str]:
    """
    Split content into chunks that don't exceed the maximum token size.

    Args:
        content (List[str]): List of content strings
        max_tokens (int): Maximum number of tokens per chunk

    Returns:
        List[str]: Content chunks
    """
    chunks = []
    current_chunk = []
    current_size = 0
    max_chars = max_tokens * CHARS_PER_TOKEN  # Approximate conversion

    for item in content:
        item_size = len(item)

        if current_size + item_size <= max_chars:
            current_chunk.append(item)
            current_size += item_size
        else:
            # If current item is too large for a new chunk, split it
            if item_size > max_chars:
                # Add current chunk to chunks if not empty
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Split large item into multiple chunks
                remaining = item
                while remaining:
                    chunk_text = remaining[:max_chars]
                    chunks.append(chunk_text)
                    remaining = remaining[max_chars:]
            else:
                # Add current chunk to chunks and start a new one with the current item
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [item]
                current_size = item_size

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def _create_crystallization_prompt(content: str, semantic_patterns: List[str], instructions: str) -> str:
    """
    Create a prompt for the LLM to crystallize the information.

    Args:
        content (str): The content to crystallize
        semantic_patterns (List[str]): Patterns to focus on
        instructions (str): Custom instructions

    Returns:
        str: The prompt for the LLM
    """
    patterns_str = ", ".join(f'"{p}"' for p in semantic_patterns) if semantic_patterns else "any relevant information"

    prompt = f"""
You are a web content analyzer tasked with extracting and crystallizing the most relevant information.

Your task is to analyze the following web content and extract information related to {patterns_str}.

{instructions if instructions else "Extract the most important and relevant information."}

Return ONLY the crystallized content maintaining the original structure with section labels.

Preserve the semantic structure and important details while condensing redundant information.

WEB CONTENT:
{content}
"""
    return prompt


def digest_scraped_data(results: List[str], semantic_patterns: List[str] = None, instructions: str = None) -> List[str]:
    """
    Filter and refine the scraped data based on semantic patterns and instructions.

    1. Deduplicates the result data
    2. Calls prompt_model on 16K token chunks of results to iteratively crystallize information
    3. Handles only specific sections: TITLE, HEADINGS, and CONTENT

    Args:
        results (List[str]): List of scraped website content strings
        semantic_patterns (List[str]): Optional list of patterns to match in the content
        instructions (str): Optional specific filtering instructions

    Returns:
        List[str]: Filtered and crystallized content results
    """
    if not results:
        return []

    # Step 1: Deduplicate results
    deduplicated_results = _deduplicate_results(results)

    # Step 2: Filter results based on semantic patterns (basic filtering)
    filtered_results = []

    for content in deduplicated_results:
        sections = _extract_sections(content)

        # Check if content matches semantic patterns if specified
        if semantic_patterns:
            matched = False
            content_to_search = sections.get(LABEL_CONTENT, "").lower()
            title_to_search = sections.get(LABEL_TITLE, "").lower()
            headings_to_search = sections.get(LABEL_HEADINGS, "").lower()

            # Combine all searchable content
            all_searchable = f"{title_to_search} {headings_to_search} {content_to_search}"

            # Check each pattern
            for pattern in semantic_patterns:
                if pattern.lower() in all_searchable:
                    matched = True
                    break

            if not matched:
                continue

        # Keep only the sections we care about
        filtered_content = []

        if 'url' in sections:
            filtered_content.append(f"URL: {sections['url']}")

        for label in [LABEL_TITLE, LABEL_HEADINGS, LABEL_CONTENT]:
            if label in sections and sections[label]:
                filtered_content.append(f"{label}\n{sections[label]}")

        filtered_results.append("\n\n".join(filtered_content))

    # Step 3: Chunk the filtered results for LLM processing
    chunked_results = _chunk_content(filtered_results)

    # Step 4: Crystallize each chunk using the LLM
    crystallized_results = []

    for chunk in chunked_results:
        prompt = _create_crystallization_prompt(chunk, semantic_patterns, instructions)

        try:
            # Call the LLM to crystallize the information
            crystallized_chunk = prompt_model(prompt, model=WEB_SEARCH_MODEL)
            crystallized_results.append(crystallized_chunk)
            print(crystallized_chunk)
        except Exception as e:
            # If LLM processing fails, keep the original chunk
            print(f"Error crystallizing content: {str(e)}")
            crystallized_results.append(chunk)

    # Optional: If there are multiple crystallized chunks, we could crystallize them again
    # to ensure they fit within the token limit and are properly combined
    if len(crystallized_results) > 1 and all(len(result) < MAX_TOKEN_SIZE * CHARS_PER_TOKEN for result in crystallized_results):
        combined = "\n\n---\n\n".join(crystallized_results)
        if len(combined) > MAX_TOKEN_SIZE * CHARS_PER_TOKEN:
            # Further crystallize the combined results if they're too large
            final_prompt = f"""
Finalize the crystallization of these web search results.
Consolidate the information while preserving key insights.
Focus on information related to {', '.join(semantic_patterns) if semantic_patterns else 'the main topics'}.

{combined}
"""
            try:
                final_result = prompt_model(final_prompt, model=WEB_SEARCH_MODEL)
                return [final_result]
            except Exception as e:
                print(f"Error in final crystallization: {str(e)}")
                return crystallized_results

    return crystallized_results
