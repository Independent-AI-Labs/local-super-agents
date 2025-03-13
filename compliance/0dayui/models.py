import concurrent.futures
import json
import os
import re
from typing import Dict, List, Tuple, Any, Optional

import feedparser
import pandas as pd
from bs4 import BeautifulSoup

# Cache directory for saved feeds
CACHE_DIR = "feed_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


class FeedModel:
    def __init__(self, feeds_list: List[Dict[str, str]]):
        """Initialize the feed model with a list of RSS feeds.

        Args:
            feeds_list: List of dictionaries with 'name' and 'url' keys
        """
        self.feeds = feeds_list
        self.data_frame = pd.DataFrame()
        self.all_tags = []

    def clean_html(self, html_text: str) -> str:
        """Remove HTML tags and convert HTML entities to plain text."""
        if not html_text:
            return ""
        soup = BeautifulSoup(html_text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        if len(text) > 500:
            text = text[:497] + "..."
        return text

    def extract_tags(self, entry: Dict[str, Any]) -> List[str]:
        """Extract tags/categories from an entry if available."""
        tags = []
        if 'tags' in entry:
            tags.extend([tag.get('term', '') for tag in entry['tags']])
        elif 'category' in entry:
            if isinstance(entry['category'], list):
                tags.extend(entry['category'])
            else:
                tags.append(entry['category'])
        elif 'categories' in entry:
            tags.extend(entry['categories'])

        cleaned_tags = []
        for tag in tags:
            if isinstance(tag, str):
                tag_str = tag.strip()
                if tag_str and tag_str not in cleaned_tags:
                    cleaned_tags.append(tag_str)
        return cleaned_tags

    def parse_feed(self, feed_info: Dict[str, str]) -> List[Dict[str, Any]]:
        """Parse a single RSS feed and extract relevant information."""
        feed_name = feed_info["name"]
        feed_url = feed_info["url"]

        cache_file = os.path.join(CACHE_DIR, f"{feed_name.replace(' ', '_').lower()}.json")

        try:
            feed = feedparser.parse(feed_url)
            if feed.status != 200:
                print(f"Failed to fetch {feed_name}: HTTP {feed.status}")
                if os.path.exists(cache_file):
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                return []

            entries = []
            for entry in feed.entries:
                title = entry.get('title', 'No title')
                link = entry.get('link', '#')
                published = entry.get('published', entry.get('pubDate', entry.get('updated', 'Unknown')))

                try:
                    pub_date = pd.to_datetime(published)
                    published = pub_date.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    pass

                summary = entry.get('summary', entry.get('description', ''))
                summary = self.clean_html(summary)

                content = ''
                if 'content' in entry:
                    if isinstance(entry.content, list):
                        content = ''.join([item.get('value', '') for item in entry.content])
                    else:
                        content = entry.content
                elif 'content_encoded' in entry:
                    content = entry['content_encoded']

                content = self.clean_html(content)
                if not summary and content:
                    summary = content[:500] + "..." if len(content) > 500 else content

                tags = self.extract_tags(entry)
                author = entry.get('author', 'Unknown')

                entries.append({
                    'source': feed_name,
                    'title': title,
                    'link': link,
                    'published': published,
                    'summary': summary,
                    'content': content,
                    'tags': tags,
                    'author': author
                })

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(entries, f, ensure_ascii=False)
            return entries

        except Exception as e:
            print(f"Error parsing {feed_name}: {str(e)}")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []

    def fetch_all_feeds(self, max_workers: int = 10) -> Tuple[pd.DataFrame, List[str]]:
        """Fetch all feeds concurrently and combine them."""
        all_entries = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_feed = {executor.submit(self.parse_feed, feed): feed for feed in self.feeds}
            for future in concurrent.futures.as_completed(future_to_feed):
                feed = future_to_feed[future]
                try:
                    entries = future.result()
                    all_entries.extend(entries)
                    print(f"Fetched {len(entries)} entries from {feed['name']}")
                except Exception as e:
                    print(f"Exception processing {feed['name']}: {str(e)}")

        df = pd.DataFrame(all_entries)
        if not df.empty:
            try:
                df['published'] = pd.to_datetime(df['published'], errors='coerce')
                df = df.sort_values('published', ascending=False)
            except Exception as e:
                print(f"Error sorting by date: {str(e)}")

        all_tags = set()
        for tags_list in df['tags'].tolist():
            if isinstance(tags_list, list):
                all_tags.update(tags_list)

        self.data_frame = df
        self.all_tags = sorted(list(all_tags))

        return df, self.all_tags

    def filter_entries(self,
                       search_text: str = "",
                       selected_sources: Optional[List[str]] = None,
                       selected_tags: Optional[List[str]] = None,
                       date_range: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
        """Filter entries based on search text, sources, tags, and date range."""
        if self.data_frame.empty:
            return self.data_frame

        filtered_df = self.data_frame.copy()

        if search_text:
            pattern = re.compile(search_text, re.IGNORECASE)
            mask = (
                    filtered_df['title'].str.contains(pattern, na=False) |
                    filtered_df['summary'].str.contains(pattern, na=False) |
                    filtered_df['content'].str.contains(pattern, na=False)
            )
            filtered_df = filtered_df[mask]

        if selected_sources and len(selected_sources) > 0 and 'All Sources' not in selected_sources:
            filtered_df = filtered_df[filtered_df['source'].isin(selected_sources)]

        if selected_tags and len(selected_tags) > 0 and 'All Tags' not in selected_tags:
            mask = filtered_df['tags'].apply(
                lambda x: any(tag in x for tag in selected_tags) if isinstance(x, list) else False
            )
            filtered_df = filtered_df[mask]

        if date_range:
            start_date, end_date = date_range
            if start_date and end_date:
                try:
                    filtered_df = filtered_df[
                        (filtered_df['published'] >= pd.to_datetime(start_date)) &
                        (filtered_df['published'] <= pd.to_datetime(end_date))
                        ]
                except Exception as e:
                    print(f"Date filtering error: {e}")

        return filtered_df

    def get_page(self, filtered_df: pd.DataFrame, page: int = 1, per_page: int = 10) -> Tuple[pd.DataFrame, int, int]:
        """Get a specific page of articles after filtering."""
        total_articles = len(filtered_df)
        total_pages = max(1, (total_articles + per_page - 1) // per_page)

        if page < 1:
            page = 1
        elif page > total_pages and total_pages > 0:
            page = total_pages

        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, total_articles)

        page_df = filtered_df.iloc[start_idx:end_idx] if not filtered_df.empty else filtered_df

        return page_df, total_articles, total_pages
