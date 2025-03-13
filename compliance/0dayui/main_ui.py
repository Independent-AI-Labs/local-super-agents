import concurrent.futures
import json
import os
import re

import feedparser
import gradio as gr
import pandas as pd
from bs4 import BeautifulSoup

# List of RSS feeds
FEEDS = [
    {"name": "Darknet Diaries", "url": "https://podcast.darknetdiaries.com/"},
    {"name": "Graham Cluley", "url": "https://grahamcluley.com/feed/"},
    {"name": "Krebs on Security", "url": "https://krebsonsecurity.com/feed/"},
    {"name": "SANS Internet Storm Center", "url": "https://isc.sans.edu/rssfeed_full.xml"},
    {"name": "Schneier on Security", "url": "https://www.schneier.com/feed/atom/"},
    {"name": "Securelist", "url": "https://securelist.com/feed/"},
    {"name": "Security Operations – Sophos News", "url": "https://news.sophos.com/en-us/category/security-operations/feed/"},
    {"name": "The Hacker News", "url": "https://feeds.feedburner.com/TheHackersNews?format=xml"},
    {"name": "Threat Research – Sophos News", "url": "https://news.sophos.com/en-us/category/threat-research/feed/"},
    {"name": "Troy Hunt", "url": "https://www.troyhunt.com/rss/"},
    {"name": "USOM Threats", "url": "https://www.usom.gov.tr/rss/tehdit.rss"},
    {"name": "USOM Announcements", "url": "https://www.usom.gov.tr/rss/duyuru.rss"},
    {"name": "WeLiveSecurity", "url": "https://feeds.feedburner.com/eset/blog"}
]

# Cache directory for saved feeds
CACHE_DIR = "feed_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def clean_html(html_text):
    """Remove HTML tags and convert HTML entities to plain text."""
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    if len(text) > 500:
        text = text[:497] + "..."
    return text


def extract_tags(entry):
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


def parse_feed(feed_info):
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
            summary = clean_html(summary)
            content = ''
            if 'content' in entry:
                if isinstance(entry.content, list):
                    content = ''.join([item.get('value', '') for item in entry.content])
                else:
                    content = entry.content
            elif 'content_encoded' in entry:
                content = entry['content_encoded']
            content = clean_html(content)
            if not summary and content:
                summary = content[:500] + "..." if len(content) > 500 else content
            tags = extract_tags(entry)
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


def fetch_all_feeds(max_workers=10):
    """Fetch all feeds concurrently and combine them."""
    all_entries = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_feed = {executor.submit(parse_feed, feed): feed for feed in FEEDS}
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
    return df, sorted(list(all_tags))


def filter_entries(df, search_text="", selected_sources=None, selected_tags=None, date_range=None):
    """Filter entries based on search text, sources, tags, and date range."""
    if df.empty:
        return df
    filtered_df = df.copy()
    if search_text:
        pattern = re.compile(search_text, re.IGNORECASE)
        mask = (
                filtered_df['title'].str.contains(pattern, na=False) |
                filtered_df['summary'].str.contains(pattern, na=False) |
                filtered_df['content'].str.contains(pattern, na=False)
        )
        filtered_df = filtered_df[mask]
    if selected_sources and selected_sources and len(selected_sources) > 0 and 'All Sources' not in selected_sources:
        filtered_df = filtered_df[filtered_df['source'].isin(selected_sources)]
    if selected_tags and selected_tags and len(selected_tags) > 0 and 'All Tags' not in selected_tags:
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


def format_article(row):
    """Format a single article for display."""
    try:
        title = row.get('title', 'No Title')
        source = row.get('source', 'Unknown Source')
        published = row.get('published', 'Unknown Date')
        if isinstance(published, pd.Timestamp):
            published = published.strftime('%Y-%m-%d %H:%M:%S')
        link = row.get('link', '#')
        summary = row.get('summary', 'No summary available')
        tags = row.get('tags', [])
        if isinstance(tags, list):
            tags_str = ", ".join(tags) if tags else "No tags"
        else:
            tags_str = str(tags)
        html = f"""
        <div style="margin-bottom: 20px; padding: 15px; border-radius: 10px; background-color: #f8f9fa; 
                   box-shadow: 0 3px 6px rgba(0,0,0,0.1); transition: transform 0.2s, box-shadow 0.2s;">
            <h3 style="margin-top: 0; color: #2c3e50; font-weight: bold; font-size: 1.3em;">
                <a href="{link}" target="_blank" style="text-decoration: none; color: #3498db; 
                   transition: color 0.2s;">{title}</a>
            </h3>
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between; margin-bottom: 10px; 
                       font-size: 0.9em; color: #7f8c8d;">
                <span style="background: #e0e7ff; padding: 3px 8px; border-radius: 12px; color: #4a5568; 
                            display: inline-block; margin-bottom: 5px;"><strong>Source:</strong> {source}</span>
                <span style="background: #e0fffc; padding: 3px 8px; border-radius: 12px; color: #4a5568; 
                            display: inline-block; margin-bottom: 5px;"><strong>Published:</strong> {published}</span>
            </div>
            <p style="margin-bottom: 15px; color: #34495e; line-height: 1.6;">{summary}</p>
            <div style="font-size: 0.85em; margin-top: 10px;">
                <strong>Tags:</strong> 
                <div style="display: flex; flex-wrap: wrap; gap: 5px; margin-top: 5px;">
                    {" ".join([f'<span style="background: #f0f4f8; padding: 2px 8px; border-radius: 10px; color: #4a5568;">{tag}</span>' for tag in tags]) if isinstance(tags, list) and tags else '<span style="color: #7f8c8d;">No tags</span>'}
                </div>
            </div>
        </div>
        """
        return html
    except Exception as e:
        return f"<div>Error formatting article: {str(e)}</div>"


def create_feed_interface():
    """Create the Gradio interface for the RSS feed application."""
    custom_css = """
    .gradio-container {
        font-family: 'Montserrat', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1200px;
        margin: 0 auto;
    }
    .top-header {
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .source-filter, .tag-filter {
        border-radius: 8px;
        margin-bottom: 10px;
        border: 1px solid #e1e4e8;
    }
    .search-box input {
        border-radius: 20px !important;
        padding: 10px 15px !important;
        border: 1px solid #e1e4e8 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
        font-size: 16px !important;
    }
    .date-input input {
        border-radius: 8px !important;
        padding: 8px 12px !important;
        border: 1px solid #e1e4e8 !important;
    }
    .refresh-btn button {
        background-color: #2ecc71 !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }
    .refresh-btn button:hover {
        background-color: #27ae60 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        transform: translateY(-2px) !important;
    }
    .nav-btn button {
        background-color: #3498db !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 8px 16px !important;
        font-weight: bold !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }
    .nav-btn button:hover {
        background-color: #2980b9 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        transform: translateY(-2px) !important;
    }
    .feed-display {
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .article-counter {
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .page-number input {
        text-align: center !important;
        font-weight: bold !important;
    }
    """

    with gr.Blocks(css=custom_css) as app:
        with gr.Group(elem_classes=["top-header"]):
            gr.Markdown("""
            # 🛡️ Cybersecurity News Tracker
            Stay updated with the latest cybersecurity news from various trusted sources.
            """)

        df_json = gr.State("")

        # Initialize with default values to prevent errors
        source_choices = ["All Sources"]
        tag_choices = ["All Tags"]

        with gr.Row():
            with gr.Column(scale=3):
                search_box = gr.Textbox(
                    placeholder="Search for vulnerabilities, threats, exploits...",
                    label="Search",
                    elem_classes=["search-box"]
                )
                with gr.Row():
                    with gr.Column():
                        source_filter = gr.Dropdown(
                            choices=source_choices,
                            value=[],
                            multiselect=True,
                            label="Sources",
                            elem_classes=["source-filter"]
                        )
                    with gr.Column():
                        tag_filter = gr.Dropdown(
                            choices=tag_choices,
                            value=[],
                            multiselect=True,
                            label="Tags",
                            elem_classes=["tag-filter"]
                        )
                with gr.Row():
                    with gr.Column():
                        start_date_filter = gr.Textbox(
                            label="Start Date (YYYY-MM-DD)",
                            elem_classes=["date-input"]
                        )
                    with gr.Column():
                        end_date_filter = gr.Textbox(
                            label="End Date (YYYY-MM-DD)",
                            elem_classes=["date-input"]
                        )
                article_counter = gr.Markdown("Loading...", elem_classes=["article-counter"])

                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Feeds", elem_classes=["refresh-btn"])
                    prev_btn = gr.Button("◀️ Previous", elem_classes=["nav-btn"])
                    page_number = gr.Number(value=1, label="Page", precision=0, elem_classes=["page-number"])
                    next_btn = gr.Button("Next ▶️", elem_classes=["nav-btn"])

        articles_display = gr.HTML(
            value="<div style='text-align:center; padding:50px;'><h3>Loading articles...</h3><p>Please wait while we fetch the latest cybersecurity news.</p></div>",
            elem_classes=["feed-display"]
        )

        def load_feeds():
            try:
                df, tags = fetch_all_feeds()
                sources = ['All Sources'] + sorted(df['source'].unique().tolist())
                all_tags = ['All Tags'] + tags
                articles_html = ""
                for _, row in df.head(10).iterrows():
                    articles_html += format_article(row)

                if df.empty:
                    articles_html = """
                    <div style="text-align: center; padding: 50px; color: #7f8c8d;">
                        <h3>No articles found</h3>
                        <p>Try refreshing the feeds or check your internet connection.</p>
                    </div>
                    """

                return (
                    df.to_json(),
                    gr.update(choices=sources, value=[]),
                    gr.update(choices=all_tags, value=[]),
                    gr.update(value=""),
                    gr.update(value=""),
                    articles_html,
                    gr.update(value=f"Showing 1-{min(10, len(df))} of {len(df)} articles")
                )
            except Exception as e:
                error_msg = f"Error loading feeds: {str(e)}"
                print(error_msg)
                return (
                    "",
                    gr.update(choices=["All Sources"], value=[]),
                    gr.update(choices=["All Tags"], value=[]),
                    gr.update(value=""),
                    gr.update(value=""),
                    f"""
                    <div style="text-align: center; padding: 50px; color: #e74c3c;">
                        <h3>Error Loading Feeds</h3>
                        <p>{error_msg}</p>
                        <p>Please try refreshing the page or try again later.</p>
                    </div>
                    """,
                    gr.update(value="Error loading feeds")
                )

        def update_display(df_json, search_text, selected_sources, selected_tags, start_date, end_date, page=1, per_page=10):
            if not df_json:
                return """
                    <div style="text-align: center; padding: 50px; color: #7f8c8d;">
                        <h3>No data available</h3>
                        <p>Please refresh the feeds to load articles.</p>
                    </div>
                    """, "No articles to display.", 1
            try:
                df = pd.read_json(df_json)
                date_range = (start_date, end_date)
                filtered_df = filter_entries(df, search_text, selected_sources, selected_tags, date_range)
                total_articles = len(filtered_df)
                total_pages = max(1, (total_articles + per_page - 1) // per_page)
                if page < 1:
                    page = 1
                elif page > total_pages and total_pages > 0:
                    page = total_pages
                start_idx = (page - 1) * per_page
                end_idx = min(start_idx + per_page, total_articles)

                if filtered_df.empty:
                    articles_html = """
                    <div style="text-align: center; padding: 50px; color: #7f8c8d;">
                        <h3>No articles found matching your filters</h3>
                        <p>Try adjusting your search criteria or filter selections.</p>
                    </div>
                    """
                    counter_text = "No articles found"
                else:
                    page_df = filtered_df.iloc[start_idx:end_idx]
                    articles_html = ""
                    for _, row in page_df.iterrows():
                        articles_html += format_article(row)
                    counter_text = f"Showing {start_idx + 1}-{end_idx} of {total_articles} articles (Page {page} of {total_pages})"

                return articles_html, counter_text, page
            except Exception as e:
                error_msg = f"Error updating display: {str(e)}"
                print(error_msg)
                return f"""
                <div style="color: #e74c3c; padding: 20px; text-align: center; background-color: #fef7f7; border-radius: 10px;">
                    <h3>Error Occurred</h3>
                    <p>{error_msg}</p>
                    <p>Try refreshing the feeds or check the console for more details.</p>
                </div>
                """, "Error occurred", 1

        # Load the initial data when the app starts
        app.load(load_feeds, outputs=[
            df_json,
            source_filter,
            tag_filter,
            start_date_filter,
            end_date_filter,
            articles_display,
            article_counter
        ])

        filter_inputs = [df_json, search_box, source_filter, tag_filter, start_date_filter, end_date_filter, page_number]

        search_box.change(
            update_display,
            inputs=filter_inputs,
            outputs=[articles_display, article_counter, page_number]
        )

        source_filter.change(
            update_display,
            inputs=filter_inputs,
            outputs=[articles_display, article_counter, page_number]
        )

        tag_filter.change(
            update_display,
            inputs=filter_inputs,
            outputs=[articles_display, article_counter, page_number]
        )

        start_date_filter.change(
            update_display,
            inputs=filter_inputs,
            outputs=[articles_display, article_counter, page_number]
        )

        end_date_filter.change(
            update_display,
            inputs=filter_inputs,
            outputs=[articles_display, article_counter, page_number]
        )

        refresh_btn.click(
            load_feeds,
            outputs=[
                df_json,
                source_filter,
                tag_filter,
                start_date_filter,
                end_date_filter,
                articles_display,
                article_counter
            ]
        )

        def go_to_prev_page(current_page):
            return max(1, current_page - 1)

        def go_to_next_page(current_page):
            return current_page + 1

        prev_btn.click(
            go_to_prev_page,
            inputs=[page_number],
            outputs=[page_number]
        ).then(
            update_display,
            inputs=filter_inputs,
            outputs=[articles_display, article_counter, page_number]
        )

        next_btn.click(
            go_to_next_page,
            inputs=[page_number],
            outputs=[page_number]
        ).then(
            update_display,
            inputs=filter_inputs,
            outputs=[articles_display, article_counter, page_number]
        )

        page_number.change(
            update_display,
            inputs=filter_inputs,
            outputs=[articles_display, article_counter, page_number]
        )

    return app


if __name__ == "__main__":
    app = create_feed_interface()
    app.launch()
