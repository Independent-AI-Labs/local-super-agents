from typing import Dict, Any


class FeedView:
    """View class for rendering feed data and UI components."""

    @staticmethod
    def get_css() -> str:
        """Return the custom CSS for the application."""
        return """
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

    @staticmethod
    def get_header() -> str:
        """Return the header markdown content."""
        return """
        # 🛡️ Cybersecurity News Tracker
        Stay updated with the latest cybersecurity news from various trusted sources.
        """

    @staticmethod
    def format_article(row: Dict[str, Any]) -> str:
        """Format a single article for display.

        Args:
            row: Dictionary containing article data

        Returns:
            HTML string for displaying the article
        """
        try:
            title = row.get('title', 'No Title')
            source = row.get('source', 'Unknown Source')
            published = row.get('published', 'Unknown Date')

            if isinstance(published, str):
                # Keep as is
                pass
            elif hasattr(published, 'strftime'):
                published = published.strftime('%Y-%m-%d %H:%M:%S')

            link = row.get('link', '#')
            summary = row.get('summary', 'No summary available')
            tags = row.get('tags', [])

            if isinstance(tags, list):
                tags_str = ", ".join(tags) if tags else "No tags"
                tags_html = " ".join([
                    f'<span style="background: #f0f4f8; padding: 2px 8px; border-radius: 10px; color: #4a5568;">{tag}</span>'
                    for tag in tags
                ]) if tags else '<span style="color: #7f8c8d;">No tags</span>'
            else:
                tags_str = str(tags)
                tags_html = f'<span style="color: #7f8c8d;">{tags_str}</span>'

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
                        {tags_html}
                    </div>
                </div>
            </div>
            """
            return html
        except Exception as e:
            return f"<div>Error formatting article: {str(e)}</div>"

    @staticmethod
    def format_articles_list(articles_df) -> str:
        """Format a list of articles for display.

        Args:
            articles_df: DataFrame containing articles

        Returns:
            HTML string with all articles formatted
        """
        if articles_df.empty:
            return """
            <div style="text-align: center; padding: 50px; color: #7f8c8d;">
                <h3>No articles found matching your filters</h3>
                <p>Try adjusting your search criteria or filter selections.</p>
            </div>
            """

        articles_html = ""
        for _, row in articles_df.iterrows():
            articles_html += FeedView.format_article(row)

        return articles_html

    @staticmethod
    def loading_message() -> str:
        """Return the loading message HTML."""
        return """
        <div style='text-align:center; padding:50px;'>
            <h3>Loading articles...</h3>
            <p>Please wait while we fetch the latest cybersecurity news.</p>
        </div>
        """

    @staticmethod
    def error_message(error: str) -> str:
        """Return an error message HTML.

        Args:
            error: Error message to display

        Returns:
            HTML string with formatted error message
        """
        return f"""
        <div style="text-align: center; padding: 50px; color: #e74c3c;">
            <h3>Error Loading Feeds</h3>
            <p>{error}</p>
            <p>Please try refreshing the page or try again later.</p>
        </div>
        """

    @staticmethod
    def format_counter_text(start_idx: int, end_idx: int, total_articles: int, page: int, total_pages: int) -> str:
        """Format the counter text displaying pagination info.

        Args:
            start_idx: Index of the first article on the page
            end_idx: Index of the last article on the page
            total_articles: Total number of articles after filtering
            page: Current page number
            total_pages: Total number of pages

        Returns:
            Formatted counter text
        """
        return f"Showing {start_idx + 1}-{end_idx} of {total_articles} articles (Page {page} of {total_pages})"

    @staticmethod
    def no_data_message() -> str:
        """Return message when no data is available."""
        return """
        <div style="text-align: center; padding: 50px; color: #7f8c8d;">
            <h3>No data available</h3>
            <p>Please refresh the feeds to load articles.</p>
        </div>
        """
