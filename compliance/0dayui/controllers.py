from typing import Dict, List, Tuple

import pandas as pd

from models import FeedModel
from views import FeedView


class FeedController:
    """Controller class for handling the application logic."""

    def __init__(self, feeds_list: List[Dict[str, str]]):
        """Initialize the controller with a feed model and view.

        Args:
            feeds_list: List of dictionaries with 'name' and 'url' keys
        """
        self.model = FeedModel(feeds_list)
        self.view = FeedView()

    def load_feeds(self) -> Tuple[str, Dict, Dict, Dict, Dict, str, Dict]:
        """Load feeds and prepare the UI data.

        Returns:
            Tuple containing data for updating the UI
        """
        try:
            df, tags = self.model.fetch_all_feeds()
            sources = ['All Sources'] + sorted(df['source'].unique().tolist())
            all_tags = ['All Tags'] + tags

            if df.empty:
                articles_html = self.view.no_data_message()
                counter_text = "No articles found"
            else:
                # Display first page of articles
                page_df, total_articles, total_pages = self.model.get_page(df)
                articles_html = self.view.format_articles_list(page_df)
                start_idx = 0
                end_idx = min(10, len(df))
                counter_text = self.view.format_counter_text(
                    start_idx, end_idx, total_articles, 1, total_pages
                )

            return (
                df.to_json(),
                {"choices": sources, "value": []},
                {"choices": all_tags, "value": []},
                {"value": ""},
                {"value": ""},
                articles_html,
                {"value": counter_text}
            )

        except Exception as e:
            error_msg = f"Error loading feeds: {str(e)}"
            print(error_msg)

            return (
                "",
                {"choices": ["All Sources"], "value": []},
                {"choices": ["All Tags"], "value": []},
                {"value": ""},
                {"value": ""},
                self.view.error_message(error_msg),
                {"value": "Error loading feeds"}
            )

    def update_display(self,
                       df_json: str,
                       search_text: str,
                       selected_sources: List[str],
                       selected_tags: List[str],
                       start_date: str,
                       end_date: str,
                       page: int = 1,
                       per_page: int = 10) -> Tuple[str, str, int]:
        """Update the display based on filters and pagination.

        Args:
            df_json: JSON string of the DataFrame
            search_text: Text to search for
            selected_sources: List of selected sources
            selected_tags: List of selected tags
            start_date: Start date filter
            end_date: End date filter
            page: Current page number
            per_page: Number of articles per page

        Returns:
            Tuple of (articles_html, counter_text, page_number)
        """
        if not df_json:
            return self.view.no_data_message(), "No articles to display.", 1

        try:
            df = pd.read_json(df_json)
            date_range = (start_date, end_date)
            filtered_df = self.model.filter_entries(search_text, selected_sources, selected_tags, date_range)

            if filtered_df.empty:
                return self.view.format_articles_list(filtered_df), "No articles found", 1

            page_df, total_articles, total_pages = self.model.get_page(filtered_df, page, per_page)
            start_idx = (page - 1) * per_page
            end_idx = min(start_idx + per_page, total_articles)

            articles_html = self.view.format_articles_list(page_df)
            counter_text = self.view.format_counter_text(
                start_idx, end_idx, total_articles, page, total_pages
            )

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

    def go_to_prev_page(self, current_page: int) -> int:
        """Go to the previous page.

        Args:
            current_page: Current page number

        Returns:
            New page number
        """
        return max(1, current_page - 1)

    def go_to_next_page(self, current_page: int) -> int:
        """Go to the next page.

        Args:
            current_page: Current page number

        Returns:
            New page number
        """
        return current_page + 1
