from utils.api_client import TDXClient
from core.schemas import TdxArticle
from typing import List, Dict, Any, Tuple
from pydantic import HttpUrl
from core.storage_raw import PostgresClient
from datetime import datetime


class ArticleProcessor:
    """
    Orchestrates the ingestion of articles from TDX API into the database.

    Responsibilities:
    - Compare API state vs database state
    - Identify new, updated, and unchanged articles
    - Filter and transform articles according to business rules
    - Coordinate between API client and storage layer
    """

    def __init__(self):
        self.tdx_client = TDXClient()
        self.db_client = PostgresClient()

    def sync_articles(self) -> Dict[str, List[Any]]:
        """
        Main sync operation: compare API vs DB and return categorized articles.

        Returns:
            Dict with keys:
                - 'new': Articles that don't exist in DB
                - 'updated': Articles that exist but have newer modified dates
                - 'unchanged': Articles that haven't changed
                - 'skipped': Articles that failed to retrieve from API
        """
        # Step 1: Get current state from database
        db_metadata = self.db_client.get_article_metadata()

        # Step 2: Get current state from API
        api_articles, skipped_articles = self.tdx_client.retrieve_all_articles()

        # Step 3: Identify changes (this is the ingestion layer's core logic)
        new_articles, updated_articles, unchanged_articles = self._categorize_articles(
            api_articles, db_metadata
        )

        # Step 4: Process articles (filter, transform, validate)
        processed_new = self.process_articles(new_articles)
        processed_updated = self.process_articles(updated_articles)

        return {
            "new": processed_new,
            "updated": processed_updated,
            "unchanged": unchanged_articles,
            "skipped": skipped_articles,
        }

    def _categorize_articles(
        self, api_articles: List[Dict[str, Any]], db_metadata: Dict[int, datetime]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
        """
        Compare API articles against DB metadata to categorize changes.

        This is pure business logic - no API calls, no DB queries.

        Args:
            api_articles: Raw article data from API
            db_metadata: {article_id: last_modified_date} from database

        Returns:
            Tuple of (new_articles, updated_articles, unchanged_article_ids)
        """
        new_articles = []
        updated_articles = []
        unchanged_ids = []

        for article in api_articles:
            article_id = article.get("ID")
            api_modified_date = article.get("ModifiedDate")

            # Parse the modified date if it's a string
            if isinstance(api_modified_date, str):
                api_modified_date = datetime.fromisoformat(
                    api_modified_date.replace("Z", "+00:00")
                )

            # Skip articles without required metadata
            if article_id is None or api_modified_date is None:
                continue

            if article_id not in db_metadata:
                # Brand new article
                new_articles.append(article)
            elif api_modified_date > db_metadata[article_id]:
                # Article exists but has been updated
                updated_articles.append(article)
            else:
                # Article exists and hasn't changed
                unchanged_ids.append(article_id)

        return new_articles, updated_articles, unchanged_ids

    def process_articles(self, articles: List[Dict[str, Any]]) -> List[TdxArticle]:
        """
        Apply business rules to filter and transform raw articles.

        Business rules:
        - Filter out "Recent Phishing Emails" category
        - Construct public URL from article ID
        - Validate and structure data using Pydantic

        Args:
            articles: Raw article dictionaries from API

        Returns:
            List of validated TdxArticle objects
        """
        processed_articles = []

        for article in articles:
            # Business rule: filter out phishing category
            if article.get("CategoryName") == "Recent Phishing Emails":
                continue

            # Transform raw data into structured format
            article_id = article.get("ID")
            if article_id is None:
                continue  # Skip articles without an ID

            title = article.get("Subject")
            if title is None:
                continue  # Skip articles without a title

            # Business rule: construct public URL (could be moved to config)
            url = f"https://utc.teamdynamix.com/TDClient/2717/Portal/KB/ArticleDet?ID={article_id}"

            content_html = article.get("Body")
            if content_html is None:
                continue  # Skip articles without content

            last_modified_date = article.get("ModifiedDate")
            if last_modified_date is None:
                continue  # Skip articles without a modified date

            # Validate with Pydantic schema
            processed_articles.append(
                TdxArticle(
                    id=article_id,
                    title=title,
                    url=HttpUrl(url),
                    content_html=content_html,
                    last_modified_date=last_modified_date,
                )
            )

        return processed_articles

    def ingest_and_store(self) -> Dict[str, int]:
        """
        Complete ingestion workflow: sync from API and persist to database.

        Returns:
            Statistics: {'new_count': X, 'updated_count': Y, 'skipped_count': Z}
        """
        # Sync to identify changes
        sync_results = self.sync_articles()

        # Persist new articles
        if sync_results["new"]:
            self.db_client.insert_articles(sync_results["new"])

        # Update existing articles
        if sync_results["updated"]:
            self.db_client.update_articles(sync_results["updated"])

        # Return statistics
        return {
            "new_count": len(sync_results["new"]),
            "updated_count": len(sync_results["updated"]),
            "unchanged_count": len(sync_results["unchanged"]),
            "skipped_count": len(sync_results["skipped"]),
        }

    def identify_deleted_articles(self) -> List[int]:
        """
        Find articles that exist in DB but no longer exist in API.

        This is useful for cleanup - articles removed from TDX should
        potentially be marked as deleted/archived in your database.

        Returns:
            List of article IDs that exist in DB but not in API
        """
        # Get what's in the database
        db_article_ids = self.db_client.get_existing_article_ids()

        # Get what's in the API
        api_article_ids = self.tdx_client.list_article_ids()
        api_ids_set = set(api_article_ids)

        # Find the difference
        deleted_ids = db_article_ids - api_ids_set

        return list(deleted_ids)
