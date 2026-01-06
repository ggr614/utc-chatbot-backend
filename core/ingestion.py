from utils.api_client import TDXClient
from core.schemas import TdxArticle
from typing import List, Dict, Any, Tuple
from pydantic import HttpUrl
from core.storage_raw import PostgresClient
from datetime import datetime
from utils.logger import get_logger, PerformanceLogger

logger = get_logger(__name__)


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
        logger.info("Initializing ArticleProcessor")
        self.tdx_client = TDXClient()
        self.db_client = PostgresClient()
        logger.debug("ArticleProcessor initialized successfully")

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
        logger.info("Starting article sync operation")

        with PerformanceLogger(logger, "Article sync operation"):
            # Step 1: Get current state from database
            logger.debug("Fetching article metadata from database")
            db_metadata = self.db_client.get_article_metadata()
            logger.info(
                f"Retrieved metadata for {len(db_metadata)} articles from database"
            )

            # Step 2: Get current state from API
            logger.debug("Fetching articles from TDX API")
            api_articles, skipped_articles = self.tdx_client.retrieve_all_articles()
            logger.info(
                f"Retrieved {len(api_articles)} articles from API, {len(skipped_articles)} skipped"
            )

            # Step 3: Identify changes (this is the ingestion layer's core logic)
            logger.debug("Categorizing articles by change status")
            new_articles, updated_articles, unchanged_articles = (
                self._categorize_articles(api_articles, db_metadata)
            )
            logger.info(
                f"Categorization complete: {len(new_articles)} new, "
                f"{len(updated_articles)} updated, {len(unchanged_articles)} unchanged"
            )

            # Step 4: Process articles (filter, transform, validate)
            logger.debug(f"Processing {len(new_articles)} new articles")
            processed_new = self.process_articles(new_articles)
            logger.debug(f"Processing {len(updated_articles)} updated articles")
            processed_updated = self.process_articles(updated_articles)

            logger.info(
                f"Processing complete: {len(processed_new)} new processed, "
                f"{len(processed_updated)} updated processed"
            )

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
        - Safety check for "Recent Phishing Emails" category (already filtered at API level)
        - Construct public URL from article ID
        - Validate and structure data using Pydantic

        Args:
            articles: Raw article dictionaries from API

        Returns:
            List of validated TdxArticle objects
        """
        logger.debug(f"Processing {len(articles)} articles")
        processed_articles = []
        filtered_count = 0
        skipped_count = 0

        for article in articles:
            try:
                # Safety check: filter out phishing category (already filtered at API level)
                # This is a redundant check in case the API filter is bypassed
                if article.get("CategoryName") == "Recent Phishing Emails":
                    filtered_count += 1
                    logger.warning(
                        f"Unexpected phishing article {article.get('ID')} - "
                        f"should have been filtered at API level"
                    )
                    continue

                # Transform raw data into structured format
                article_id = article.get("ID")
                if article_id is None:
                    skipped_count += 1
                    logger.warning("Skipping article without ID")
                    continue

                title = article.get("Subject")
                if title is None:
                    skipped_count += 1
                    logger.warning(f"Skipping article {article_id} without title")
                    continue

                # Business rule: construct public URL (could be moved to config)
                url = f"https://utc.teamdynamix.com/TDClient/2717/Portal/KB/ArticleDet?ID={article_id}"

                content_html = article.get("Body")
                if content_html is None:
                    skipped_count += 1
                    logger.warning(f"Skipping article {article_id} without content")
                    continue

                last_modified_date = article.get("ModifiedDate")
                if last_modified_date is None:
                    skipped_count += 1
                    logger.warning(
                        f"Skipping article {article_id} without modified date"
                    )
                    continue

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
                logger.debug(f"Successfully processed article {article_id}: {title}")

            except Exception as e:
                skipped_count += 1
                article_id = article.get("ID", "unknown")
                logger.error(f"Error processing article {article_id}: {str(e)}")

        logger.info(
            f"Article processing complete: {len(processed_articles)} processed, "
            f"{filtered_count} filtered, {skipped_count} skipped"
        )
        return processed_articles

    def ingest_and_store(self) -> Dict[str, int]:
        """
        Complete ingestion workflow: sync from API and persist to database.

        Returns:
            Statistics: {'new_count': X, 'updated_count': Y, 'skipped_count': Z}
        """
        logger.info("Starting complete ingestion and storage workflow")

        with PerformanceLogger(logger, "Complete ingestion workflow"):
            # Sync to identify changes
            sync_results = self.sync_articles()

            # Persist new articles
            if sync_results["new"]:
                logger.info(
                    f"Inserting {len(sync_results['new'])} new articles into database"
                )
                self.db_client.insert_articles(sync_results["new"])
                logger.info("New articles inserted successfully")
            else:
                logger.info("No new articles to insert")

            # Update existing articles
            if sync_results["updated"]:
                logger.info(
                    f"Updating {len(sync_results['updated'])} existing articles in database"
                )
                self.db_client.update_articles(sync_results["updated"])
                logger.info("Existing articles updated successfully")
            else:
                logger.info("No articles to update")

            stats = {
                "new_count": len(sync_results["new"]),
                "updated_count": len(sync_results["updated"]),
                "unchanged_count": len(sync_results["unchanged"]),
                "skipped_count": len(sync_results["skipped"]),
            }

            logger.info(
                f"Ingestion complete: {stats['new_count']} new, {stats['updated_count']} updated, "
                f"{stats['unchanged_count']} unchanged, {stats['skipped_count']} skipped"
            )

            return stats

    def identify_deleted_articles(self) -> List[int]:
        """
        Find articles that exist in DB but no longer exist in API.

        This is useful for cleanup - articles removed from TDX should
        potentially be marked as deleted/archived in your database.

        Returns:
            List of article IDs that exist in DB but not in API
        """
        logger.info("Identifying deleted articles")

        # Get what's in the database
        logger.debug("Fetching existing article IDs from database")
        db_article_ids = self.db_client.get_existing_article_ids()
        logger.debug(f"Found {len(db_article_ids)} articles in database")

        # Get what's in the API
        logger.debug("Fetching article IDs from TDX API")
        api_article_ids = self.tdx_client.list_article_ids()
        api_ids_set = set(api_article_ids)
        logger.debug(f"Found {len(api_ids_set)} articles in API")

        # Find the difference
        deleted_ids = db_article_ids - api_ids_set
        logger.info(f"Identified {len(deleted_ids)} deleted articles")

        if deleted_ids:
            logger.warning(f"Articles deleted from API: {sorted(list(deleted_ids))}")

        return list(deleted_ids)
