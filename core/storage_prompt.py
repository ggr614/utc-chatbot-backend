"""
Storage client for tag-based system prompts.

This module provides database operations for managing system prompts
associated with article tags, including priority-based resolution
when articles have multiple tags.
"""

from typing import Dict, Optional, List
from core.storage_base import BaseStorageClient
from utils.logger import get_logger

logger = get_logger(__name__)


class PromptStorageClient(BaseStorageClient):
    """
    Storage client for tag-based system prompts.

    Provides methods for fetching prompts by tag and resolving prompts
    for articles with multiple tags using priority-based resolution.
    """

    def get_prompt_by_tag(self, tag_name: str) -> Optional[str]:
        """
        Get system prompt for a specific tag.

        Args:
            tag_name: Tag name to look up (e.g., "vpn", "__default__")

        Returns:
            System prompt text if found, None otherwise

        Example:
            >>> client = PromptStorageClient()
            >>> prompt = client.get_prompt_by_tag("vpn")
            >>> print(prompt)
            'You are a VPN specialist...'
        """
        logger.debug(f"Fetching prompt for tag: {tag_name}")

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT system_prompt FROM tag_system_prompts WHERE tag_name = %s",
                    (tag_name,),
                )
                row = cur.fetchone()

                if row:
                    logger.debug(
                        f"Found prompt for tag '{tag_name}' ({len(row[0])} chars)"
                    )
                    return row[0]
                else:
                    logger.debug(f"No prompt found for tag '{tag_name}'")
                    return None

    def get_prompts_for_article_ids(self, article_ids: List[str]) -> Dict[str, str]:
        """
        Batch fetch system prompts for multiple articles.

        Uses priority-based resolution: when an article has multiple tags
        with different prompts, returns the prompt with highest priority.
        Falls back to __default__ prompt if no tags match.

        Args:
            article_ids: List of article UUID strings

        Returns:
            Dict mapping article_id (str) -> system_prompt (str)

        Example:
            >>> client = PromptStorageClient()
            >>> prompts = client.get_prompts_for_article_ids(["uuid-1", "uuid-2"])
            >>> print(prompts)
            {'uuid-1': 'You are a VPN specialist...', 'uuid-2': 'You are a helpful...'}
        """
        if not article_ids:
            logger.debug("No article IDs provided, returning empty dict")
            return {}

        logger.debug(f"Batch fetching prompts for {len(article_ids)} articles")

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # For each article, find the matching category prompt,
                # falling back to __default__ if no category matches
                cur.execute(
                    """
                    SELECT
                        a.id AS article_id,
                        COALESCE(
                            (SELECT tsp.system_prompt
                             FROM tag_system_prompts tsp
                             WHERE tsp.tag_name = a.category_name
                             LIMIT 1),
                            (SELECT system_prompt FROM tag_system_prompts
                             WHERE tag_name = '__default__')
                        ) AS system_prompt
                    FROM articles a
                    WHERE a.id = ANY(%s)
                """,
                    (article_ids,),
                )

                results = {str(row[0]): row[1] for row in cur.fetchall()}

                logger.debug(f"Resolved prompts for {len(results)} articles")

                # If some articles didn't get prompts, use default for all missing
                if len(results) < len(article_ids):
                    default_prompt = self.get_default_prompt()
                    for article_id in article_ids:
                        if article_id not in results:
                            logger.debug(
                                f"No prompt found for article {article_id}, using default"
                            )
                            results[article_id] = default_prompt

                return results

    def get_default_prompt(self) -> str:
        """
        Get the default system prompt (__default__ tag).

        Returns:
            Default system prompt text

        Raises:
            ValueError: If no default prompt is configured

        Example:
            >>> client = PromptStorageClient()
            >>> default = client.get_default_prompt()
            >>> print(default)
            'You are a helpful IT helpdesk assistant...'
        """
        logger.debug("Fetching default system prompt")

        prompt = self.get_prompt_by_tag("__default__")

        if not prompt:
            error_msg = "No default prompt found in database. Run migration to seed default prompt."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"Retrieved default prompt ({len(prompt)} chars)")
        return prompt

    def create_prompt(
        self,
        tag_name: str,
        system_prompt: str,
        priority: int = 0,
        description: Optional[str] = None,
    ) -> str:
        """
        Create a new tag-based system prompt.

        Args:
            tag_name: Unique tag name (e.g., "vpn", "password-reset")
            system_prompt: LLM system prompt text
            priority: Priority for conflict resolution (default: 0)
            description: Optional admin notes about this prompt

        Returns:
            UUID of created prompt

        Raises:
            ValueError: If tag_name already exists

        Example:
            >>> client = PromptStorageClient()
            >>> prompt_id = client.create_prompt(
            ...     tag_name="vpn",
            ...     system_prompt="You are a VPN specialist...",
            ...     priority=10
            ... )
        """
        logger.info(f"Creating prompt for tag '{tag_name}' (priority={priority})")

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tag_system_prompts (tag_name, system_prompt, priority, description)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """,
                    (tag_name, system_prompt, priority, description),
                )

                prompt_id = cur.fetchone()[0]
                logger.info(f"Created prompt {prompt_id} for tag '{tag_name}'")
                return str(prompt_id)

    def update_prompt(
        self,
        tag_name: str,
        system_prompt: Optional[str] = None,
        priority: Optional[int] = None,
        description: Optional[str] = None,
    ) -> bool:
        """
        Update an existing prompt.

        Args:
            tag_name: Tag name to update
            system_prompt: New prompt text (optional)
            priority: New priority (optional)
            description: New description (optional)

        Returns:
            True if prompt was updated, False if not found

        Example:
            >>> client = PromptStorageClient()
            >>> updated = client.update_prompt("vpn", priority=15)
            >>> print(updated)
            True
        """
        logger.info(f"Updating prompt for tag '{tag_name}'")

        # Build dynamic UPDATE query based on provided fields
        updates = []
        params = []

        if system_prompt is not None:
            updates.append("system_prompt = %s")
            params.append(system_prompt)

        if priority is not None:
            updates.append("priority = %s")
            params.append(priority)

        if description is not None:
            updates.append("description = %s")
            params.append(description)

        if not updates:
            logger.warning(f"No fields to update for tag '{tag_name}'")
            return False

        # Always update updated_at
        updates.append("updated_at = NOW()")
        params.append(tag_name)

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                query = f"""
                    UPDATE tag_system_prompts
                    SET {", ".join(updates)}
                    WHERE tag_name = %s
                """
                cur.execute(query, params)

                updated = cur.rowcount > 0
                if updated:
                    logger.info(f"Updated prompt for tag '{tag_name}'")
                else:
                    logger.warning(f"No prompt found for tag '{tag_name}'")

                return updated

    def delete_prompt(self, tag_name: str) -> bool:
        """
        Delete a prompt by tag name.

        Args:
            tag_name: Tag name to delete

        Returns:
            True if deleted, False if not found

        Note:
            Cannot delete __default__ prompt (enforced by application logic)

        Example:
            >>> client = PromptStorageClient()
            >>> deleted = client.delete_prompt("old-tag")
            >>> print(deleted)
            True
        """
        if tag_name == "__default__":
            logger.warning("Cannot delete __default__ prompt")
            return False

        logger.info(f"Deleting prompt for tag '{tag_name}'")

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM tag_system_prompts WHERE tag_name = %s", (tag_name,)
                )

                deleted = cur.rowcount > 0
                if deleted:
                    logger.info(f"Deleted prompt for tag '{tag_name}'")
                else:
                    logger.warning(f"No prompt found for tag '{tag_name}'")

                return deleted

    def get_distinct_article_categories(self) -> List[str]:
        """
        Get all distinct category names from the articles table.

        Returns:
            Sorted list of unique category names
        """
        logger.debug("Fetching distinct article categories")

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT category_name
                    FROM articles
                    WHERE category_name IS NOT NULL AND category_name != ''
                    ORDER BY category_name
                """)
                categories = [row[0] for row in cur.fetchall()]
                logger.debug(f"Found {len(categories)} distinct categories")
                return categories

    def bulk_upsert_prompts(
        self,
        system_prompt: str,
        priority: int,
        description: Optional[str],
        tags_to_upsert: List[str],
        tags_to_delete: List[str],
    ) -> Dict[str, List[str]]:
        """
        Atomically upsert prompts for multiple tags and delete removed tags.

        All operations run in a single transaction via one get_connection() call.

        Args:
            system_prompt: The prompt text to assign
            priority: Priority value for conflict resolution
            description: Optional admin notes
            tags_to_upsert: Tags to create or update with this prompt
            tags_to_delete: Tags to remove (unchecked in UI)

        Returns:
            Dict with keys 'created', 'updated', 'deleted'
        """
        created, updated, deleted = [], [], []

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for tag_name in tags_to_upsert:
                    cur.execute(
                        """
                        INSERT INTO tag_system_prompts
                            (tag_name, system_prompt, priority, description)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (tag_name) DO UPDATE SET
                            system_prompt = EXCLUDED.system_prompt,
                            priority = EXCLUDED.priority,
                            description = EXCLUDED.description,
                            updated_at = NOW()
                        RETURNING (xmax = 0) AS is_insert
                        """,
                        (tag_name, system_prompt, priority, description),
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        created.append(tag_name)
                    else:
                        updated.append(tag_name)

                for tag_name in tags_to_delete:
                    if tag_name == "__default__":
                        continue
                    cur.execute(
                        "DELETE FROM tag_system_prompts WHERE tag_name = %s",
                        (tag_name,),
                    )
                    if cur.rowcount > 0:
                        deleted.append(tag_name)

        logger.info(
            f"Bulk upsert complete: {len(created)} created, "
            f"{len(updated)} updated, {len(deleted)} deleted"
        )
        return {"created": created, "updated": updated, "deleted": deleted}

    def list_all_prompts(self) -> List[Dict]:
        """
        List all prompts ordered by priority (descending).

        Returns:
            List of dicts with keys: tag_name, system_prompt, priority, description

        Example:
            >>> client = PromptStorageClient()
            >>> prompts = client.list_all_prompts()
            >>> for p in prompts:
            ...     print(f"{p['tag_name']}: priority={p['priority']}")
            __default__: priority=1000
            vpn: priority=10
        """
        logger.debug("Listing all prompts")

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT tag_name, system_prompt, priority, description, created_at, updated_at
                    FROM tag_system_prompts
                    ORDER BY priority DESC, tag_name ASC
                """)

                prompts = [
                    {
                        "tag_name": row[0],
                        "system_prompt": row[1],
                        "priority": row[2],
                        "description": row[3],
                        "created_at": row[4],
                        "updated_at": row[5],
                    }
                    for row in cur.fetchall()
                ]

                logger.debug(f"Retrieved {len(prompts)} prompts")
                return prompts
