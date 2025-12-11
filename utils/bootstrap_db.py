"""
Database Bootstrap Script

Sets up the PostgreSQL database with all required tables and extensions.

Usage:
    python -m utils.bootstrap_db                # Normal setup
    python -m utils.bootstrap_db --dry-run      # Check current status without changes
    python -m utils.bootstrap_db --full-reset   # Drop all tables and rebuild
"""

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any

import psycopg
from psycopg import Connection, sql

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import get_settings


class DatabaseBootstrap:
    """
    Handles database initialization and schema management.
    """

    def __init__(self, dry_run: bool = False):
        """
        Initialize the bootstrap manager.

        Args:
            dry_run: If True, only check status without making changes
        """
        settings = get_settings()
        self.db_host = settings.DB_HOST
        self.db_user = settings.DB_USER
        self.db_password = settings.DB_PASSWORD.get_secret_value()
        self.db_name = settings.DB_NAME
        self.db_port = 5432
        self.dry_run = dry_run

        self._connection_params = {
            "host": self.db_host,
            "user": self.db_user,
            "password": self.db_password,
            "dbname": self.db_name,
            "port": self.db_port,
        }

    @contextmanager
    def get_connection(self):
        """Get a database connection with automatic cleanup."""
        conn = None
        try:
            conn = psycopg.connect(**self._connection_params)
            yield conn
            if not self.dry_run:
                conn.commit()
        except psycopg.Error as e:
            if conn:
                conn.rollback()
            raise ConnectionError(f"Database error: {e}") from e
        finally:
            if conn and not conn.closed:
                conn.close()

    def check_table_exists(self, conn: Connection, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            conn: Database connection
            table_name: Name of the table to check

        Returns:
            True if table exists, False otherwise
        """
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = %s
                );
                """,
                (table_name,),
            )
            result = cur.fetchone()
            return result[0] if result else False

    def check_extension_exists(self, conn: Connection, extension_name: str) -> bool:
        """
        Check if an extension is installed.

        Args:
            conn: Database connection
            extension_name: Name of the extension to check

        Returns:
            True if extension exists, False otherwise
        """
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM pg_extension
                    WHERE extname = %s
                );
                """,
                (extension_name,),
            )
            result = cur.fetchone()
            return result[0] if result else False

    def get_table_info(self, conn: Connection, table_name: str) -> Dict[str, Any]:
        """
        Get information about a table's structure.

        Args:
            conn: Database connection
            table_name: Name of the table

        Returns:
            Dictionary with table information
        """
        with conn.cursor() as cur:
            # Get column information
            cur.execute(
                """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position;
                """,
                (table_name,),
            )
            columns = cur.fetchall()

            # Get row count
            try:
                cur.execute(
                    sql.SQL("SELECT COUNT(*) FROM {}").format(
                        sql.Identifier(table_name)
                    )
                )
                row_count = cur.fetchone()[0]  # type: ignore
            except psycopg.Error:
                row_count = 0

            return {"columns": columns, "row_count": row_count}

    def create_extensions(self, conn: Connection) -> None:
        """
        Create required PostgreSQL extensions.

        Args:
            conn: Database connection
        """
        extensions = ["vector"]

        for ext in extensions:
            exists = self.check_extension_exists(conn, ext)

            if self.dry_run:
                if exists:
                    print(f"  [OK] Extension '{ext}' already exists")
                else:
                    print(f"  [->] Would create extension '{ext}'")
            else:
                if not exists:
                    with conn.cursor() as cur:
                        cur.execute(
                            sql.SQL("CREATE EXTENSION IF NOT EXISTS {}").format(
                                sql.Identifier(ext)
                            )
                        )
                    print(f"  [OK] Created extension '{ext}'")
                else:
                    print(f"  [OK] Extension '{ext}' already exists")

    def create_articles_table(self, conn: Connection) -> None:
        """
        Create the articles table for raw article storage.

        Args:
            conn: Database connection
        """
        table_name = "articles"
        exists = self.check_table_exists(conn, table_name)

        if self.dry_run:
            if exists:
                info = self.get_table_info(conn, table_name)
                print(
                    f"  [OK] Table '{table_name}' already exists ({info['row_count']} rows)"
                )
                print(f"    Columns: {len(info['columns'])}")
            else:
                print(f"  -> Would create table '{table_name}'")
            return

        if exists:
            print(f"  [OK] Table '{table_name}' already exists")
            return

        create_table_sql = """
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            content_html TEXT NOT NULL,
            last_modified_date TIMESTAMP WITH TIME ZONE NOT NULL,
            raw_ingestion_date TIMESTAMP WITH TIME ZONE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """

        create_index_sql = """
        CREATE INDEX idx_articles_last_modified ON articles(last_modified_date);
        CREATE INDEX idx_articles_ingestion_date ON articles(raw_ingestion_date);
        """

        with conn.cursor() as cur:
            cur.execute(create_table_sql)
            cur.execute(create_index_sql)

        print(f"  [OK] Created table '{table_name}' with indexes")

    def create_chunks_table(self, conn: Connection) -> None:
        """
        Create the chunks table for cleaned article storage.

        Args:
            conn: Database connection
        """
        table_name = "article_chunks"
        exists = self.check_table_exists(conn, table_name)

        if self.dry_run:
            if exists:
                info = self.get_table_info(conn, table_name)
                print(
                    f"  [OK] Table '{table_name}' already exists ({info['row_count']} rows)"
                )
                print(f"    Columns: {len(info['columns'])}")
            else:
                print(f"  -> Would create table '{table_name}'")
            return

        if exists:
            print(f"  [OK] Table '{table_name}' already exists")
            return

        create_table_sql = """
        CREATE TABLE article_chunks (
            id INTEGER PRIMARY KEY,
            parent_article_id INTEGER NOT NULL,
            chunk_sequence INTEGER NOT NULL,
            text_content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            url TEXT NOT NULL,
            last_modified_date TIMESTAMP WITH TIME ZONE NOT NULL
        );
        """

        create_index_sql = """
        CREATE INDEX idx_chunks_parent_article ON article_chunks(parent_article_id);
        CREATE INDEX idx_chunks_chunk_sequence ON article_chunks(parent_article_id, chunk_sequence);
        """

        with conn.cursor() as cur:
            cur.execute(create_table_sql)
            cur.execute(create_index_sql)

        print(f"  [OK] Created table '{table_name}' with indexes")

    def create_embeddings_table_openai(self, conn: Connection) -> None:
        """
        Create the embeddings table for OpenAI text-embedding-3-large vectors (3072 dimensions).

        Args:
            conn: Database connection
        """
        table_name = "embeddings_openai"
        exists = self.check_table_exists(conn, table_name)

        if self.dry_run:
            if exists:
                info = self.get_table_info(conn, table_name)
                print(
                    f"  [OK] Table '{table_name}' already exists ({info['row_count']} rows)"
                )
                print(f"    Columns: {len(info['columns'])}")
            else:
                print(f"  -> Would create table '{table_name}'")
            return

        if exists:
            print(f"  [OK] Table '{table_name}' already exists")
            return

        # Check if vector extension exists before creating the table
        if not self.check_extension_exists(conn, "vector"):
            print(
                f"  [!] Warning: vector extension not installed. Skipping '{table_name}' table creation."
            )
            return

        create_table_sql = """
        CREATE TABLE embeddings_openai (
            chunk_id TEXT PRIMARY KEY,
            parent_article_id INTEGER NOT NULL,
            chunk_sequence INTEGER NOT NULL,
            text_content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            source_url TEXT NOT NULL,
            embedding vector(3072),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (parent_article_id) REFERENCES articles(id) ON DELETE CASCADE
        );
        """

        create_index_sql = """
        CREATE INDEX idx_embeddings_openai_parent_article ON embeddings_openai(parent_article_id);
        CREATE INDEX idx_embeddings_openai_chunk_sequence ON embeddings_openai(parent_article_id, chunk_sequence);
        """

        with conn.cursor() as cur:
            cur.execute(create_table_sql)
            cur.execute(create_index_sql)

        print(f"  [OK] Created table '{table_name}' with indexes and foreign key (3072 dimensions)")

    def create_embeddings_table_cohere(self, conn: Connection) -> None:
        """
        Create the embeddings table for AWS Cohere Embed v4 vectors (1536 dimensions).

        Args:
            conn: Database connection
        """
        table_name = "embeddings_cohere"
        exists = self.check_table_exists(conn, table_name)

        if self.dry_run:
            if exists:
                info = self.get_table_info(conn, table_name)
                print(
                    f"  [OK] Table '{table_name}' already exists ({info['row_count']} rows)"
                )
                print(f"    Columns: {len(info['columns'])}")
            else:
                print(f"  -> Would create table '{table_name}'")
            return

        if exists:
            print(f"  [OK] Table '{table_name}' already exists")
            return

        # Check if vector extension exists before creating the table
        if not self.check_extension_exists(conn, "vector"):
            print(
                f"  [!] Warning: vector extension not installed. Skipping '{table_name}' table creation."
            )
            return

        create_table_sql = """
        CREATE TABLE embeddings_cohere (
            chunk_id TEXT PRIMARY KEY,
            parent_article_id INTEGER NOT NULL,
            chunk_sequence INTEGER NOT NULL,
            text_content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            source_url TEXT NOT NULL,
            embedding vector(1536),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (parent_article_id) REFERENCES articles(id) ON DELETE CASCADE
        );
        """

        create_index_sql = """
        CREATE INDEX idx_embeddings_cohere_parent_article ON embeddings_cohere(parent_article_id);
        CREATE INDEX idx_embeddings_cohere_chunk_sequence ON embeddings_cohere(parent_article_id, chunk_sequence);
        CREATE INDEX idx_embeddings_cohere_vector ON embeddings_cohere USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        """

        with conn.cursor() as cur:
            cur.execute(create_table_sql)
            cur.execute(create_index_sql)

        print(f"  [OK] Created table '{table_name}' with indexes and foreign key (1536 dimensions)")

    def drop_all_tables(self, conn: Connection) -> None:
        """
        Drop all tables in the database.

        Args:
            conn: Database connection
        """
        tables = ["embeddings_openai", "embeddings_cohere", "articles", "article_chunks"]

        if self.dry_run:
            print("  -> Would drop the following tables:")
            for table in tables:
                if self.check_table_exists(conn, table):
                    info = self.get_table_info(conn, table)
                    print(f"    - {table} ({info['row_count']} rows)")
            return

        with conn.cursor() as cur:
            for table in tables:
                if self.check_table_exists(conn, table):
                    cur.execute(
                        sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                            sql.Identifier(table)
                        )
                    )
                    print(f"  [OK] Dropped table '{table}'")

    def setup_database(self, full_reset: bool = False) -> None:
        """
        Set up the database with all required tables and extensions.

        Args:
            full_reset: If True, drop all tables before recreating them
        """
        try:
            with self.get_connection() as conn:
                if self.dry_run:
                    print("\n=== DRY RUN MODE - No changes will be made ===\n")

                print(f"Connected to database: {self.db_name}@{self.db_host}")
                print()

                if full_reset:
                    print("[!] FULL RESET MODE - Dropping all tables...")
                    self.drop_all_tables(conn)
                    print()

                print("Setting up extensions...")
                self.create_extensions(conn)
                print()

                print("Setting up tables...")
                self.create_articles_table(conn)
                self.create_embeddings_table_openai(conn)
                self.create_embeddings_table_cohere(conn)
                self.create_chunks_table(conn)
                print()

                if self.dry_run:
                    print("=== DRY RUN COMPLETE - No changes were made ===")
                else:
                    print("[OK] Database setup complete!")

        except ConnectionError as e:
            print(f"[X] Error connecting to database: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[X] Unexpected error: {e}")
            sys.exit(1)

    def check_status(self) -> None:
        """
        Check and display the current database status.
        """
        try:
            with self.get_connection() as conn:
                print(f"\n=== Database Status: {self.db_name}@{self.db_host} ===\n")

                # Check extensions
                print("Extensions:")
                extensions = ["vector"]
                for ext in extensions:
                    exists = self.check_extension_exists(conn, ext)
                    status = "[OK] Installed" if exists else "[X] Not installed"
                    print(f"  {ext}: {status}")
                print()

                # Check tables
                print("Tables:")
                tables = ["articles", "embeddings_openai", "embeddings_cohere", "article_chunks"]
                for table in tables:
                    exists = self.check_table_exists(conn, table)
                    if exists:
                        info = self.get_table_info(conn, table)
                        print(
                            f"  {table}: [OK] Exists ({info['row_count']} rows, {len(info['columns'])} columns)"
                        )
                    else:
                        print(f"  {table}: [X] Does not exist")
                print()

        except ConnectionError as e:
            print(f"[X] Error connecting to database: {e}")
            sys.exit(1)


def main():
    """Main entry point for the bootstrap script."""
    parser = argparse.ArgumentParser(
        description="Bootstrap the PostgreSQL database for the RAG backend service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m utils.bootstrap_db                    # Normal setup
  python -m utils.bootstrap_db --dry-run          # Check status and show what would be done
  python -m utils.bootstrap_db --full-reset       # Drop all tables and rebuild
  python -m utils.bootstrap_db --status           # Check current database status
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check current database status and show what would be done without making changes",
    )

    parser.add_argument(
        "--full-reset",
        action="store_true",
        help="Drop all existing tables and rebuild the database (WARNING: This will delete all data!)",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Check and display current database status",
    )

    args = parser.parse_args()

    # Confirm full reset
    if args.full_reset and not args.dry_run:
        print("[!] WARNING: Full reset will delete all data in the database!")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    bootstrap = DatabaseBootstrap(dry_run=args.dry_run)

    if args.status:
        bootstrap.check_status()
    else:
        bootstrap.setup_database(full_reset=args.full_reset)


if __name__ == "__main__":
    main()
