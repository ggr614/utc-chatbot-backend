#!/bin/bash
set -e

# init-db.sh
# Creates the LiteLLM database and user on the shared PostgreSQL instance.
# Runs once during first container initialization (empty data volume).
#
# The RAG database ($POSTGRES_DB) is created automatically by the postgres entrypoint.
# The pgvector extension for the RAG database is created by Alembic migrations.

echo "Creating LiteLLM database and user..."

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create LiteLLM user if it does not exist
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '${LITELLM_DB_USER}') THEN
            CREATE USER "${LITELLM_DB_USER}" WITH PASSWORD '${LITELLM_DB_PASSWORD}';
        END IF;
    END
    \$\$;
EOSQL

# Create LiteLLM database if it does not exist
if ! psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" -lqt | cut -d \| -f 1 | grep -qw "${LITELLM_DB_NAME}"; then
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        CREATE DATABASE "${LITELLM_DB_NAME}" OWNER "${LITELLM_DB_USER}";
EOSQL
fi

# Grant privileges
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    GRANT ALL PRIVILEGES ON DATABASE "${LITELLM_DB_NAME}" TO "${LITELLM_DB_USER}";
EOSQL

echo "LiteLLM database '${LITELLM_DB_NAME}' created successfully."
