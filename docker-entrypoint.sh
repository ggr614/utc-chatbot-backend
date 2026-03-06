#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "RAG Helpdesk Backend - Docker Entrypoint"
echo "=========================================="

# Function to wait for database
wait_for_db() {
    echo -e "${YELLOW}Waiting for PostgreSQL to be ready...${NC}"

    MAX_RETRIES=30
    RETRY_COUNT=0

    until pg_isready -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -q; do
        RETRY_COUNT=$((RETRY_COUNT + 1))

        if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
            echo -e "${RED}ERROR: Database not ready after $MAX_RETRIES attempts${NC}"
            echo -e "${RED}Please check database connectivity and credentials${NC}"
            exit 1
        fi

        echo "Attempt $RETRY_COUNT/$MAX_RETRIES: Database not ready yet..."
        sleep 2
    done

    echo -e "${GREEN}✓ Database is ready!${NC}"
}

# Function to run migrations
run_migrations() {
    echo -e "${YELLOW}Running Alembic migrations...${NC}"

    # Check current migration status
    echo "Current migration status:"
    alembic current || true

    # Run migrations
    if alembic upgrade head; then
        echo -e "${GREEN}✓ Migrations completed successfully${NC}"
    else
        echo -e "${RED}ERROR: Migration failed${NC}"
        exit 1
    fi
}

# Function to start uvicorn
start_server() {
    echo "=========================================="
    echo "Starting Uvicorn server..."
    echo "Configuration:"
    echo "  - Host: ${API_HOST:-0.0.0.0}"
    echo "  - Port: ${API_PORT:-8000}"
    echo "  - Workers: ${API_WORKERS:-4}"
    echo "  - Log Level: ${API_LOG_LEVEL:-info}"
    echo "=========================================="

    exec uvicorn api.main:app \
        --host "${API_HOST:-0.0.0.0}" \
        --port "${API_PORT:-8000}" \
        --workers "${API_WORKERS:-4}" \
        --log-level "${API_LOG_LEVEL:-info}" \
        --no-access-log \
        --limit-concurrency 100
}

# Main execution flow
main() {
    # Step 1: Wait for database
    wait_for_db

    # Step 2: Run migrations
    run_migrations

    # Step 3: Start server
    start_server
}

# Handle signals for graceful shutdown
trap 'echo -e "${YELLOW}Received shutdown signal${NC}"; exit 0' SIGTERM SIGINT

# Execute main flow
main
