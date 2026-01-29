# Multi-stage Dockerfile for RAG Helpdesk Backend
# Stage 1: Builder - compile dependencies
# Stage 2: Runtime - minimal production image

# ============================================================
# Stage 1: Builder
# ============================================================
FROM python:3.11-slim-bookworm AS builder

# Install build dependencies for compiling Python packages
# - gcc, g++: C/C++ compilers for psycopg, tiktoken, etc.
# - libpq-dev: PostgreSQL client library headers
# - postgresql-client: psql and pg_isready utilities
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv

# Activate virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Note: pywin32-ctypes (line 86) is Windows-only and will be skipped on Linux
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================
# Stage 2: Runtime
# ============================================================
FROM python:3.11-slim-bookworm AS runtime

# Install runtime dependencies only (no compilers)
# - libpq5: PostgreSQL client library (runtime)
# - postgresql-client: psql and pg_isready utilities
# - curl: for health checks
RUN apt-get update && apt-get install -y \
    libpq5 \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
# UID 1000 is standard for first non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Activate virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code with proper ownership
# Copy in order of least to most frequently changed (layer caching optimization)
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser alembic/ ./alembic/
COPY --chown=appuser:appuser core/ ./core/
COPY --chown=appuser:appuser utils/ ./utils/
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser main.py ./

# Copy entrypoint script and fix line endings (Windows CRLF -> Unix LF)
COPY docker-entrypoint.sh ./
RUN sed -i 's/\r$//' docker-entrypoint.sh && \
    chmod +x docker-entrypoint.sh && \
    chown appuser:appuser docker-entrypoint.sh

# Create logs directory with proper ownership
RUN mkdir -p /app/logs && chown -R appuser:appuser /app/logs

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check using readiness endpoint (no authentication required)
# - interval: check every 30 seconds
# - timeout: fail if check takes longer than 10 seconds
# - start-period: wait 60 seconds before starting checks (BM25 corpus loading)
# - retries: mark unhealthy after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/ready || exit 1

# Default command
CMD ["./docker-entrypoint.sh"]
