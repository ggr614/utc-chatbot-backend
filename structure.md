rag-backend/
├── .env                    # Environment variables (API keys, DB URLs) - gitignored
├── .gitignore              # Files to exclude from git
├── pyproject.toml          # Project metadata and dependencies (Modern standard)
├── README.md               # Project documentation
├── main.py                 # Entry point to run the pipeline
│
├── core/                   # Main Application Logic
│   ├── __init__.py
│   ├── config.py           # Configuration loader (e.g., using Pydantic Settings)
│   ├── schemas.py          # Data models (Pydantic) for Articles, Chunks, and Embeddings
│   ├── ingestion.py        # Logic to fetch data from the Helpdesk API
│   ├── processing.py       # Logic for cleaning text and splitting (chunking)
│   ├── embedding.py        # Logic to generate vector embeddings (e.g., OpenAI, HF)
│   ├── storage_raw.py      # CRUD operations for raw article storage (SQL/NoSQL/File)
│   ├── storage_vector.py   # CRUD operations for Vector DB (e.g., Qdrant, Pinecone)
│   └── pipeline.py         # Orchestrator that ties ingestion -> processing -> storage
│
├── utils/                  # Helper functions (agnostic to business logic)
│   ├── __init__.py
│   ├── logger.py           # Custom logging setup
│   ├── api_client.py       # Generic API wrapper (retry logic, rate limiting)
│   └── text_tools.py       # Low-level string manipulation (hashing, whitespace)
│
├── tests/                  # Test Suite
│   ├── __init__.py
│   ├── conftest.py         # Pytest fixtures
│   ├── test_ingestion.py
│   ├── test_processing.py
│   └── test_storage.py
