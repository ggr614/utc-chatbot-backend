# RAG Backend Service for Helpdesk Chatbot

Leverages the TDX API to retrieve articles (phishing articles filtered out). Articles are stored in a postgresql database table with appropriate metadata. Articles are then chunked (if needed) using semantic chunking. Chunks are then turned into embeddings using AWS/Azure AI API endpoints. Embeddings are stored in the postgres database using the pgvector extension.

