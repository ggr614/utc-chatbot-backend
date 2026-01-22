"""
Vector Search Example

Demonstrates how to use the vector-based semantic retrieval system
using OpenAI embeddings and pgvector.
"""

from core.vector_search import VectorRetriever
import json


def main():
    """Run vector search examples."""
    print("=" * 80)
    print("Vector Semantic Search Example")
    print("=" * 80)

    # Initialize retriever
    print("\n1. Initializing Vector retriever...")
    print("   (This will use OpenAI embeddings and pgvector storage)")
    retriever = VectorRetriever()

    # Get retriever stats
    print("\n2. Retriever Statistics:")
    stats = retriever.get_stats()
    print(f"   - Number of embeddings: {stats['num_embeddings']}")
    print(f"   - Embedding dimension: {stats['embedding_dimension']}")
    print(f"   - Model: {stats['model']}")
    print(f"   - Provider: {stats['provider']}")

    # Example queries - testing semantic understanding
    queries = [
        "How do I reset my password?",
        "I can't log into my account",  # Similar meaning to password reset
        "Setting up VPN connection",
        "Configuring email on mobile device",
        "Two-factor authentication not working",
    ]

    print("\n3. Running semantic searches...")
    print("   (Note: Vector search finds semantically similar content,")
    print("    even when exact keywords don't match)")

    for query in queries:
        print(f"\n   Query: '{query}'")
        print("   " + "-" * 70)

        # Search with top 3 results and minimum similarity threshold
        results = retriever.search(
            query=query,
            top_k=3,
            min_similarity=0.6  # Only show results with >60% similarity
        )

        if not results:
            print("   No results found above similarity threshold.")
            continue

        for result in results:
            print(f"   [{result.rank}] Similarity: {result.similarity:.4f}")
            print(f"       Chunk ID: {result.chunk.chunk_id}")
            print(f"       Source: {result.chunk.source_url}")
            # Show first 100 characters of content
            content_preview = result.chunk.text_content[:100].replace("\n", " ")
            print(f"       Content: {content_preview}...")
            print()

    # Demonstrate semantic similarity
    print("\n4. Semantic Similarity Example:")
    print("   Searching for 'authentication problems'...")
    results = retriever.search(
        query="authentication problems",
        top_k=5,
        min_similarity=0.5
    )

    print(f"   Found {len(results)} semantically related results")
    print("   (These may not contain the exact words 'authentication' or 'problems')")
    for result in results[:3]:
        print(f"   - Similarity: {result.similarity:.4f}")
        print(f"     Preview: {result.chunk.text_content[:80]}...")
        print()

    # Batch search example
    print("\n5. Batch Search Example:")
    print("   Running batch search for multiple queries...")
    batch_queries = [
        "password recovery",
        "network connectivity",
        "software installation"
    ]

    batch_results = retriever.batch_search(
        queries=batch_queries,
        top_k=2,
        min_similarity=0.6
    )

    for query, results in batch_results.items():
        print(f"\n   '{query}': {len(results)} results")
        if results:
            print(f"      Top match similarity: {results[0].similarity:.4f}")

    # Find similar chunks example
    print("\n6. Find Similar Chunks Example:")
    print("   Finding articles related to a specific chunk...")

    # Get the first result from a search
    sample_results = retriever.search(query="password reset", top_k=1)

    if sample_results:
        sample_chunk_id = str(sample_results[0].chunk.chunk_id)
        print(f"   Base chunk: {sample_chunk_id}")
        print(f"   Content preview: {sample_results[0].chunk.text_content[:80]}...")

        # Find similar chunks
        similar = retriever.find_similar_to_chunk(
            chunk_id=sample_chunk_id,
            top_k=5,
            min_similarity=0.5
        )

        print(f"\n   Found {len(similar)} similar chunks:")
        for result in similar[:3]:
            print(f"   [{result.rank}] Similarity: {result.similarity:.4f}")
            print(f"       Preview: {result.chunk.text_content[:80]}...")
            print()

    # Export results to JSON
    print("\n7. Exporting results to JSON...")
    query = "email configuration"
    results = retriever.search(query=query, top_k=5, min_similarity=0.5)

    output = {
        "query": query,
        "search_type": "vector_semantic",
        "num_results": len(results),
        "model": stats["model"],
        "results": [result.to_dict() for result in results],
    }

    with open("data/vector_search_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"   Saved {len(results)} results to data/vector_search_results.json")

    # Comparison with keyword search
    print("\n8. Vector vs Keyword Search:")
    print("   Vector search excels at:")
    print("   - Understanding synonyms and paraphrases")
    print("   - Finding conceptually similar content")
    print("   - Handling queries with different wording")
    print("   ")
    print("   Example: Query 'login issues' will match chunks about")
    print("   'authentication problems', 'sign-in errors', 'access denied', etc.")

    # Clean up
    retriever.close()

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
