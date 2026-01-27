"""
BM25 Search Example

Demonstrates how to use the BM25 retrieval system for keyword-based search.
"""

from core.bm25_search import BM25Retriever
from core.storage_raw import PostgresClient
import json


def main():
    """Run BM25 search examples."""
    print("=" * 80)
    print("BM25 Search Example")
    print("=" * 80)

    # Initialize retriever
    print("\n1. Initializing BM25 retriever...")
    retriever = BM25Retriever(
        k1=1.5,  # Term frequency saturation
        b=0.75,  # Length normalization
        use_cache=True,  # Cache corpus statistics
    )

    # Get retriever stats
    print("\n2. Retriever Statistics:")
    stats = retriever.get_stats()
    print(f"   - Number of chunks: {stats['num_chunks']}")
    print(f"   - Average chunk length: {stats['avg_doc_length']:.1f} tokens")
    print(f"   - Unique terms in corpus: {stats['num_unique_terms']}")
    print(f"   - BM25 parameters: k1={stats['k1']}, b={stats['b']}")

    # Example queries
    queries = [
        "How do I reset my password?",
        "VPN connection issues",
        "Microsoft Teams installation",
        "Email configuration",
        "Two-factor authentication setup",
    ]

    print("\n3. Running searches...")
    for query in queries:
        print(f"\n   Query: '{query}'")
        print("   " + "-" * 70)

        # Search with top 3 results
        results = retriever.search(query=query, top_k=3, min_score=0.5)

        if not results:
            print("   No results found.")
            continue

        for result in results:
            print(f"   [{result.rank}] Score: {result.score:.4f}")
            print(f"       Chunk ID: {result.chunk.chunk_id}")
            print(f"       Source: {result.chunk.source_url}")
            # Show first 100 characters of content
            content_preview = result.chunk.text_content[:100].replace("\n", " ")
            print(f"       Content: {content_preview}...")
            print()

    # Batch search example
    print("\n4. Batch Search Example:")
    print("   Running batch search for multiple queries...")
    batch_results = retriever.batch_search(queries=queries[:3], top_k=2, min_score=1.0)

    for query, results in batch_results.items():
        print(f"\n   '{query}': {len(results)} results")

    # Export results to JSON
    print("\n5. Exporting results to JSON...")
    query = "password reset"
    results = retriever.search(query=query, top_k=5)

    output = {
        "query": query,
        "num_results": len(results),
        "results": [result.to_dict() for result in results],
    }

    with open("data/bm25_search_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"   Saved {len(results)} results to data/bm25_search_results.json")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
