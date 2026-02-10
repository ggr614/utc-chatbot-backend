"""
Load / Performance Testing for RAG Search API

Supports two test modes:
- Burst: Fires N concurrent requests simultaneously (peak load)
- Sustained: Simulates N users making requests over a duration (steady state)

Uses questions from the eval dataset as realistic query payloads.

Usage:
    python -m qa.load_test burst --concurrency 30 --endpoint hybrid
    python -m qa.load_test sustained --users 30 --duration 300 --endpoint hybrid
    python -m qa.load_test all --concurrency 30 --users 30 --duration 300
"""

import argparse
import asyncio
import os
import random
import sys
import time
from dataclasses import dataclass, field

import httpx

ENDPOINT_PATHS = {
    "bm25": "/api/v1/search/bm25",
    "vector": "/api/v1/search/vector",
    "hybrid": "/api/v1/search/hybrid",
    "hyde": "/api/v1/search/hyde",
}


@dataclass
class RequestResult:
    """Result of a single HTTP request."""

    status_code: int
    client_latency_ms: float
    server_latency_ms: float | None
    error: str | None
    endpoint: str
    query: str
    timestamp: float


@dataclass
class TestMetrics:
    """Aggregated metrics for a completed test run."""

    test_type: str
    endpoint: str
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration_s: float
    throughput_rps: float
    error_rate_pct: float
    client_min: float
    client_max: float
    client_mean: float
    client_p50: float
    client_p95: float
    client_p99: float
    server_min: float | None
    server_max: float | None
    server_mean: float | None
    server_p50: float | None
    server_p95: float | None
    server_p99: float | None
    status_code_counts: dict[int, int] = field(default_factory=dict)


@dataclass
class LoadTestConfig:
    """Configuration for a load test run."""

    base_url: str
    api_key: str
    endpoint: str
    test_type: str
    concurrency: int
    users: int
    duration_s: int
    top_k: int
    email: str
    think_time_min: float
    think_time_max: float
    timeout_s: float


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute percentile using linear interpolation."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_values):
        return sorted_values[-1]
    d = k - f
    return sorted_values[f] + d * (sorted_values[c] - sorted_values[f])


def compute_metrics(
    results: list[RequestResult],
    test_type: str,
    endpoint: str,
    concurrency: int,
    total_duration: float,
) -> TestMetrics:
    """Compute aggregated metrics from a list of request results."""
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    client_latencies = sorted(r.client_latency_ms for r in successful)
    server_latencies = sorted(
        r.server_latency_ms for r in successful if r.server_latency_ms is not None
    )

    status_counts: dict[int, int] = {}
    for r in results:
        status_counts[r.status_code] = status_counts.get(r.status_code, 0) + 1

    total = len(results)

    return TestMetrics(
        test_type=test_type,
        endpoint=endpoint,
        concurrency=concurrency,
        total_requests=total,
        successful_requests=len(successful),
        failed_requests=len(failed),
        total_duration_s=total_duration,
        throughput_rps=total / total_duration if total_duration > 0 else 0,
        error_rate_pct=(len(failed) / total * 100) if total > 0 else 0,
        client_min=client_latencies[0] if client_latencies else 0,
        client_max=client_latencies[-1] if client_latencies else 0,
        client_mean=(
            sum(client_latencies) / len(client_latencies) if client_latencies else 0
        ),
        client_p50=_percentile(client_latencies, 50),
        client_p95=_percentile(client_latencies, 95),
        client_p99=_percentile(client_latencies, 99),
        server_min=server_latencies[0] if server_latencies else None,
        server_max=server_latencies[-1] if server_latencies else None,
        server_mean=(
            sum(server_latencies) / len(server_latencies) if server_latencies else None
        ),
        server_p50=_percentile(server_latencies, 50) if server_latencies else None,
        server_p95=_percentile(server_latencies, 95) if server_latencies else None,
        server_p99=_percentile(server_latencies, 99) if server_latencies else None,
        status_code_counts=status_counts,
    )


def format_report(metrics: TestMetrics) -> str:
    """Format test metrics into a readable report."""
    lines = [
        "",
        "=" * 50,
        "  Load Test Results",
        "=" * 50,
        f"  Test:        {metrics.test_type}",
        f"  Endpoint:    {metrics.endpoint}",
        f"  Concurrency: {metrics.concurrency}",
        f"  Duration:    {metrics.total_duration_s:.1f}s",
        f"  Requests:    {metrics.total_requests} "
        f"({metrics.successful_requests} ok, {metrics.failed_requests} failed)",
        "",
        "  Latency (client-side):",
        f"    Min:   {metrics.client_min:>8.0f}ms",
        f"    Mean:  {metrics.client_mean:>8.0f}ms",
        f"    p50:   {metrics.client_p50:>8.0f}ms",
        f"    p95:   {metrics.client_p95:>8.0f}ms",
        f"    p99:   {metrics.client_p99:>8.0f}ms",
        f"    Max:   {metrics.client_max:>8.0f}ms",
    ]

    if metrics.server_mean is not None:
        lines.extend(
            [
                "",
                "  Latency (server-side):",
                f"    Min:   {metrics.server_min:>8.0f}ms",
                f"    Mean:  {metrics.server_mean:>8.0f}ms",
                f"    p50:   {metrics.server_p50:>8.0f}ms",
                f"    p95:   {metrics.server_p95:>8.0f}ms",
                f"    p99:   {metrics.server_p99:>8.0f}ms",
                f"    Max:   {metrics.server_max:>8.0f}ms",
            ]
        )

    lines.extend(
        [
            "",
            f"  Throughput:  {metrics.throughput_rps:.1f} req/s",
            f"  Error Rate:  {metrics.error_rate_pct:.1f}%",
        ]
    )

    if metrics.status_code_counts:
        lines.append("")
        lines.append("  Status Codes:")
        for code, count in sorted(metrics.status_code_counts.items()):
            label = "timeout/conn" if code == 0 else str(code)
            lines.append(f"    {label}: {count}")

    lines.append("=" * 50)
    return "\n".join(lines)


def print_summary_table(all_metrics: list[TestMetrics]) -> None:
    """Print a comparison table for multiple test runs."""
    print("\n" + "=" * 72)
    print("  COMPARISON SUMMARY")
    print("=" * 72)
    header = (
        f"  {'Test':<10} {'Endpoint':<10} {'Reqs':>6} {'Errs':>5} "
        f"{'p50':>8} {'p95':>8} {'p99':>8} {'RPS':>6}"
    )
    print(header)
    print("  " + "-" * 68)
    for m in all_metrics:
        print(
            f"  {m.test_type:<10} {m.endpoint:<10} "
            f"{m.total_requests:>6} {m.failed_requests:>5} "
            f"{m.client_p50:>7.0f}ms {m.client_p95:>7.0f}ms "
            f"{m.client_p99:>7.0f}ms {m.throughput_rps:>5.1f}"
        )
    print("=" * 72)


# ---------------------------------------------------------------------------
# HTTP request logic
# ---------------------------------------------------------------------------


async def send_request(
    client: httpx.AsyncClient,
    config: LoadTestConfig,
    endpoint_path: str,
    query: str,
) -> RequestResult:
    """Send a single search request and measure latency."""
    payload = {
        "query": query,
        "top_k": config.top_k,
        "email": config.email,
    }
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": config.api_key,
    }

    timestamp = time.time()
    start = time.perf_counter()

    try:
        response = await client.post(endpoint_path, json=payload, headers=headers)
        client_latency_ms = (time.perf_counter() - start) * 1000

        server_latency_ms = None
        if response.status_code == 200:
            try:
                data = response.json()
                server_latency_ms = data.get("latency_ms")
            except Exception:
                pass

        error = None if response.status_code == 200 else f"HTTP {response.status_code}"

        return RequestResult(
            status_code=response.status_code,
            client_latency_ms=client_latency_ms,
            server_latency_ms=server_latency_ms,
            error=error,
            endpoint=endpoint_path,
            query=query,
            timestamp=timestamp,
        )

    except httpx.TimeoutException:
        return RequestResult(
            status_code=0,
            client_latency_ms=(time.perf_counter() - start) * 1000,
            server_latency_ms=None,
            error="Timeout",
            endpoint=endpoint_path,
            query=query,
            timestamp=timestamp,
        )
    except Exception as e:
        return RequestResult(
            status_code=0,
            client_latency_ms=(time.perf_counter() - start) * 1000,
            server_latency_ms=None,
            error=str(e),
            endpoint=endpoint_path,
            query=query,
            timestamp=timestamp,
        )


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------


async def run_burst_test(
    config: LoadTestConfig,
    questions: list[str],
    endpoint_path: str,
) -> TestMetrics:
    """Fire N concurrent requests simultaneously."""
    selected = [questions[i % len(questions)] for i in range(config.concurrency)]

    limits = httpx.Limits(
        max_connections=config.concurrency + 5,
        max_keepalive_connections=config.concurrency,
    )
    async with httpx.AsyncClient(
        base_url=config.base_url,
        limits=limits,
        timeout=httpx.Timeout(config.timeout_s),
    ) as client:
        start = time.time()
        tasks = [send_request(client, config, endpoint_path, q) for q in selected]
        results = await asyncio.gather(*tasks)
        duration = time.time() - start

    return compute_metrics(
        list(results), "burst", endpoint_path, config.concurrency, duration
    )


async def _simulate_user(
    client: httpx.AsyncClient,
    config: LoadTestConfig,
    endpoint_path: str,
    questions: list[str],
    end_time: float,
    rng: random.Random,
) -> list[RequestResult]:
    """Simulate a single user making requests with think-time pauses."""
    results: list[RequestResult] = []
    while time.time() < end_time:
        query = rng.choice(questions)
        result = await send_request(client, config, endpoint_path, query)
        results.append(result)

        think_time = rng.uniform(config.think_time_min, config.think_time_max)
        remaining = end_time - time.time()
        if remaining <= 0:
            break
        await asyncio.sleep(min(think_time, remaining))

    return results


async def _progress_reporter(
    user_results: list[list[RequestResult]],
    end_time: float,
    start_time: float,
    interval: float = 30.0,
) -> None:
    """Print periodic progress during sustained test."""
    while time.time() < end_time:
        await asyncio.sleep(interval)
        total = sum(len(r) for r in user_results)
        elapsed = time.time() - start_time
        print(f"  [progress] {elapsed:.0f}s elapsed, {total} requests completed")


async def run_sustained_test(
    config: LoadTestConfig,
    questions: list[str],
    endpoint_path: str,
) -> TestMetrics:
    """Simulate N users making requests over a duration."""
    limits = httpx.Limits(
        max_connections=config.users + 10,
        max_keepalive_connections=config.users,
    )
    async with httpx.AsyncClient(
        base_url=config.base_url,
        limits=limits,
        timeout=httpx.Timeout(config.timeout_s),
    ) as client:
        start_time = time.time()
        end_time = start_time + config.duration_s

        # Each user gets a mutable list that the progress reporter can read
        user_results: list[list[RequestResult]] = [[] for _ in range(config.users)]

        async def user_wrapper(idx: int) -> list[RequestResult]:
            results = await _simulate_user(
                client,
                config,
                endpoint_path,
                questions,
                end_time,
                random.Random(42 + idx),
            )
            user_results[idx] = results
            return results

        user_tasks = [user_wrapper(i) for i in range(config.users)]
        reporter_task = asyncio.create_task(
            _progress_reporter(user_results, end_time, start_time)
        )

        all_user_results = await asyncio.gather(*user_tasks)
        reporter_task.cancel()
        try:
            await reporter_task
        except asyncio.CancelledError:
            pass

        duration = time.time() - start_time

    all_results = [r for user_list in all_user_results for r in user_list]
    return compute_metrics(
        all_results, "sustained", endpoint_path, config.users, duration
    )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_questions(dataset_path: str, sample_size: int) -> list[str]:
    """Load questions from the eval dataset."""
    from qa.eval_dataset import load_eval_dataset

    dataset = load_eval_dataset(
        filepath=dataset_path,
        filter_sufficient_context=True,
        sample_size=sample_size,
        random_seed=42,
    )
    questions = [q.question for q in dataset.questions]

    if not questions:
        print(f"ERROR: No questions loaded from {dataset_path}")
        sys.exit(1)

    print(f"Loaded {len(questions)} questions from eval dataset")
    return questions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m qa.load_test",
        description="Load/performance testing for RAG search API endpoints",
    )

    subparsers = parser.add_subparsers(dest="test_type", required=True)

    def add_common_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--endpoint",
            choices=["bm25", "vector", "hybrid", "hyde", "all"],
            default="hybrid",
            help="Endpoint to test (default: hybrid)",
        )
        sp.add_argument(
            "--base-url",
            default=None,
            help="API base URL (default: from env or http://localhost:8000)",
        )
        sp.add_argument(
            "--api-key",
            default=None,
            help="API key (default: from API_API_KEY env var)",
        )
        sp.add_argument(
            "--top-k", type=int, default=10, help="top_k parameter (default: 10)"
        )
        sp.add_argument(
            "--timeout",
            type=float,
            default=30.0,
            help="Request timeout in seconds (default: 30)",
        )
        sp.add_argument(
            "--dataset", default="data/qa_pairs.jsonl", help="Path to QA pairs JSONL"
        )
        sp.add_argument(
            "--sample-size",
            type=int,
            default=200,
            help="Number of questions to load (default: 200)",
        )

    burst_parser = subparsers.add_parser(
        "burst", help="Fire N concurrent requests simultaneously"
    )
    burst_parser.add_argument(
        "--concurrency",
        type=int,
        default=30,
        help="Number of concurrent requests (default: 30)",
    )
    add_common_args(burst_parser)

    sustained_parser = subparsers.add_parser(
        "sustained", help="Simulate N users over a duration"
    )
    sustained_parser.add_argument(
        "--users", type=int, default=30, help="Number of simulated users (default: 30)"
    )
    sustained_parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Test duration in seconds (default: 300)",
    )
    sustained_parser.add_argument(
        "--think-time-min",
        type=float,
        default=2.0,
        help="Min think time in seconds (default: 2.0)",
    )
    sustained_parser.add_argument(
        "--think-time-max",
        type=float,
        default=8.0,
        help="Max think time in seconds (default: 8.0)",
    )
    add_common_args(sustained_parser)

    all_parser = subparsers.add_parser("all", help="Run both burst and sustained tests")
    all_parser.add_argument(
        "--concurrency", type=int, default=30, help="Burst concurrency (default: 30)"
    )
    all_parser.add_argument(
        "--users", type=int, default=30, help="Sustained user count (default: 30)"
    )
    all_parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Sustained duration in seconds (default: 300)",
    )
    all_parser.add_argument(
        "--think-time-min",
        type=float,
        default=2.0,
        help="Min think time (default: 2.0)",
    )
    all_parser.add_argument(
        "--think-time-max",
        type=float,
        default=8.0,
        help="Max think time (default: 8.0)",
    )
    add_common_args(all_parser)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    base_url = args.base_url or os.environ.get(
        "LOAD_TEST_BASE_URL", "http://localhost:8000"
    )
    api_key = args.api_key or os.environ.get("API_API_KEY", "")

    if not api_key:
        print("ERROR: No API key. Use --api-key or set API_API_KEY env var.")
        sys.exit(1)

    questions = load_questions(args.dataset, args.sample_size)

    endpoints = (
        list(ENDPOINT_PATHS.keys()) if args.endpoint == "all" else [args.endpoint]
    )

    all_metrics: list[TestMetrics] = []

    try:
        for endpoint_name in endpoints:
            endpoint_path = ENDPOINT_PATHS[endpoint_name]

            if args.test_type in ("burst", "all"):
                config = LoadTestConfig(
                    base_url=base_url,
                    api_key=api_key,
                    endpoint=endpoint_name,
                    test_type="burst",
                    concurrency=args.concurrency,
                    users=0,
                    duration_s=0,
                    top_k=args.top_k,
                    email="loadtest@test.com",
                    think_time_min=0,
                    think_time_max=0,
                    timeout_s=args.timeout,
                )
                print(
                    f"\n>>> Running BURST test on {endpoint_name} "
                    f"({args.concurrency} concurrent)..."
                )
                metrics = asyncio.run(run_burst_test(config, questions, endpoint_path))
                all_metrics.append(metrics)
                print(format_report(metrics))

            if args.test_type in ("sustained", "all"):
                config = LoadTestConfig(
                    base_url=base_url,
                    api_key=api_key,
                    endpoint=endpoint_name,
                    test_type="sustained",
                    concurrency=0,
                    users=args.users,
                    duration_s=args.duration,
                    top_k=args.top_k,
                    email="loadtest@test.com",
                    think_time_min=args.think_time_min,
                    think_time_max=args.think_time_max,
                    timeout_s=args.timeout,
                )
                print(
                    f"\n>>> Running SUSTAINED test on {endpoint_name} "
                    f"({args.users} users, {args.duration}s)..."
                )
                metrics = asyncio.run(
                    run_sustained_test(config, questions, endpoint_path)
                )
                all_metrics.append(metrics)
                print(format_report(metrics))

        if len(all_metrics) > 1:
            print_summary_table(all_metrics)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        if all_metrics:
            print_summary_table(all_metrics)
        sys.exit(130)


if __name__ == "__main__":
    main()
