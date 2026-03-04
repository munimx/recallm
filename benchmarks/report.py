"""Benchmark report formatter."""
from __future__ import annotations


def format_report(results: list[dict]) -> str:
    """Format benchmark results as a markdown table."""
    lines = [
        "## Benchmark Results",
        "",
        "| Use Case | Expected Hit Rate | Actual Hit Rate | Hits | Misses | Total |",
        "|----------|------------------|-----------------|------|--------|-------|",
    ]
    for r in results:
        hit_pct = f"{r['hit_rate']:.1%}"
        lines.append(
            f"| {r['use_case']} | {r['expected']} | {hit_pct} "
            f"| {r['hits']} | {r['misses']} | {r['total']} |"
        )
    lines.append("")
    lines.append(
        "_Note: Benchmarks use a hash-based fake embedder, not a real ML model. "
        "Actual hit rates with all-MiniLM-L6-v2 will differ. "
        "Run with a real embedder for production-representative numbers._"
    )
    return "\n".join(lines)
