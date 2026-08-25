"""Scenario comparison, not a test suite.

This file exists to answer the interview question "did you compare weight
configurations and other allocation methods before picking the current
scoring approach". It runs 5 hand picked scenarios through 4 allocation
methods (round robin, least loaded, lowest latency, and weighted sum) plus
3 alternate weight variants of the weighted sum on top of the weights
actually used in selector.py, then prints a table of which methods pick the
server a reasonable person would expect for each scenario. Nothing here
asserts, nothing here fails the process, the table is the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from selector import Request, Server


@dataclass
class Scenario:
    name: str
    description: str
    servers: list[Server]
    expected: str
    request: Request = field(
        default_factory=lambda: Request(model_name="llama-3-70b", min_gpu_memory_gb=40.0)
    )


SCENARIOS: list[Scenario] = [
    Scenario(
        name="latency vs utilization",
        description="When queues are equal, latency should beat GPU utilization because util is a noisy signal.",
        servers=[
            Server("server-near", latency_ms=30, gpu_util_pct=85, queue_len=0, gpu_memory_gb=80, has_model=True),
            Server("server-far", latency_ms=200, gpu_util_pct=15, queue_len=0, gpu_memory_gb=80, has_model=True),
        ],
        expected="server-near",
    ),
    Scenario(
        name="queue dominates when large",
        description="A queue of 20 costs more wait time than 140ms of extra network latency.",
        servers=[
            Server("server-empty-slow", latency_ms=180, gpu_util_pct=50, queue_len=0, gpu_memory_gb=80, has_model=True),
            Server("server-busy-fast", latency_ms=40, gpu_util_pct=50, queue_len=20, gpu_memory_gb=80, has_model=True),
        ],
        expected="server-empty-slow",
    ),
    Scenario(
        name="queue is the only real difference",
        description="When latency and util are tied, the empty queue obviously wins.",
        servers=[
            Server("server-a", latency_ms=100, gpu_util_pct=50, queue_len=5, gpu_memory_gb=80, has_model=True),
            Server("server-b", latency_ms=100, gpu_util_pct=50, queue_len=0, gpu_memory_gb=80, has_model=True),
            Server("server-c", latency_ms=100, gpu_util_pct=50, queue_len=5, gpu_memory_gb=80, has_model=True),
        ],
        expected="server-b",
    ),
    Scenario(
        name="latency is the only real difference",
        description="When queue and util are tied, lower latency wins.",
        servers=[
            Server("server-a", latency_ms=200, gpu_util_pct=50, queue_len=2, gpu_memory_gb=80, has_model=True),
            Server("server-b", latency_ms=50, gpu_util_pct=50, queue_len=2, gpu_memory_gb=80, has_model=True),
        ],
        expected="server-b",
    ),
    Scenario(
        name="everything roughly equal",
        description="True tie, deterministic tie-break must fire. Alpha order wins.",
        servers=[
            Server("server-a", latency_ms=100, gpu_util_pct=50, queue_len=2, gpu_memory_gb=80, has_model=True),
            Server("server-b", latency_ms=100, gpu_util_pct=50, queue_len=2, gpu_memory_gb=80, has_model=True),
        ],
        expected="server-a",
    ),
]


def _eligible(request: Request, servers: list[Server]) -> list[Server]:
    return [s for s in servers if s.has_model and s.gpu_memory_gb >= request.min_gpu_memory_gb]


_round_robin_counter = 0


def reset_round_robin() -> None:
    """Reset the round robin counter so a comparison run is reproducible."""
    global _round_robin_counter
    _round_robin_counter = 0


def round_robin(request: Request, servers: list[Server]) -> str:
    """Cycle through eligible servers in order, ignoring load entirely."""
    global _round_robin_counter
    pool = _eligible(request, servers)
    pick = pool[_round_robin_counter % len(pool)]
    _round_robin_counter += 1
    return pick.name


def least_loaded(request: Request, servers: list[Server]) -> str:
    """Pick the shortest queue, tie-break by name."""
    pool = _eligible(request, servers)
    return min(pool, key=lambda s: (s.queue_len, s.name)).name


def lowest_latency(request: Request, servers: list[Server]) -> str:
    """Pick the lowest latency, tie-break by name."""
    pool = _eligible(request, servers)
    return min(pool, key=lambda s: (s.latency_ms, s.name)).name


def _normalize_invert(values: list[float]) -> list[float]:
    """Min-max normalize then invert, so 1.0 is the lowest (best) raw value. Copied from selector.py on purpose."""
    lo, hi = min(values), max(values)
    if lo == hi:
        return [1.0] * len(values)
    return [1.0 - (v - lo) / (hi - lo) for v in values]


def weighted_sum(request: Request, servers: list[Server], weights: tuple[float, float, float]) -> str:
    """Same scoring algorithm as selector.py, weights passed in as (latency, gpu, queue)."""
    pool = _eligible(request, servers)
    w_lat, w_gpu, w_queue = weights
    latency_scores = _normalize_invert([s.latency_ms for s in pool])
    gpu_scores = _normalize_invert([s.gpu_util_pct for s in pool])
    queue_scores = _normalize_invert([s.queue_len for s in pool])
    scores = {
        s.name: w_lat * lat + w_gpu * gpu + w_queue * q
        for s, lat, gpu, q in zip(pool, latency_scores, gpu_scores, queue_scores, strict=True)
    }
    return min(pool, key=lambda s: (-scores[s.name], s.latency_ms, s.name)).name


# order is (latency, gpu, queue) to match weighted_sum's weights argument,
# same order as WEIGHT_LATENCY, WEIGHT_GPU, WEIGHT_QUEUE in selector.py
WEIGHT_VARIANTS: list[tuple[str, tuple[float, float, float]]] = [
    ("equal", (0.333, 0.333, 0.334)),
    ("latency-heavy", (0.60, 0.20, 0.20)),
    ("queue-heavy", (0.20, 0.20, 0.60)),
    ("chosen (0.35/0.25/0.40)", (0.35, 0.25, 0.40)),
]


def _run_row(label: str, pick_fn) -> tuple[str, list[str], int]:
    """Run one method across every scenario, return (label, cell strings, pass count)."""
    cells = []
    passed = 0
    for scenario in SCENARIOS:
        picked = pick_fn(scenario.request, scenario.servers)
        ok = picked == scenario.expected
        passed += ok
        cells.append(f"{picked} ({'PASS' if ok else 'FAIL'})")
    return label, cells, passed


def _print_table(rows: list[tuple[str, list[str], int]]) -> None:
    headers = ["method"] + [f"S{i + 1}" for i in range(len(SCENARIOS))] + ["passed"]
    method_w = max(len(headers[0]), max(len(r[0]) for r in rows))
    col_widths = [
        max(len(headers[i + 1]), max(len(r[1][i]) for r in rows))
        for i in range(len(SCENARIOS))
    ]
    passed_w = max(len(headers[-1]), len("N/5"))

    def fmt_row(label: str, cells: list[str], passed_text: str) -> str:
        parts = [label.ljust(method_w)]
        parts += [cell.ljust(w) for cell, w in zip(cells, col_widths, strict=True)]
        parts.append(passed_text.ljust(passed_w))
        return "  ".join(parts)

    print(fmt_row(headers[0], headers[1:-1], headers[-1]))
    print("-" * (method_w + sum(col_widths) + passed_w + 2 * (len(headers) - 1)))
    for label, cells, passed in rows:
        print(fmt_row(label, cells, f"{passed}/{len(SCENARIOS)}"))


def main() -> None:
    rows: list[tuple[str, list[str], int]] = []

    reset_round_robin()
    rows.append(_run_row("round_robin", round_robin))
    rows.append(_run_row("least_loaded", least_loaded))
    rows.append(_run_row("lowest_latency", lowest_latency))
    for label, weights in WEIGHT_VARIANTS:
        rows.append(_run_row(f"weighted_sum[{label}]", lambda req, srv, w=weights: weighted_sum(req, srv, w)))

    _print_table(rows)

    chosen_label = "weighted_sum[chosen (0.35/0.25/0.40)]"
    chosen_passed = next(passed for label, _, passed in rows if label == chosen_label)
    others = [(label, passed) for label, _, passed in rows if label != chosen_label]
    next_label, next_passed = max(others, key=lambda x: x[1])

    print()
    print(
        f"Chosen weights (0.35/0.25/0.40) pass {chosen_passed}/{len(SCENARIOS)}. "
        f"Next best: {next_label} passes {next_passed}/{len(SCENARIOS)}. "
        "Justification for the chosen weights is in README under Scoring."
    )


if __name__ == "__main__":
    main()
