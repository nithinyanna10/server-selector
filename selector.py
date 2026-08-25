"""Pick the best server to route an LLM request to, given a snapshot of server load."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Request:
    model_name: str | None = None
    min_gpu_memory_gb: float | None = None


@dataclass
class Server:
    name: str | None = None
    latency_ms: float | None = None
    gpu_util_pct: float | None = None
    queue_len: int | None = None
    gpu_memory_gb: float | None = None
    has_model: bool | None = None


@dataclass
class Result:
    selected: str | None
    score: float | None
    reason: str
    rejected: list[tuple[str, str]] = field(default_factory=list)


WEIGHT_LATENCY = 0.35
WEIGHT_GPU = 0.25
WEIGHT_QUEUE = 0.40


def validate_request(request: Request) -> None:
    """Raise ValueError if the request is malformed."""
    if request.model_name is None:
        raise ValueError("request missing required field: model_name")
    if request.model_name == "":
        raise ValueError("model_name must not be empty")
    if request.min_gpu_memory_gb is None:
        raise ValueError("request missing required field: min_gpu_memory_gb")
    if request.min_gpu_memory_gb < 0:
        raise ValueError("min_gpu_memory_gb must not be negative")


def validate_server(server: Server) -> None:
    """Raise ValueError if a server record is malformed."""
    required = ("name", "latency_ms", "gpu_util_pct", "queue_len", "gpu_memory_gb", "has_model")
    for field_name in required:
        if getattr(server, field_name) is None:
            raise ValueError(f"server missing required field: {field_name}")
    if server.latency_ms < 0:
        raise ValueError(f"server {server.name}: latency_ms must not be negative")
    if not isinstance(server.queue_len, int) or isinstance(server.queue_len, bool):
        raise ValueError(f"server {server.name}: queue_len must be an integer")
    if server.queue_len < 0:
        raise ValueError(f"server {server.name}: queue_len must not be negative")
    if server.gpu_memory_gb < 0:
        raise ValueError(f"server {server.name}: gpu_memory_gb must not be negative")
    if not (0 <= server.gpu_util_pct <= 100):
        raise ValueError(f"server {server.name}: gpu_util_pct must be between 0 and 100")


def filter_eligible(request: Request, servers: list[Server]) -> tuple[list[Server], list[tuple[str, str]]]:
    """Split servers into (eligible, rejected-with-reason). A server can fail both checks at once."""
    eligible: list[Server] = []
    rejected: list[tuple[str, str]] = []
    for s in servers:
        reasons = []
        if not s.has_model:
            reasons.append("model not available on this server")
        if s.gpu_memory_gb < request.min_gpu_memory_gb:
            reasons.append("insufficient gpu memory")
        if reasons:
            rejected.append((s.name, "; ".join(reasons)))
        else:
            eligible.append(s)
    return eligible, rejected


def _normalize_invert(values: list[float]) -> list[float]:
    """Min-max normalize then invert, so 1.0 is the lowest (best) raw value."""
    lo, hi = min(values), max(values)
    if lo == hi:
        return [1.0] * len(values)
    return [1.0 - (v - lo) / (hi - lo) for v in values]


def _score_all(eligible: list[Server]) -> dict[str, float]:
    """Weighted score per server name, using min-max normalization across eligible servers."""
    latency_scores = _normalize_invert([s.latency_ms for s in eligible])
    gpu_scores = _normalize_invert([s.gpu_util_pct for s in eligible])
    queue_scores = _normalize_invert([s.queue_len for s in eligible])
    scores = {}
    for s, lat, gpu, q in zip(eligible, latency_scores, gpu_scores, queue_scores, strict=True):
        scores[s.name] = WEIGHT_LATENCY * lat + WEIGHT_GPU * gpu + WEIGHT_QUEUE * q
    return scores


def _pick_winner(eligible: list[Server], scores: dict[str, float]) -> Server:
    """Highest score wins, ties broken by lower latency then by name."""
    return min(eligible, key=lambda s: (-scores[s.name], s.latency_ms, s.name))


def _build_reason(winner: Server, eligible: list[Server]) -> str:
    """Describe the signals the winner led on, in weight order (queue, latency, gpu)."""
    # values below are raw input, no arithmetic applied to them, so exact equality is safe
    min_queue = min(s.queue_len for s in eligible)
    min_latency = min(s.latency_ms for s in eligible)
    min_gpu = min(s.gpu_util_pct for s in eligible)

    signals = []
    if winner.queue_len == min_queue:
        signals.append("shortest queue")
    if winner.latency_ms == min_latency:
        signals.append("lowest latency")
    if winner.gpu_util_pct == min_gpu:
        signals.append("lowest gpu utilization")

    if not signals:
        signal_text = "the best overall score"
    elif len(signals) == 1:
        signal_text = signals[0]
    else:
        signal_text = ", ".join(signals[:-1]) + " and " + signals[-1]

    return f"{winner.name} selected for {signal_text}."


def select_server(request: Request, servers: list[Server]) -> Result:
    """Validate, filter, score, and pick a server for the request."""
    validate_request(request)
    for s in servers:
        validate_server(s)

    eligible, rejected = filter_eligible(request, servers)
    if not eligible:
        return Result(selected=None, score=None, reason="No server can run this request", rejected=rejected)

    scores = _score_all(eligible)
    winner = _pick_winner(eligible, scores)
    reason = _build_reason(winner, eligible)

    return Result(
        selected=winner.name,
        score=round(scores[winner.name], 3),
        reason=reason,
        rejected=rejected,
    )


if __name__ == "__main__":
    demo_request = Request(model_name="llama-3-70b", min_gpu_memory_gb=40.0)
    demo_servers = [
        Server(name="gpu-a", latency_ms=120, gpu_util_pct=70, queue_len=3, gpu_memory_gb=80, has_model=True),
        Server(name="gpu-b", latency_ms=80, gpu_util_pct=40, queue_len=1, gpu_memory_gb=80, has_model=True),
        Server(name="gpu-c", latency_ms=200, gpu_util_pct=20, queue_len=0, gpu_memory_gb=24, has_model=True),
    ]
    result = select_server(demo_request, demo_servers)
    print(result)
