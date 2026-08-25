# Server Selector

Picks the best server to route an LLM inference request to, from a snapshot of server load.

## Run

```
python -m pytest test_selector.py -v
```

Example / demo:

```
python selector.py
```

## Scoring method

Each eligible server's `latency_ms`, `gpu_util_pct`, and `queue_len` are min-max normalized
across the eligible set, then inverted so 1.0 is the best (lowest raw) value and 0.0 is the
worst. If every eligible server has the same value for a metric, there's nothing to compare,
so that metric scores 1.0 for everyone instead of dividing by zero.

The three normalized scores are combined as a weighted sum:

```
score = 0.35 * latency_score + 0.25 * gpu_score + 0.40 * queue_score
```

Queue length gets the highest weight because it's the most direct signal of how long a new
request will wait before it even starts running. Latency is next, it reflects current
responsiveness. GPU utilization is weighted lowest because a busy GPU can still serve a
request quickly if the queue is short; it's a secondary signal compared to queue depth or
observed latency.

Ties break deterministically: higher score wins, then lower raw latency, then alphabetically
earlier name. This guarantees the same input always produces the same output.

## Assumptions

- Server metrics are a single point-in-time snapshot, no history or trend.
- No cache awareness (KV cache, prefix cache) factors into the score.
- Server-reported metrics are trusted as accurate, no adversarial or faulty reporting.
- `has_model` and `gpu_memory_gb` are hard requirements, not scored, a server either can
  run the request or it can't.

## Limitations

- No prefix-cache locality, requests aren't routed toward servers that already have
  relevant KV cache warm.
- No P2C (power of two choices) randomization, so under high concurrent request volume
  many callers could pick the same "best" server at once (thundering herd).
- No SLO-based routing, all requests are scored the same way regardless of latency budget.
- No prefill/decode split awareness, treats every server as a single undifferentiated pool.

## First 3 production improvements

1. **KV cache-aware routing via prefix matching.** Route requests toward servers that
   already hold a matching prompt prefix in cache (the llm-d EPP pattern), which cuts
   time-to-first-token substantially for repeated or chat-style prompts.
2. **Power of Two Choices for herd avoidance.** Instead of always picking the single best
   scored server, sample two candidates at random and pick the better of the two. This
   avoids every concurrent request piling onto the same "best" server.
3. **Health-checked staleness with EWMA smoothing on metrics.** Reject or down-weight
   servers whose last metrics report is too old, and smooth queue/latency/util readings
   with an exponentially weighted moving average so a single noisy sample can't flip
   routing decisions.
