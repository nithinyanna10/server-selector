import pytest

from selector import Request, Server, select_server


def make_request(**overrides):
    defaults = dict(model_name="llama-3-70b", min_gpu_memory_gb=40.0)
    defaults.update(overrides)
    return Request(**defaults)


def make_server(**overrides):
    defaults = dict(name="s1", latency_ms=100, gpu_util_pct=50, queue_len=2, gpu_memory_gb=80, has_model=True)
    defaults.update(overrides)
    return Server(**defaults)


def test_normal_case_picks_best_server():
    request = make_request()
    servers = [
        make_server(name="slow", latency_ms=300, gpu_util_pct=80, queue_len=10, gpu_memory_gb=80, has_model=True),
        make_server(name="best", latency_ms=50, gpu_util_pct=10, queue_len=0, gpu_memory_gb=80, has_model=True),
        make_server(name="mid", latency_ms=150, gpu_util_pct=50, queue_len=5, gpu_memory_gb=80, has_model=True),
    ]
    result = select_server(request, servers)
    assert result.selected == "best"
    assert result.score == 1.0


def test_missing_field_raises_value_error_with_message():
    request = make_request()
    servers = [make_server(name=None)]
    with pytest.raises(ValueError, match="missing required field: name"):
        select_server(request, servers)


def test_negative_latency_raises_value_error():
    request = make_request()
    servers = [make_server(latency_ms=-1)]
    with pytest.raises(ValueError, match="latency_ms"):
        select_server(request, servers)


def test_gpu_util_out_of_range_raises_value_error():
    request = make_request()
    servers = [make_server(gpu_util_pct=150)]
    with pytest.raises(ValueError, match="gpu_util_pct"):
        select_server(request, servers)


def test_empty_server_list_returns_no_capacity():
    request = make_request()
    result = select_server(request, [])
    assert result.selected is None
    assert result.score is None
    assert result.reason == "No server can run this request"
    assert result.rejected == []


def test_all_servers_filtered_by_memory_returns_rejection_reasons():
    request = make_request(min_gpu_memory_gb=80.0)
    servers = [
        make_server(name="s1", gpu_memory_gb=24, has_model=True),
        make_server(name="s2", gpu_memory_gb=40, has_model=True),
    ]
    result = select_server(request, servers)
    assert result.selected is None
    assert result.rejected == [
        ("s1", "insufficient gpu memory"),
        ("s2", "insufficient gpu memory"),
    ]


def test_tie_resolved_by_lower_latency():
    request = make_request()
    servers = [
        make_server(name="a", latency_ms=100, gpu_util_pct=50, queue_len=2, gpu_memory_gb=80, has_model=True),
        make_server(name="b", latency_ms=50, gpu_util_pct=50, queue_len=2, gpu_memory_gb=80, has_model=True),
    ]
    result = select_server(request, servers)
    assert result.selected == "b"


def test_tie_with_equal_latency_resolved_by_name():
    request = make_request()
    servers = [
        make_server(name="zebra", latency_ms=100, gpu_util_pct=50, queue_len=2, gpu_memory_gb=80, has_model=True),
        make_server(name="alpha", latency_ms=100, gpu_util_pct=50, queue_len=2, gpu_memory_gb=80, has_model=True),
    ]
    result = select_server(request, servers)
    assert result.selected == "alpha"


def test_same_input_twice_is_deterministic():
    request = make_request()
    servers = [
        make_server(name="a", latency_ms=100, gpu_util_pct=50, queue_len=2, gpu_memory_gb=80, has_model=True),
        make_server(name="b", latency_ms=60, gpu_util_pct=20, queue_len=1, gpu_memory_gb=80, has_model=True),
    ]
    result1 = select_server(request, servers)
    result2 = select_server(request, servers)
    assert result1 == result2


def test_gpu_util_change_flips_winner():
    request = make_request()
    servers_before = [
        make_server(name="a", latency_ms=100, gpu_util_pct=10, queue_len=2, gpu_memory_gb=80, has_model=True),
        make_server(name="b", latency_ms=100, gpu_util_pct=90, queue_len=2, gpu_memory_gb=80, has_model=True),
    ]
    result_before = select_server(request, servers_before)
    assert result_before.selected == "a"

    servers_after = [
        make_server(name="a", latency_ms=100, gpu_util_pct=90, queue_len=2, gpu_memory_gb=80, has_model=True),
        make_server(name="b", latency_ms=100, gpu_util_pct=10, queue_len=2, gpu_memory_gb=80, has_model=True),
    ]
    result_after = select_server(request, servers_after)
    assert result_after.selected == "b"


def test_extreme_latency_still_handled_deterministically():
    request = make_request()
    servers = [
        make_server(name="a", latency_ms=10_000, gpu_util_pct=50, queue_len=2, gpu_memory_gb=80, has_model=True),
        make_server(name="b", latency_ms=10, gpu_util_pct=50, queue_len=2, gpu_memory_gb=80, has_model=True),
    ]
    result1 = select_server(request, servers)
    result2 = select_server(request, servers)
    assert result1 == result2
    assert result1.selected == "b"


def test_model_not_available_anywhere_returns_no_capacity():
    request = make_request()
    servers = [
        make_server(name="s1", has_model=False),
        make_server(name="s2", has_model=False),
    ]
    result = select_server(request, servers)
    assert result.selected is None
    assert result.rejected == [
        ("s1", "model not available on this server"),
        ("s2", "model not available on this server"),
    ]
