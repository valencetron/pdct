"""Background graph-rebuild rate-limit (2026-07-22 latency fix).

events.jsonl's mtime is in the graph cache key and the daemon appends to it
every turn, so the key misses every turn. Without a cap that kicks a full-vault
re-embed per turn — a perpetual GIL hog. These tests prove _kick_background_rebuild
now spawns at most one rebuild per _GRAPH_REBUILD_MIN_INTERVAL_S per identity,
while still allowing a rebuild once the interval elapses. The stale graph keeps
serving between rebuilds (unchanged behaviour), so content/ranking is untouched."""
import time
from pathlib import Path

from dct.retrieval import service


def _kick(sk):
    key = ("v1", "ev", 123.0, sk[1], sk[2], sk[3], sk[4], 0.0)
    return service._kick_background_rebuild(
        key, sk, Path("ev"),
        topic_id=sk[1], ignore_feedback=sk[2],
        vec_near_flag=sk[3], vec_near_thresh=sk[4])


def _setup(monkeypatch, interval):
    calls = []

    def fake_build(key, stable_key, events_path, **kw):
        calls.append(stable_key)
        return "GRAPH"

    monkeypatch.setattr(service, "_build_graph_now", fake_build)
    monkeypatch.setattr(service, "_GRAPH_REBUILD_MIN_INTERVAL_S", interval)
    service._GRAPH_LAST_REBUILD.clear()
    service._GRAPH_REBUILDING.clear()
    return calls


def test_second_kick_within_interval_is_rate_limited(monkeypatch):
    calls = _setup(monkeypatch, interval=100.0)
    sk = ("ev", 1, False, True, 0.7)

    assert _kick(sk) is True            # first: spawns
    service.join_rebuilds(timeout=5)
    assert len(calls) == 1

    assert _kick(sk) is False           # second (immediate): rate-limited
    service.join_rebuilds(timeout=5)
    assert len(calls) == 1              # no extra rebuild


def test_kick_allowed_again_after_interval(monkeypatch):
    calls = _setup(monkeypatch, interval=100.0)
    sk = ("ev", 1, False, True, 0.7)

    assert _kick(sk) is True
    service.join_rebuilds(timeout=5)
    assert len(calls) == 1

    # simulate the interval elapsing by backdating the last-rebuild stamp
    service._GRAPH_LAST_REBUILD[sk] = time.monotonic() - 200.0
    assert _kick(sk) is True            # interval passed: rebuild allowed
    service.join_rebuilds(timeout=5)
    assert len(calls) == 2


def test_distinct_identities_are_independent(monkeypatch):
    calls = _setup(monkeypatch, interval=100.0)
    a = ("ev", 1, False, True, 0.7)
    b = ("ev", 2, False, True, 0.7)   # different topic_id → different identity

    assert _kick(a) is True
    assert _kick(b) is True            # b not blocked by a's stamp
    service.join_rebuilds(timeout=5)
    assert sorted(c[1] for c in calls) == [1, 2]


def test_interval_zero_disables_rate_limit(monkeypatch):
    # interval=0 → every kick allowed (the escape hatch tests/tuning use).
    calls = _setup(monkeypatch, interval=0.0)
    sk = ("ev", 1, False, True, 0.7)
    assert _kick(sk) is True
    service.join_rebuilds(timeout=5)
    assert _kick(sk) is True
    service.join_rebuilds(timeout=5)
    assert len(calls) == 2


def test_parse_rebuild_interval_rejects_bad_values():
    # Codex r1 P2: NaN silently disables the cap, inf freezes refresh forever.
    p = service._parse_rebuild_interval
    assert p("120") == 120.0
    assert p(None) == 300.0       # unset -> default
    assert p("abc") == 300.0      # non-numeric -> default
    assert p("-5") == 300.0       # negative -> default
    assert p("nan") == 300.0      # NaN -> default
    assert p("inf") == 300.0      # inf -> default
    assert p("0") == 0.0          # 0 honoured (explicit no-rate-limit)


def test_rate_limit_map_stays_bounded(monkeypatch):
    # topic_id is caller-supplied; the stamp map must not grow unboundedly.
    _setup(monkeypatch, interval=100.0)
    for i in range(service._GRAPH_LATEST_MAX + 8):
        _kick(("ev", i, False, True, 0.7))   # distinct identity each kick
    service.join_rebuilds(timeout=10)
    assert len(service._GRAPH_LAST_REBUILD) <= service._GRAPH_LATEST_MAX
