import threading

import pytest

from finetuner.core.engine import ENGINE


def test_calls_run_on_engine_thread():
    assert ENGINE.call(lambda: threading.current_thread().name) == "finetuner-mlx-engine"


def test_result_and_error_propagation():
    assert ENGINE.call(lambda a, b: a + b, 2, 3) == 5
    with pytest.raises(ValueError, match="boom"):
        ENGINE.call(lambda: (_ for _ in ()).throw(ValueError("boom")))


def test_concurrent_callers_serialized():
    seen = []

    def work(i):
        seen.append(threading.current_thread().name)
        return i * 2

    results = []
    threads = [threading.Thread(target=lambda i=i: results.append(ENGINE.call(work, i)))
               for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [i * 2 for i in range(8)]
    assert set(seen) == {"finetuner-mlx-engine"}


def test_nested_call_from_engine_thread():
    # A job target running on the engine thread may call helpers that also
    # route through ENGINE.call — must not deadlock.
    assert ENGINE.call(lambda: ENGINE.call(lambda: 42)) == 42
