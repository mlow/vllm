# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager

import pytest
import torch

from vllm.utils.multi_stream_utils import (
    CUDAGraphCaptureEventPool,
    execute_in_parallel,
)


class _FakeEvent:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def record(self) -> None:
        self.calls.append(f"{self.name}.record")

    def wait(self) -> None:
        self.calls.append(f"{self.name}.wait")


def test_cudagraph_capture_event_pool_isolates_capture_generations(monkeypatch):
    created = []

    def fake_event():
        event = object()
        created.append(event)
        return event

    monkeypatch.setattr(torch.cuda, "Event", fake_event)
    capturing = False
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)

    pool = CUDAGraphCaptureEventPool(2)
    assert pool.get() is pool.default_events

    capturing = True
    capture_a = pool.get()
    capture_b = pool.get()
    assert capture_a is not capture_b
    assert set(map(id, capture_a)).isdisjoint(map(id, capture_b))
    assert set(map(id, pool.default_events)).isdisjoint(map(id, capture_a))
    assert len(created) == 6


@pytest.mark.parametrize("enqueue_default_first", [False, True])
def test_execute_in_parallel_enqueue_order(monkeypatch, enqueue_default_first):
    calls: list[str] = []

    @contextmanager
    def fake_stream(_stream):
        calls.append("aux.enter")
        try:
            yield
        finally:
            calls.append("aux.exit")

    monkeypatch.setattr(torch.cuda, "stream", fake_stream)
    start = _FakeEvent("start", calls)
    done = _FakeEvent("done", calls)

    default_result, aux_results = execute_in_parallel(
        lambda: calls.append("default") or "default-result",
        [lambda: calls.append("aux") or "aux-result"],
        start,
        [done],
        [object()],
        enable=True,
        enqueue_default_first=enqueue_default_first,
    )

    assert default_result == "default-result"
    assert aux_results == ["aux-result"]
    assert calls == (
        [
            "start.record",
            "default",
            "aux.enter",
            "start.wait",
            "aux",
            "done.record",
            "aux.exit",
            "done.wait",
        ]
        if enqueue_default_first
        else [
            "start.record",
            "aux.enter",
            "start.wait",
            "aux",
            "done.record",
            "aux.exit",
            "default",
            "done.wait",
        ]
    )
