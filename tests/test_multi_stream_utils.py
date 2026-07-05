# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager

import pytest
import torch

from vllm.utils.multi_stream_utils import execute_in_parallel


class _FakeEvent:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def record(self) -> None:
        self.calls.append(f"{self.name}.record")

    def wait(self) -> None:
        self.calls.append(f"{self.name}.wait")


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
