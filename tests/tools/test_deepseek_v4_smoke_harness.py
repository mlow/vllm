# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path


def _load_smoke_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "tools" / "sm120" / "deepseek_v4_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("deepseek_v4_smoke", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _chat_response(content="", finish_reason="stop", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"finish_reason": finish_reason, "message": message}]}


def test_basic_text_expectation_is_case_insensitive():
    smoke = _load_smoke_module()
    expectation = smoke.Expectation(all_terms=("Paris", "France"))

    result = smoke.check_response(
        expectation,
        _chat_response("the capital of france is paris."),
    )

    assert result.ok


def test_tool_call_expectation_requires_finish_reason_and_name():
    smoke = _load_smoke_module()
    expectation = smoke.Expectation(tool_name="read", finish_reason="tool_calls")
    tool_calls = [
        {
            "type": "function",
            "function": {"name": "read", "arguments": '{"path": "/tmp/a"}'},
        }
    ]

    assert smoke.check_response(
        expectation,
        _chat_response(finish_reason="tool_calls", tool_calls=tool_calls),
    ).ok
    assert not smoke.check_response(
        expectation,
        _chat_response(
            "I should read the file first.", finish_reason="stop", tool_calls=[]
        ),
    ).ok


def test_html_artifact_expectation_rejects_reasoning_without_code():
    smoke = _load_smoke_module()
    expectation = smoke.Expectation(
        require_html_artifact=True,
        all_terms=("aquarium", "fish", "food"),
    )

    result = smoke.check_response(
        expectation,
        _chat_response(
            "I will create an aquarium and think through fish behavior first."
        ),
    )

    assert not result.ok


def test_default_cases_cover_collected_smoke_scenarios():
    smoke = _load_smoke_module()
    case_names = {
        case.name for case in smoke.build_cases("deepseek-ai/DeepSeek-V4-Flash")
    }

    assert {
        "math_7_times_8",
        "capital_of_france",
        "spanish_greeting",
        "openclaw_read_tool",
        "aquarium_html",
        "clock_html",
    } <= case_names


def test_build_payload_applies_case_defaults_and_tools():
    smoke = _load_smoke_module()
    case = next(
        case
        for case in smoke.build_cases("deepseek-ai/DeepSeek-V4-Flash")
        if case.name == "openclaw_read_tool"
    )

    payload = smoke.build_payload(case, default_max_tokens=128, default_temperature=0.0)

    assert payload["model"] == "deepseek-ai/DeepSeek-V4-Flash"
    assert payload["max_tokens"] == case.max_tokens
    assert payload["temperature"] == case.temperature
    assert payload["tools"][0]["function"]["name"] == "read"
    assert payload["tool_choice"] == "auto"
