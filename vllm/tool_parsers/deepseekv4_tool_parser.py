# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.deepseekv32_tool_parser import DeepSeekV32ToolParser
from vllm.tool_parsers.structural_tag_registry import (
    get_enable_structured_outputs_in_reasoning,
    get_model_structural_tag,
)

ESCAPED_ARGUMENTS_PARAM_NAME = "__vllm_param_arguments__"


class DeepSeekV4ToolParser(DeepSeekV32ToolParser):
    """
    DeepSeek V4 DSML tool parser.

    V4 keeps the V3.2 DSML invoke/parameter grammar, but wraps tool calls in
    ``<｜DSML｜tool_calls>`` instead of ``<｜DSML｜function_calls>``.
    """

    tool_call_start_token: str = "<｜DSML｜tool_calls>"
    tool_call_end_token: str = "</｜DSML｜tool_calls>"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameter_complete_regex = re.compile(
            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</｜DSML｜parameter>',
            re.DOTALL,
        )

    def get_structural_tag(self, request: ChatCompletionRequest):
        return get_model_structural_tag(
            model="deepseek_v4",
            tools=request.tools,
            tool_choice=request.tool_choice,
            reasoning=get_enable_structured_outputs_in_reasoning(),
        )

    @staticmethod
    def _function_name(tool) -> str | None:
        if isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict):
                return function.get("name")
            return getattr(function, "name", None)
        return getattr(getattr(tool, "function", None), "name", None)

    @staticmethod
    def _function_parameters(tool):
        if isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict):
                return function.get("parameters")
            return getattr(function, "parameters", None)
        return getattr(getattr(tool, "function", None), "parameters", None)

    def _get_param_config(
        self,
        function_name: str | None,
        request: ChatCompletionRequest | None = None,
    ) -> dict[str, dict]:
        if not function_name:
            return {}

        tools = list(self.tools)
        if request and request.tools:
            tools.extend(request.tools)
        for tool in tools:
            if self._function_name(tool) != function_name:
                continue
            params = self._function_parameters(tool)
            if isinstance(params, dict):
                properties = params.get("properties")
                if isinstance(properties, dict):
                    return properties
            return {}

        return {}

    @staticmethod
    def _extract_param_name(param_name: str) -> str:
        if param_name == ESCAPED_ARGUMENTS_PARAM_NAME:
            return "arguments"
        return param_name

    def _coerce_param_value(
        self,
        value: str,
        *,
        string_attr: str,
        param_type,
    ) -> Any:
        if string_attr == "true":
            return value
        if param_type:
            return self._convert_param_value(value, param_type)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    @staticmethod
    def _repair_param_dict(
        param_dict: dict[str, Any],
        param_config: dict[str, dict],
    ) -> dict[str, Any]:
        allowed = set(param_config.keys())
        for wrapper in ("arguments", "input"):
            if set(param_dict.keys()) != {wrapper} or wrapper in allowed:
                continue
            inner = param_dict[wrapper]
            if isinstance(inner, str):
                try:
                    inner = json.loads(inner)
                except json.JSONDecodeError:
                    return param_dict
            if isinstance(inner, dict) and set(inner.keys()).issubset(allowed):
                return inner
        return param_dict

    def _parse_invoke_params(
        self,
        invoke_str: str,
        request: ChatCompletionRequest | None = None,
        function_name: str | None = None,
    ) -> dict[str, Any]:
        param_config = self._get_param_config(function_name, request=request)
        param_dict: dict[str, Any] = {}

        for param_name, string_attr, param_val in self.parameter_complete_regex.findall(
            invoke_str
        ):
            original_param_name = param_name
            param_name = self._extract_param_name(param_name)
            param_type = None
            if (
                original_param_name == ESCAPED_ARGUMENTS_PARAM_NAME
                and "arguments" in param_config
            ):
                param_type = param_config["arguments"].get("type")
            elif param_name in param_config and isinstance(
                param_config[param_name], dict
            ):
                param_type = param_config[param_name].get("type")

            param_dict[param_name] = self._coerce_param_value(
                param_val,
                string_attr=string_attr,
                param_type=param_type,
            )

        return self._repair_param_dict(param_dict, param_config)

    def _convert_params_with_schema(
        self,
        function_name: str,  # pylint: disable=unused-argument
        param_dict: dict[str, Any],
    ) -> dict[str, Any]:
        return param_dict
