# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Exact-match scoring for ToolAlpaca tool calls.

Mirrors the offline scorer in ``LlamaFactory/0_tool_eval/toolcall_parse.py`` so
an RL reward and the reported test-set ``exact`` are the same number.  Calls are
compared as unordered bags: ToolAlpaca gold lists a plan whose steps are often
independent, so ordering is not meaningful, but multiplicity is (a query asking
for small/medium/large needs three calls).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _load_json_blob(payload: str) -> Any:
    payload = payload.strip()
    if not payload:
        return None
    if payload.startswith(("{", "[")):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            pass
    starts = [i for i in (payload.find("{"), payload.find("[")) if i != -1]
    if not starts:
        return None
    start = min(starts)
    end = payload.rfind("]" if payload[start] == "[" else "}")
    if end < start:
        return None
    try:
        return json.loads(payload[start : end + 1])
    except json.JSONDecodeError:
        return None


def _as_calls(obj: Any) -> list[tuple[str, Any]]:
    if isinstance(obj, dict) and "name" in obj:
        return [(obj.get("name"), obj.get("arguments", {}))]
    if isinstance(obj, list):
        return [
            (item.get("name"), item.get("arguments", {}))
            for item in obj
            if isinstance(item, dict) and "name" in item
        ]
    return []


def parse_calls(text: str) -> list[tuple[str, Any]] | None:
    """Every call the model emitted, or None when nothing parses."""
    matches = TOOL_CALL.findall(text or "")
    # The base model often drops the wrapper tags; scoring bare JSON keeps the
    # comparison about tool choice rather than formatting.
    blobs = matches if matches else [text or ""]
    calls: list[tuple[str, Any]] = []
    for blob in blobs:
        calls.extend(_as_calls(_load_json_blob(blob)))
    return calls or None


def _canon(arguments: Any) -> str:
    return json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)


def compute_score(solution_str: str, ground_truth: str | list, extra_info: dict | None = None) -> dict:
    gold = _as_calls(json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth)
    pred = parse_calls(solution_str)

    wrapped = bool(TOOL_CALL.search(solution_str or ""))
    if pred is None:
        return {"score": 0.0, "parse": 0.0, "wrapped": float(wrapped), "name": 0.0, "args": 0.0}

    name_ok = Counter(n for n, _ in pred) == Counter(n for n, _ in gold)
    args_ok = Counter(_canon(a) for _, a in pred) == Counter(_canon(a) for _, a in gold)
    exact = Counter((n, _canon(a)) for n, a in pred) == Counter((n, _canon(a)) for n, a in gold)

    return {
        "score": float(exact),
        "parse": 1.0,
        "wrapped": float(wrapped),
        "name": float(name_ok),
        "args": float(args_ok),
    }
