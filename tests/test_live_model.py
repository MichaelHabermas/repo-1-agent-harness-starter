"""
Tests for the live OpenRouter model path.

Most tests use canned JSON — no network, no key, no spend.
One optional end-to-end test runs only when OPENROUTER_API_KEY is set.

    python3 tests/test_live_model.py
    python3 -m pytest tests/ -q
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.model import (
    Final,
    OpenRouterModel,
    ToolCall,
    Turn,
    build_openrouter_request,
    estimate_cost_usd,
    parse_openrouter_decision,
    tools_as_openai_schema,
    turns_to_messages,
    usage_from_response,
)


# --------------------------------------------------------------------------
# Fixtures (no network)
# --------------------------------------------------------------------------

CANNED_TOOL_RESPONSE = {
    "id": "gen-test",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "get_pto_balance",
                    "arguments": '{"employee_id": "e-1001"}',
                },
            }],
        },
        "finish_reason": "tool_calls",
    }],
    "usage": {
        "prompt_tokens": 120,
        "completion_tokens": 25,
        "total_tokens": 145,
    },
}

CANNED_FINAL_RESPONSE = {
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "You have 12 days left.",
        },
        "finish_reason": "stop",
    }],
    "usage": {"prompt_tokens": 80, "completion_tokens": 12, "total_tokens": 92},
}

CANNED_BAD_TOOL_NAME = {
    "choices": [{
        "message": {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_x",
                "type": "function",
                "function": {
                    "name": "delete_everything",
                    "arguments": "{}",
                },
            }],
        },
    }],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

CANNED_MALFORMED_ARGS = {
    "choices": [{
        "message": {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_y",
                "type": "function",
                "function": {
                    "name": "get_pto_balance",
                    "arguments": "not-json{{",
                },
            }],
        },
    }],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

CANNED_MISSING_NAME = {
    "choices": [{
        "message": {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_z",
                "type": "function",
                "function": {
                    "arguments": '{"employee_id": "e-1001"}',
                },
            }],
        },
    }],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}


class TestRequestBuilding(unittest.TestCase):
    def test_tools_schema_includes_registry_tools(self):
        tools = tools_as_openai_schema()
        names = {t["function"]["name"] for t in tools}
        self.assertIn("get_pto_balance", names)
        self.assertIn("submit_pto_request", names)
        self.assertIn("search_handbook", names)

    def test_build_request_has_model_system_tools(self):
        turns = [Turn(role="user", content="how many days off do I have?")]
        body = build_openrouter_request(
            model="google/gemini-2.5-flash-lite",
            system="You are the assistant.",
            turns=turns,
        )
        self.assertEqual(body["model"], "google/gemini-2.5-flash-lite")
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(body["messages"][1]["role"], "user")
        self.assertTrue(body["tools"])
        self.assertEqual(body["tool_choice"], "auto")
        # Never put a key in the request body (key is a header only).
        blob = json.dumps(body)
        self.assertNotIn("Authorization", blob)
        self.assertNotIn("api_key", blob.lower())

    def test_turns_to_messages_tags_observations(self):
        turns = [
            Turn(role="user", content="q"),
            Turn(role="observation", content="12 days", tool="get_pto_balance", ok=True),
        ]
        msgs = turns_to_messages(turns)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertIn("[tool:get_pto_balance ok]", msgs[1]["content"])


class TestResponseParsing(unittest.TestCase):
    def test_parse_tool_call(self):
        d = parse_openrouter_decision(CANNED_TOOL_RESPONSE)
        self.assertIsInstance(d, ToolCall)
        assert isinstance(d, ToolCall)
        self.assertEqual(d.name, "get_pto_balance")
        self.assertEqual(d.args["employee_id"], "e-1001")

    def test_parse_final(self):
        d = parse_openrouter_decision(CANNED_FINAL_RESPONSE)
        self.assertIsInstance(d, Final)
        assert isinstance(d, Final)
        self.assertIn("12 days", d.text)

    def test_malformed_tool_name_passed_through(self):
        d = parse_openrouter_decision(CANNED_BAD_TOOL_NAME)
        self.assertIsInstance(d, ToolCall)
        assert isinstance(d, ToolCall)
        self.assertEqual(d.name, "delete_everything")

    def test_malformed_json_args_not_repaired_into_fake_fields(self):
        d = parse_openrouter_decision(CANNED_MALFORMED_ARGS)
        self.assertIsInstance(d, ToolCall)
        assert isinstance(d, ToolCall)
        self.assertEqual(d.name, "get_pto_balance")
        # Empty dict: validate_call will report missing required employee_id.
        self.assertEqual(d.args, {})

    def test_missing_name_not_invented(self):
        d = parse_openrouter_decision(CANNED_MISSING_NAME)
        self.assertIsInstance(d, ToolCall)
        assert isinstance(d, ToolCall)
        self.assertEqual(d.name, "")

    def test_usage_and_cost_estimate(self):
        usage = usage_from_response(CANNED_TOOL_RESPONSE)
        self.assertEqual(usage["prompt_tokens"], 120)
        self.assertEqual(usage["completion_tokens"], 25)
        expected = estimate_cost_usd(120, 25)
        self.assertAlmostEqual(usage["cost_usd"], expected, places=8)
        self.assertGreater(usage["cost_usd"], 0)


class TestOpenRouterModelUnit(unittest.TestCase):
    def test_decide_uses_canned_body_no_network(self):
        m = OpenRouterModel(api_key="test-key-not-real", model="test/model")
        m._post = mock.Mock(return_value=CANNED_TOOL_RESPONSE)  # type: ignore[method-assign]
        decision = m.decide("system", [Turn(role="user", content="days off?")])
        self.assertIsInstance(decision, ToolCall)
        assert isinstance(decision, ToolCall)
        self.assertEqual(decision.name, "get_pto_balance")
        self.assertEqual(m.last_usage["prompt_tokens"], 120)
        self.assertGreater(m.last_usage["cost_usd"], 0)
        m._post.assert_called_once()
        # Key never appears in the JSON body passed to _post
        body = m._post.call_args[0][0]
        self.assertNotIn("test-key-not-real", json.dumps(body))

    def test_decide_without_key_raises_before_network(self):
        m = OpenRouterModel(api_key="")
        with mock.patch.object(m, "_post") as post:
            with self.assertRaises(RuntimeError) as ctx:
                m.decide("sys", [Turn(role="user", content="hi")])
            self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))
            post.assert_not_called()


class TestCliPreflight(unittest.TestCase):
    def test_live_without_key_exits_nonzero(self):
        from app import cli

        env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            # Ensure key is absent
            os.environ.pop("OPENROUTER_API_KEY", None)
            code = cli.main(["--live", "how many days off do I have?"])
        self.assertEqual(code, 1)


@unittest.skipUnless(
    os.environ.get("OPENROUTER_API_KEY"),
    "OPENROUTER_API_KEY not set — skipping live end-to-end test",
)
class TestLiveEndToEnd(unittest.TestCase):
    def test_one_real_decision(self):
        """One real network call. Skips cleanly without a key."""
        from agent import db
        from agent.harness import Agent, AgentConfig

        db.reset()
        model = OpenRouterModel()
        result = Agent(model=model, config=AgentConfig()).run(
            "how many days off do I have?"
        )
        self.assertTrue(result.output)
        self.assertIn(result.trace.stopped_because, {
            "completed",
            "budget:budget.max_steps",
            "budget:budget.max_usd_per_run",
            "budget:budget.max_cost_usd",
            "budget:budget.max_duration_s",
        })
        # Key must not appear in the serialised trace
        blob = json.dumps(result.trace.to_dict(), default=str)
        self.assertNotIn(os.environ["OPENROUTER_API_KEY"], blob)


if __name__ == "__main__":
    unittest.main(verbosity=2)
