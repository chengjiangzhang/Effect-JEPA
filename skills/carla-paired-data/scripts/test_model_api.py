#!/usr/bin/env python3
"""Offline transport/validation tests. No paid API calls or CARLA connection."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from model_api import ModelClient, ModelError, SchemaError, check_schema, validate_json


SCHEMA = {"type": "object", "properties": {"choice": {"type": "integer", "minimum": 0, "maximum": 3}},
          "required": ["choice"], "additionalProperties": False}
SECRET = "offline-test-secret-not-a-real-key"


def response_for(provider, text='{"choice":2}'):
    if provider == "openai_chat":
        return {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}
    if provider == "openai_responses":
        return {"status": "completed", "output": [{"type": "reasoning", "summary": []},
                 {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]}
    if provider == "anthropic":
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}
    return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}]}


@contextmanager
def mock_server(responses):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            received.append({"path": self.path, "headers": dict(self.headers), "body": json.loads(raw)})
            status, body, headers = responses[min(len(received) - 1, len(responses) - 1)]
            body = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def client_config(provider="openai_chat", **changes):
    config = {"provider": provider, "model": "test-model", "timeout_s": 1, "max_retries": 0,
              "max_calls": 4, "retry_backoff_s": 0}
    config.update(changes)
    return config


class ModelApiTest(unittest.TestCase):
    def test_provider_endpoints_payload_and_text_extraction(self):
        for provider, suffix in (("openai_chat", "/chat/completions"), ("openai_responses", "/responses"),
                                 ("anthropic", "/messages"), ("gemini", "/models/test-model:generateContent")):
            with self.subTest(provider=provider), mock_server([(200, response_for(provider), {})]) as (base, seen), \
                    patch.dict(os.environ, {"SCENE_TEST_KEY": SECRET}):
                client = ModelClient(client_config(provider, base_url=base, key_env="SCENE_TEST_KEY", json_mode=True))
                result = client.request_json({"operation": "propose", "catalog_candidates": [2]}, schema=SCHEMA)
                self.assertEqual(result.value, {"choice": 2})
                self.assertEqual(seen[0]["path"], "/v1" + suffix)
                headers = {k.lower(): v for k, v in seen[0]["headers"].items()}
                auth_header = "x-goog-api-key" if provider == "gemini" else "authorization"
                self.assertEqual(headers[auth_header], SECRET if provider == "gemini" else "Bearer " + SECRET)
                self.assertNotIn(SECRET, json.dumps(result.provenance))
                self.assertEqual(result.provenance["calls_used"], 1)
                self.assertNotIn(SECRET, json.dumps(seen[0]["body"]))
                body = seen[0]["body"]
                if provider == "openai_chat":
                    self.assertEqual(body["messages"][0]["role"], "system")
                    self.assertEqual(body["response_format"]["type"], "json_object")
                    self.assertIn("max_completion_tokens", body)
                elif provider == "openai_responses":
                    self.assertFalse(body["store"])
                    self.assertIn("max_output_tokens", body)
                elif provider == "anthropic":
                    self.assertIn("anthropic-version", headers)
                    self.assertIn("system", body)
                    self.assertIn("max_tokens", body)
                else:
                    self.assertIn("systemInstruction", body)
                    self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")

    def test_retry_budget_and_nonretryable_auth_error(self):
        with mock_server([(429, {"error": SECRET}, {}), (200, response_for("openai_chat"), {})]) as (base, seen):
            client = ModelClient(client_config(base_url=base, max_retries=2))
            result = client.request_json({}, schema=SCHEMA)
            self.assertEqual(result.provenance["attempts"], 2)
            self.assertEqual(client.calls_used, 2)
            self.assertEqual(len(seen), 2)
        with mock_server([(401, {"error": SECRET}, {})]) as (base, seen):
            client = ModelClient(client_config(base_url=base, max_retries=3))
            with self.assertRaises(ModelError) as ctx:
                client.request_json({}, schema=SCHEMA)
            self.assertIn("401", str(ctx.exception))
            self.assertNotIn(SECRET, str(ctx.exception))
            self.assertEqual(len(seen), 1)

    def test_total_call_budget_includes_retries_and_can_resume(self):
        with mock_server([(503, {"error": "temporary"}, {})]) as (base, seen):
            client = ModelClient(client_config(base_url=base, max_retries=5, max_calls=3, calls_used=1))
            with self.assertRaises(ModelError):
                client.request_json({}, schema=SCHEMA)
            self.assertEqual(len(seen), 2)
            self.assertEqual(client.calls_used, 3)
            with self.assertRaisesRegex(ModelError, "budget exhausted"):
                client.request_json({}, schema=SCHEMA)
            self.assertEqual(len(seen), 2)

    def test_redirect_not_followed(self):
        with mock_server([(302, b"", {"Location": "https://unrelated.invalid/"})]) as (base, seen):
            client = ModelClient(client_config(base_url=base))
            with self.assertRaisesRegex(ModelError, "HTTP 302"):
                client.request_json({}, schema=SCHEMA)
            self.assertEqual(len(seen), 1)

    def test_unknown_fields_wrong_type_duplicate_nonfinite_and_prose_rejected(self):
        outputs = ['{"choice":2,"command":"rm -rf /"}', '{"choice":true}', '{"choice":7}',
                   '{"choice":1,"choice":2}', '{"choice":NaN}', '{"choice":1e999}',
                   '```json\n{"choice":2}\n```', '{"choice":2} extra', '[2]', 'null']
        for output in outputs:
            with self.subTest(output=output), mock_server([(200, response_for("openai_chat", output), {})]) as (base, seen):
                client = ModelClient(client_config(base_url=base, max_retries=4))
                with self.assertRaises(ModelError):
                    client.request_json({}, schema=SCHEMA)
                self.assertEqual(len(seen), 1, "Invalid JSON/schema must not cause paid regeneration loops")

    def test_truncated_or_refused_response_not_accepted_even_if_json_valid(self):
        cases = [("openai_chat", {"choices": [{"finish_reason": "length", "message": {"content": '{"choice":2}'}}]}),
                 ("openai_responses", {"status": "incomplete", "output": []}),
                 ("anthropic", {"stop_reason": "max_tokens", "content": [{"type": "text", "text": '{"choice":2}'}]}),
                 ("gemini", {"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]})]
        for provider, response in cases:
            with self.subTest(provider=provider), mock_server([(200, response, {})]) as (base, _):
                with self.assertRaises(ModelError):
                    ModelClient(client_config(provider, base_url=base)).request_json({}, schema=SCHEMA)

    def test_schema_checked_before_network_and_requires_closed_objects(self):
        client = ModelClient(client_config(base_url="http://127.0.0.1:9/v1"))
        for schema in ({"type": "object"}, {**SCHEMA, "oneOf": []}, {"type": "array", "items": {"type": "integer"}}):
            with self.assertRaises(SchemaError):
                client.request_json({}, schema=schema)
        with self.assertRaises(SchemaError):
            client.request_json({})
        self.assertEqual(client.calls_used, 0)

    def test_credentials_only_in_headers_and_rejected_if_echoed(self):
        with patch.dict(os.environ, {"SCENE_TEST_KEY": SECRET}), \
                mock_server([(200, response_for("openai_chat", json.dumps({"choice": SECRET})), {})]) as (base, _):
            client = ModelClient(client_config(base_url=base, key_env="SCENE_TEST_KEY"))
            with self.assertRaises(ModelError) as ctx:
                client.request_json({}, schema=SCHEMA)
            self.assertNotIn(SECRET, str(ctx.exception))
            with self.assertRaises(ModelError) as ctx:
                client.request_json({"accidental_key": SECRET}, schema=SCHEMA)
            self.assertNotIn(SECRET, str(ctx.exception))
            self.assertEqual(client.calls_used, 1)

    def test_response_byte_limit(self):
        with mock_server([(200, b" " * 2048, {})]) as (base, _):
            # A smaller request fits; the 2 KiB response exceeds the bound.
            client = ModelClient(client_config(base_url=base, max_response_bytes=1024))
            with self.assertRaisesRegex(ModelError, "response exceeds"):
                client.request_json({}, schema=SCHEMA)

    def test_escaped_credential_is_rejected_after_decoding(self):
        text = json.dumps(response_for("openai_chat", json.dumps({"choice": SECRET})))
        text = text.replace(SECRET, "".join("\\u%04x" % ord(c) for c in SECRET))
        with patch.dict(os.environ, {"SCENE_TEST_KEY": SECRET}), mock_server([(200, text.encode(), {})]) as (base, _):
            client = ModelClient(client_config(base_url=base, key_env="SCENE_TEST_KEY"))
            with self.assertRaises(ModelError) as ctx:
                client.request_json({}, schema=SCHEMA)
            self.assertNotIn(SECRET, str(ctx.exception))

    def test_model_version_and_numeric_token_usage_recorded_without_extra_body(self):
        response = response_for("openai_chat")
        response.update(model="served-model-2026-10-28", usage={"input_tokens": 25, "output_tokens": 8,
                        "arbitrary": SECRET, "completion_tokens": True, "total_tokens": -1})
        with mock_server([(200, response, {})]) as (base, _):
            result = ModelClient(client_config(base_url=base)).request_json({}, schema=SCHEMA)
            self.assertEqual(result.provenance["model_reported"], "served-model-2026-10-28")
            self.assertEqual(result.provenance["token_usage"], {"input_tokens": 25, "output_tokens": 8})
            self.assertNotIn(SECRET, json.dumps(result.provenance))

    def test_malformed_model_json_is_repairable_but_malformed_http_envelope_is_not(self):
        with mock_server([(200, response_for("openai_chat", "not JSON"), {})]) as (base, _):
            with self.assertRaises(SchemaError):
                ModelClient(client_config(base_url=base)).request_json({}, schema=SCHEMA)
        command = ModelClient(client_config("command", command=[sys.executable, "-c", "print('not JSON')"]))
        with self.assertRaises(SchemaError):
            command.request_json({}, schema=SCHEMA)
        with mock_server([(200, b"broken server envelope", {})]) as (base, _):
            with self.assertRaises(ModelError) as ctx:
                ModelClient(client_config(base_url=base)).request_json({}, schema=SCHEMA)
            self.assertNotIsInstance(ctx.exception, SchemaError)

    def test_remote_http_credentials_in_url_and_unknown_config_rejected(self):
        invalid = [{"base_url": "http://example.org/v1"}, {"base_url": "https://user:secret@example.org/v1"},
                   {"base_url": "https://example.org/v1?key=secret"}, {"api_key": "secret"},
                   {"max_retries": 100}, {"timeout_s": float("nan")}, {"command": "echo bad", "provider": "command"}]
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises(ModelError):
                ModelClient(client_config(**fields))

    def test_command_json_envelope_and_result(self):
        script = "import json,sys; e=json.load(sys.stdin); assert e['protocol']=='carla-scene-json-v1'; assert e['schema']['additionalProperties'] is False; print(json.dumps({'choice': e['request']['selected']}))"
        client = ModelClient(client_config("command", command=[sys.executable, "-c", script]))
        result = client.request_json({"operation": "repair", "selected": 3, "failure": "initial_overlap"}, schema=SCHEMA)
        self.assertEqual(result.value, {"choice": 3})
        self.assertIsNone(result.provenance["endpoint_host"])
        self.assertEqual(result.provenance["calls_used"], 1)

    def test_command_arguments_are_literal_no_shell_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "must_not_exist"
            literal = f"; touch {marker}"
            client = ModelClient(client_config("command", command=[sys.executable, "-c",
                                 "import json,sys; print(json.dumps({'text':sys.argv[1]}))", literal]))
            schema = {"type": "object", "properties": {"text": {"type": "string"}}, "additionalProperties": False}
            self.assertEqual(client.request_json({}, schema=schema).value["text"], literal)
            self.assertFalse(marker.exists())

    def test_command_error_timeout_and_output_limit(self):
        cases = [("import sys; sys.stderr.write('" + SECRET + "'); sys.exit(7)", {}, "exit code 7"),
                 ("import time; time.sleep(2)", {"timeout_s": .05}, "timed out"),
                 ("print('x'*3000)", {"max_response_bytes": 1024}, "output exceeds")]
        for script, extra, expected in cases:
            with self.subTest(expected=expected):
                client = ModelClient(client_config("command", command=[sys.executable, "-c", script], **extra))
                with self.assertRaises(ModelError) as ctx:
                    client.request_json({}, schema=SCHEMA)
                self.assertIn(expected, str(ctx.exception))
                self.assertNotIn(SECRET, str(ctx.exception))

    def test_custom_validator_is_required_when_schema_absent_and_errors_are_sanitized(self):
        client = ModelClient(client_config("command", command=[sys.executable, "-c", "print('{\"choice\":2}')"]))
        result = client.request_json({}, validator=lambda v: set(v) == {"choice"})
        self.assertEqual(result.value["choice"], 2)
        with self.assertRaisesRegex(SchemaError, "Caller validation rejected"):
            client.request_json({}, validator=lambda v: False)
        def invalid(value):
            raise RuntimeError(SECRET)
        with self.assertRaises(SchemaError) as ctx:
            client.request_json({}, validator=invalid)
        self.assertNotIn(SECRET, str(ctx.exception))

    def test_nested_schema_nullable_array_numeric_and_string_bounds(self):
        schema = {"type": "object", "properties": {
            "poses": {"type": "array", "minItems": 1, "maxItems": 2, "items": {"type": "number", "minimum": -1, "maximum": 1}},
            "role": {"type": ["string", "null"], "pattern": "^npc_[0-9]+$", "maxLength": 10}},
            "required": ["poses", "role"], "additionalProperties": False}
        check_schema(schema)
        validate_json({"poses": [0, .1], "role": None}, schema)
        for value in ({"poses": [], "role": None}, {"poses": [1.1], "role": None},
                      {"poses": [0], "role": "invalid"}, {"poses": [0], "role": "npc_123456789"}):
            with self.assertRaises(SchemaError):
                validate_json(value, schema)

    def test_cli_output_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "spec.json").write_text(json.dumps(client_config("command", command=[sys.executable, "-c", "print('{\"choice\":2}')"])))
            (root / "request.json").write_text('{"operation":"propose"}')
            (root / "schema.json").write_text(json.dumps(SCHEMA))
            argv = [sys.executable, "-B", str(Path(__file__).with_name("model_api.py")),
                    "--config", str(root / "spec.json"), "--request", str(root / "request.json"),
                    "--schema", str(root / "schema.json"), "--out", str(root / "result.json")]
            result = subprocess.run(argv, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((root / "result.json").read_text())["value"], {"choice": 2})
            repeated = subprocess.run(argv, capture_output=True, text=True, timeout=5)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("already exists", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
