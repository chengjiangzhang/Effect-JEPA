#!/usr/bin/env python3
"""Bounded, provider-independent JSON proposal and repair transport.

Usage::

    client = ModelClient({"provider": "openai_chat", "model": "YOUR_MODEL",
                          "base_url": "https://YOUR_HOST/v1",
                          "key_env": "SCENE_MODEL_KEY", "max_calls": 800})
    result = client.request_json({"operation": "propose", "cell": cell,
                                  "catalog_candidates": candidates}, schema=SPEC_SCHEMA)
    spec, audit_record = result.value, result.provenance

``base_url`` is an API root, NOT the complete generation endpoint. The appended
paths are: openai_chat -> /chat/completions; openai_responses -> /responses;
anthropic -> /messages; gemini -> /models/{model}:generateContent. Defaults:
https://api.openai.com/v1, https://api.anthropic.com/v1, and
https://generativelanguage.googleapis.com/v1beta. Gemini model may start models/.
Bearer authentication is used for OpenAI-compatible and Anthropic endpoints;
Gemini uses x-goog-api-key. Anthropic accepts configurable anthropic_version
(default 2026-10-21), anthropic_auth=x-api-key for legacy gateways, and optional
workspace_id. Model names are selected by deployment configuration, not hardcoded.

Command adapter example::

    {"provider": "command", "model": "my-wrapper-v1",
     "command": ["python3", "/opt/model_adapter.py"], "max_calls": 800}

The configured argv is executed with shell=False. It receives ONE UTF-8 JSON
object on stdin: {"protocol":"carla-scene-json-v1", "model":..., "system":...,
"request":..., "schema":...}; stdout must contain exactly ONE result JSON object.
The adapter can call any other provider or local model. It inherits the current
environment. Credentials belong in environment variables, never argv or a spec.
stderr is not copied to dataset logs. Model-generated code is never executed.

request_json requires a caller-supplied, closed-object schema, or a custom
validator. Schema support is deliberately explicit (see _SCHEMA_KEYS); unsupported
keywords fail before a model call. JSON must not contain markdown, duplicate keys,
nonfinite numbers, or trailing prose. HTTP format hints are optional (json_mode);
local validation always applies. A repair is a NEW explicit call with a failure
record; malformed output is never silently accepted or repeatedly regenerated.

max_calls counts transport attempts, including retries. It is per client; the
orchestrator must retain one client and persist its calls_used for resumed jobs.
Only transport failures / HTTP 408,429,5xx retry; limits and backoff are bounded.
CLI --config/--request/--schema can be used without importing this module.

Wire references:
https://developers.openai.com/api/reference/resources/chat
https://developers.openai.com/api/docs/guides/text
https://platform.claude.com/docs/en/api/messages/create
https://platform.claude.com/docs/en/api/overview
https://ai.google.dev/api/generate-content
https://ai.google.dev/gemini-api/docs/api-key
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
from typing import Any, Callable
from urllib import error, parse, request as urlrequest


class ModelError(RuntimeError):
    """Sanitized failure: no response body, prompt, URL query, or credential."""


class SchemaError(ModelError):
    pass


class _TransientError(ModelError):
    pass


@dataclass(frozen=True)
class ModelResult:
    value: dict
    provenance: dict


_SYSTEM = (
    "Produce exactly one JSON object matching the supplied output schema. "
    "Choose only supplied catalog IDs, supported types, and parameter bounds. "
    "Do not invent map geometry or modify the required experimental protocol. "
    "For repair, use the stated failure to revise only permitted fields. "
    "Do not emit code, commands, markdown, extra fields, or explanatory prose."
)
_SCHEMA_KEYS = {
    "type", "properties", "required", "additionalProperties", "items", "enum", "const",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minItems", "maxItems",
    "minLength", "maxLength", "pattern", "description", "title", "$schema",
}
_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ModelError("Input is not finite JSON data") from None


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate key")
        result[key] = value
    return result


def _loads(text: str | bytes) -> Any:
    try:
        value = json.loads(text, object_pairs_hook=_object_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        _json_dumps(value)  # Also reject floating-point overflow such as 1e999.
        return value
    except (UnicodeError, ValueError, TypeError, RecursionError):
        raise ModelError("Response must be a single finite JSON value without duplicate keys") from None


def check_schema(schema: dict, *, root: bool = True) -> None:
    """Reject unsupported schemas rather than silently skip validation rules."""
    if not isinstance(schema, dict) or set(schema) - _SCHEMA_KEYS:
        raise SchemaError("Unsupported schema keyword or malformed schema")
    types = schema.get("type")
    types = types if isinstance(types, list) else [types]
    if not types or any(not isinstance(t, str) or t not in _TYPES for t in types):
        raise SchemaError("Every schema node must declare supported type(s)")
    if root and types != ["object"]:
        raise SchemaError("Root output schema must be an object")
    if "object" in types:
        if schema.get("additionalProperties") is not False:
            raise SchemaError("Object schemas require additionalProperties=false")
        props = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(props, dict) or not isinstance(required, list):
            raise SchemaError("Malformed object schema")
        if any(not isinstance(k, str) or k not in props for k in required):
            raise SchemaError("Required field is not declared in properties")
        for child in props.values():
            check_schema(child, root=False)
    if "array" in types:
        if "items" not in schema:
            raise SchemaError("Array schema requires items")
        check_schema(schema["items"], root=False)
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise SchemaError("enum must be a nonempty list")
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                "minItems", "maxItems", "minLength", "maxLength"):
        if key in schema and (type(schema[key]) not in (int, float) or not math.isfinite(schema[key])):
            raise SchemaError("Invalid schema bound")
    if "pattern" in schema:
        try:
            re.compile(schema["pattern"])
        except (TypeError, re.error):
            raise SchemaError("Invalid schema pattern") from None
    _json_dumps(schema)


def validate_json(value: Any, schema: dict) -> None:
    """Validate data; errors never echo values or unknown response field names."""
    types = schema["type"]
    types = types if isinstance(types, list) else [types]
    actual = ("null" if value is None else "boolean" if type(value) is bool else
              "integer" if type(value) is int else "number" if type(value) is float else
              "string" if isinstance(value, str) else "object" if isinstance(value, dict) else
              "array" if isinstance(value, list) else "invalid")
    if actual not in types and not (actual == "integer" and "number" in types):
        raise SchemaError("Output field has incorrect type")
    if "enum" in schema and not any(type(value) is type(v) and value == v for v in schema["enum"]):
        raise SchemaError("Output field is outside allowed choices")
    if "const" in schema and (type(value) is not type(schema["const"]) or value != schema["const"]):
        raise SchemaError("Output field violates fixed value")
    if actual == "object":
        props = schema.get("properties", {})
        if set(value) - set(props):
            raise SchemaError("Output contains unknown fields")
        if set(schema.get("required", [])) - set(value):
            raise SchemaError("Output is missing required fields")
        for key, item in value.items():
            validate_json(item, props[key])
    elif actual == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", math.inf):
            raise SchemaError("Output array length is out of bounds")
        for item in value:
            validate_json(item, schema["items"])
    elif actual == "string":
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", math.inf):
            raise SchemaError("Output string length is out of bounds")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise SchemaError("Output string does not match required pattern")
    elif actual in ("number", "integer"):
        if not math.isfinite(value):
            raise SchemaError("Output number is not finite")
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            raise SchemaError("Output number is out of bounds")
        if value <= schema.get("exclusiveMinimum", -math.inf) or value >= schema.get("exclusiveMaximum", math.inf):
            raise SchemaError("Output number is outside exclusive bounds")


class _NoRedirect(urlrequest.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Do not forward credentials to a redirected host.


class ModelClient:
    """Serial client; request budgets are not safe for concurrent mutation."""

    def __init__(self, config: dict):
        allowed = {"provider", "model", "base_url", "key_env", "timeout_s", "max_retries",
                   "max_calls", "max_output_tokens", "max_response_bytes", "retry_backoff_s",
                   "json_mode", "chat_token_field", "anthropic_version", "anthropic_auth",
                   "workspace_id", "command", "calls_used"}
        if not isinstance(config, dict) or set(config) - allowed:
            raise ModelError("Invalid model configuration fields")
        self.provider = config.get("provider")
        if self.provider not in {"openai_chat", "openai_responses", "anthropic", "gemini", "command"}:
            raise ModelError("Unsupported provider; use the command adapter for other protocols")
        self.model = config.get("model", "")
        if not isinstance(self.model, str) or not self.model or len(self.model) > 256 or any(ord(c) < 32 for c in self.model):
            raise ModelError("model must be a nonempty model identifier")
        self.config = dict(config)
        self.timeout = self._number("timeout_s", 60, .05, 300)
        self.retries = self._integer("max_retries", 2, 0, 5)
        self.max_calls = self._integer("max_calls", 1000, 1, 100000)
        self.calls_used = self._integer("calls_used", 0, 0, self.max_calls)
        self.output_tokens = self._integer("max_output_tokens", 4096, 1, 100000)
        self.max_bytes = self._integer("max_response_bytes", 1048576, 128, 16777216)
        self.backoff = self._number("retry_backoff_s", 1, 0, 10)
        self.json_mode = config.get("json_mode", False)
        if type(self.json_mode) is not bool:
            raise ModelError("json_mode must be boolean")
        if config.get("chat_token_field", "max_completion_tokens") not in {"max_tokens", "max_completion_tokens"}:
            raise ModelError("Unsupported chat token limit field")
        if config.get("anthropic_auth", "bearer") not in {"bearer", "x-api-key"}:
            raise ModelError("Unsupported Anthropic authentication mode")
        for key in ("anthropic_version", "workspace_id"):
            if key in config and (not isinstance(config[key], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", config[key])):
                raise ModelError("Invalid Anthropic header configuration")
        self.key_env = config.get("key_env")
        if self.key_env is not None and (not isinstance(self.key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.key_env)):
            raise ModelError("key_env must name an environment variable")
        self.key = os.environ.get(self.key_env, "") if self.key_env else ""
        if self.key_env and not self.key:
            raise ModelError("Configured API key environment variable is empty")
        if self.key and not all(32 < ord(c) < 127 for c in self.key):
            raise ModelError("Invalid API key environment value")
        self.base_url = ""
        self.command = None
        if self.provider == "command":
            command = config.get("command")
            if not isinstance(command, list) or not command or not all(isinstance(a, str) and a and "\x00" not in a for a in command):
                raise ModelError("command must be a nonempty argv list")
            if self.key and any(self.key in a for a in command):
                raise ModelError("Credentials cannot be passed in command arguments")
            self.command = list(command)
        else:
            defaults = {"openai_chat": "https://api.openai.com/v1", "openai_responses": "https://api.openai.com/v1",
                        "anthropic": "https://api.anthropic.com/v1", "gemini": "https://generativelanguage.googleapis.com/v1beta"}
            base_url = config.get("base_url", defaults[self.provider])
            try:
                url = parse.urlsplit(base_url)
                _ = url.port
            except (ValueError, TypeError):
                raise ModelError("Invalid API root URL") from None
            if url.scheme not in {"https", "http"} or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ModelError("API root must be HTTP(S) without credentials, query, or fragment")
            if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise ModelError("Remote API roots must use HTTPS")
            self.base_url = base_url.rstrip("/")
        self._opener = urlrequest.build_opener(_NoRedirect())

    def _number(self, name, default, low, high):
        value = self.config.get(name, default)
        if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
            raise ModelError("Model numeric configuration is out of bounds")
        return value

    def _integer(self, name, default, low, high):
        value = self._number(name, default, low, high)
        if type(value) is not int:
            raise ModelError("Model budget configuration must be integer")
        return value

    def _build_http(self, envelope):
        prompt = _json_dumps({"request": envelope["request"], "output_schema": envelope["schema"]})
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.key:
            if self.provider == "gemini":
                headers["x-goog-api-key"] = self.key
            elif self.provider == "anthropic" and self.config.get("anthropic_auth") == "x-api-key":
                headers["x-api-key"] = self.key
            else:
                headers["Authorization"] = "Bearer " + self.key
        if self.provider == "openai_chat":
            suffix = "/chat/completions"
            body = {"model": self.model, "messages": [{"role": "system", "content": _SYSTEM},
                                                       {"role": "user", "content": prompt}], "stream": False,
                    self.config.get("chat_token_field", "max_completion_tokens"): self.output_tokens}
            if self.json_mode:
                body["response_format"] = {"type": "json_object"}
        elif self.provider == "openai_responses":
            suffix = "/responses"
            body = {"model": self.model, "instructions": _SYSTEM, "input": prompt,
                    "max_output_tokens": self.output_tokens, "stream": False, "store": False}
            if self.json_mode:
                body["text"] = {"format": {"type": "json_object"}}
        elif self.provider == "anthropic":
            suffix = "/messages"
            headers["anthropic-version"] = self.config.get("anthropic_version", "2026-10-21")
            if self.config.get("workspace_id"):
                headers["anthropic-workspace-id"] = self.config["workspace_id"]
            body = {"model": self.model, "system": _SYSTEM, "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.output_tokens, "stream": False}
        else:
            model = self.model.removeprefix("models/")
            suffix = "/models/" + parse.quote(model, safe="") + ":generateContent"
            body = {"systemInstruction": {"parts": [{"text": _SYSTEM}]},
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": self.output_tokens}}
            if self.json_mode:
                body["generationConfig"]["responseMimeType"] = "application/json"
        return urlrequest.Request(self.base_url + suffix, data=_json_dumps(body).encode(), headers=headers, method="POST")

    def _http_attempt(self, envelope):
        req = self._build_http(envelope)
        try:
            with self._opener.open(req, timeout=self.timeout) as response:
                body = response.read(self.max_bytes + 1)
        except error.HTTPError as exc:
            status = exc.code
            exc.close()
            kind = _TransientError if status in (408, 429) or 500 <= status < 600 else ModelError
            raise kind(f"Model endpoint returned HTTP {status}") from None
        except (error.URLError, TimeoutError, OSError):
            raise _TransientError("Model endpoint connection or timeout failure") from None
        if len(body) > self.max_bytes:
            raise ModelError("Model response exceeds byte limit")
        if self.key and self.key.encode() in body:
            raise ModelError("Model response contains a credential and was rejected")
        return _loads(body)

    @staticmethod
    def _stop_command(process):
        if process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            process.wait()

    def _command_attempt(self, envelope):
        # Files prevent unbounded communicate() buffers; poll enforces output cap.
        with tempfile.TemporaryFile() as stdin, tempfile.TemporaryFile() as stdout:
            stdin.write(_json_dumps(envelope).encode())
            stdin.seek(0)
            try:
                process = subprocess.Popen(self.command, stdin=stdin, stdout=stdout, stderr=subprocess.DEVNULL,
                                           shell=False, start_new_session=(os.name == "posix"))
            except OSError:
                raise ModelError("Model command could not be started") from None
            deadline = time.monotonic() + self.timeout
            try:
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        raise _TransientError("Model command timed out")
                    if os.fstat(stdout.fileno()).st_size > self.max_bytes:
                        raise ModelError("Model command output exceeds byte limit")
                    time.sleep(.01)
                if process.returncode:
                    raise ModelError(f"Model command failed with exit code {process.returncode}")
                stdout.seek(0)
                body = stdout.read(self.max_bytes + 1)
                if len(body) > self.max_bytes:
                    raise ModelError("Model command output exceeds byte limit")
            finally:
                self._stop_command(process)
        if self.key and self.key.encode() in body:
            raise ModelError("Model response contains a credential and was rejected")
        try:
            return _loads(body)
        except ModelError:
            raise SchemaError("Model command output is not valid JSON") from None

    def _extract_text(self, response):
        try:
            if self.provider == "openai_chat":
                choice = response["choices"][0]
                if choice.get("finish_reason") != "stop" or choice["message"].get("refusal"):
                    raise ModelError("Model response was refused, truncated, or requested a tool")
                text = choice["message"]["content"]
            elif self.provider == "openai_responses":
                if response.get("status") != "completed":
                    raise ModelError("Model response did not complete")
                parts = [p for item in response["output"] if item.get("type") == "message"
                         for p in item.get("content", [])]
                if any(p.get("type") == "refusal" for p in parts):
                    raise ModelError("Model response was refused")
                text = "".join(p["text"] for p in parts if p.get("type") == "output_text")
            elif self.provider == "anthropic":
                if response.get("stop_reason") not in {"end_turn", "stop_sequence"}:
                    raise ModelError("Model response did not complete normally")
                text = "".join(p["text"] for p in response["content"] if p.get("type") == "text")
            else:
                candidate = response["candidates"][0]
                if candidate.get("finishReason") != "STOP":
                    raise ModelError("Model response did not complete normally")
                text = "".join(p["text"] for p in candidate["content"]["parts"]
                               if "text" in p and not p.get("thought"))
            if not isinstance(text, str) or not text.strip():
                raise ModelError("Model endpoint returned no text")
            return text
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ModelError("Model endpoint response has an unexpected structure") from None

    def request_json(self, request: dict, *, schema: dict | None = None,
                     validator: Callable[[dict], Any] | None = None) -> ModelResult:
        if not isinstance(request, dict):
            raise ModelError("Model request must be a JSON object")
        if schema is None and validator is None:
            raise SchemaError("A closed output schema or caller validator is required")
        if schema is not None:
            check_schema(schema)
        envelope = {"protocol": "carla-scene-json-v1", "model": self.model,
                    "system": _SYSTEM, "request": request, "schema": schema}
        encoded = _json_dumps(envelope).encode()
        if len(encoded) > self.max_bytes:
            raise ModelError("Model request exceeds byte limit")
        if self.key and self.key.encode() in encoded:
            raise ModelError("Credentials cannot be included in a model prompt")
        attempts = 0
        started = time.monotonic()
        while True:
            if self.calls_used >= self.max_calls:
                raise ModelError("Model transport call budget exhausted")
            self.calls_used += 1
            attempts += 1
            try:
                response = self._command_attempt(envelope) if self.provider == "command" else self._http_attempt(envelope)
                break
            except _TransientError:
                if attempts > self.retries or self.calls_used >= self.max_calls:
                    raise
                time.sleep(min(self.backoff * 2 ** (attempts - 1), 10))
        # Check after decoding as well: a response may JSON-escape the credential.
        if self.key and self.key in _json_dumps(response):
            raise ModelError("Model response contains a credential and was rejected")
        if self.provider == "command":
            value = response
        else:
            text = self._extract_text(response)
            try:
                value = _loads(text)
            except ModelError:
                raise SchemaError("Model output text is not valid JSON") from None
        if self.key and self.key in _json_dumps(value):
            raise ModelError("Model response contains a credential and was rejected")
        if not isinstance(value, dict):
            raise SchemaError("Model result must be a JSON object")
        if schema is not None:
            validate_json(value, schema)
        if validator is not None:
            try:
                accepted = validator(value)
            except Exception:
                raise SchemaError("Caller validation rejected the model result") from None
            if accepted is False:
                raise SchemaError("Caller validation rejected the model result")
        provenance = {"provider": self.provider, "model": self.model,
                      "endpoint_host": parse.urlsplit(self.base_url).hostname if self.base_url else None,
                      "attempts": attempts, "calls_used": self.calls_used,
                      "elapsed_s": round(time.monotonic() - started, 3),
                      "request_sha256": hashlib.sha256(encoded).hexdigest(),
                      "response_sha256": hashlib.sha256(_json_dumps(value).encode()).hexdigest()}
        if self.provider != "command" and isinstance(response, dict):
            reported = response.get("model", response.get("modelVersion"))
            if isinstance(reported, str) and re.fullmatch(r"[A-Za-z0-9_.:/+-]{1,256}", reported):
                provenance["model_reported"] = reported
            usage = response.get("usage", response.get("usageMetadata", {}))
            if isinstance(usage, dict):
                token_keys = {"input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens",
                              "promptTokenCount", "candidatesTokenCount", "totalTokenCount", "thoughtsTokenCount"}
                provenance["token_usage"] = {k: v for k, v in usage.items()
                                             if k in token_keys and type(v) is int and 0 <= v <= 10000000000}
        return ModelResult(value, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    try:
        client = ModelClient(_loads(args.config.read_bytes()))
        result = client.request_json(_loads(args.request.read_bytes()), schema=_loads(args.schema.read_bytes()))
        with args.out.open("x", encoding="utf-8") as handle:
            json.dump({"value": result.value, "provenance": result.provenance}, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except ModelError as exc:
        parser.exit(2, f"model_api: {exc}\n")


if __name__ == "__main__":
    main()
