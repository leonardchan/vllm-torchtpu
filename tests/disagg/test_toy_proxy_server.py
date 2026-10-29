# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _load_toy_proxy_server():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "disagg"
        / "toy_proxy_server.py"
    )
    spec = importlib.util.spec_from_file_location("toy_proxy_server", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_toy_proxy_server = _load_toy_proxy_server()
_AsciiSafeStreamEncoder = _toy_proxy_server._AsciiSafeStreamEncoder
_acquire_dp_rank = _toy_proxy_server._acquire_dp_rank
_release_dp_rank = _toy_proxy_server._release_dp_rank
_replace_prompt_with_rendered_token_ids = (
    _toy_proxy_server._replace_prompt_with_rendered_token_ids
)


def test_ascii_safe_stream_encoder_handles_split_utf8() -> None:
    encoder = _AsciiSafeStreamEncoder()
    text = "\u044f\U0001f600"
    message = f'data: {{"choices": [{{"text": "{text}"}}]}}\n\n'
    encoded = message.encode("utf-8")
    split_at = encoded.index(text.encode("utf-8")) + 1

    output = encoder.encode(encoded[:split_at])
    output += encoder.encode(encoded[split_at:])
    output += encoder.encode(final=True)

    assert output.isascii()
    assert b"\\u044f" in output
    assert b"\\ud83d\\ude00" in output

    payload = output.decode("ascii").removeprefix("data: ").strip()
    assert json.loads(payload)["choices"][0]["text"] == text


def test_ascii_safe_stream_encoder_leaves_ascii_unchanged() -> None:
    encoder = _AsciiSafeStreamEncoder()
    message = b'data: {"choices": [{"text": "ok"}]}\n\n'

    assert encoder.encode(message) == message
    assert encoder.encode(final=True) == b""


def test_replace_prompt_with_rendered_token_ids_single_prompt() -> None:
    request = {
        "model": "test-model",
        "prompt": "hello",
        "prompt_embeds": b"unused-after-render",
    }
    rendered = [{"token_ids": [1, 2, 3]}]

    converted = _replace_prompt_with_rendered_token_ids(request, rendered)

    assert converted["prompt"] == [1, 2, 3]
    assert "prompt_embeds" not in converted
    assert request["prompt"] == "hello"


def test_replace_prompt_with_rendered_token_ids_multi_prompt() -> None:
    request = {
        "model": "test-model",
        "prompt": ["hello", "world"],
    }
    rendered = [{"token_ids": [1, 2, 3]}, {"token_ids": [4, 5]}]

    converted = _replace_prompt_with_rendered_token_ids(request, rendered)

    assert converted["prompt"] == [[1, 2, 3], [4, 5]]


def test_replace_prompt_with_rendered_token_ids_rejects_bad_response() -> None:
    request = {
        "model": "test-model",
        "prompt": "hello",
    }

    try:
        _replace_prompt_with_rendered_token_ids(request, [{"token_ids": ["1"]}])
    except ValueError as exc:
        assert "token_ids" in str(exc)
    else:
        raise AssertionError("malformed render response should fail")


@pytest.mark.cpu_test
def test_dp_rank_least_inflight_and_round_robin_tie_breaking() -> None:
    client_info = {
        "dp_size": 4,
        "dp_inflight": [0, 0, 0, 0],
        "dp_cursor": 0,
    }

    # Equal inflight counts step round-robin: 0, 1, 2.
    f0, r0 = _acquire_dp_rank(client_info)
    f1, r1 = _acquire_dp_rank(client_info)
    f2, r2 = _acquire_dp_rank(client_info)
    assert (f0["X-data-parallel-rank"], r0) == ("0", 0)
    assert (f1["X-data-parallel-rank"], r1) == ("1", 1)
    assert (f2["X-data-parallel-rank"], r2) == ("2", 2)
    assert client_info["dp_inflight"] == [1, 1, 1, 0]

    # Rank 3 has 0 inflight, so it must be picked next; releasing rank 1 then
    # makes rank 1 the unique minimum on the subsequent acquisition.
    f3, r3 = _acquire_dp_rank(client_info)
    assert (f3["X-data-parallel-rank"], r3) == ("3", 3)
    _release_dp_rank(client_info, r1)
    assert client_info["dp_inflight"] == [1, 0, 1, 1]

    f4, r4 = _acquire_dp_rank(client_info)
    assert (f4["X-data-parallel-rank"], r4) == ("1", 1)

    for rank in (r0, r2, r3, r4):
        _release_dp_rank(client_info, rank)
    assert client_info["dp_inflight"] == [0, 0, 0, 0]


@pytest.mark.cpu_test
def test_dp_rank_default_and_explicit_header_passthrough() -> None:
    single_dp = {"dp_size": 1, "dp_inflight": [0], "dp_cursor": 0}
    assert _acquire_dp_rank(single_dp) == ({}, None)
    assert _acquire_dp_rank(single_dp, "3") == ({"X-data-parallel-rank": "3"}, None)

    multi_dp = {"dp_size": 4, "dp_inflight": [0, 0, 0, 0], "dp_cursor": 0}
    forwarded, tracked = _acquire_dp_rank(multi_dp, "2")
    assert forwarded == {"X-data-parallel-rank": "2"}
    assert tracked == 2
    assert multi_dp["dp_inflight"] == [0, 0, 1, 0]
    _release_dp_rank(multi_dp, tracked)
    assert multi_dp["dp_inflight"] == [0, 0, 0, 0]


@pytest.mark.cpu_test
def test_parse_args_dp_size_flags() -> None:
    args = _toy_proxy_server.parse_args([])
    assert args.prefiller_dp_size == 1
    assert args.decoder_dp_size == 1

    shared = _toy_proxy_server.parse_args(["--dp-size", "4"])
    assert shared.prefiller_dp_size == 4
    assert shared.decoder_dp_size == 4
    assert shared.render_completions is None

    asymmetric = _toy_proxy_server.parse_args(
        ["--dp-size", "4", "--prefiller-dp-size", "1", "--decoder-dp-size", "8"]
    )
    assert asymmetric.prefiller_dp_size == 1
    assert asymmetric.decoder_dp_size == 8

    assert (
        _toy_proxy_server.parse_args(["--no-render-completions"]).render_completions
        is False
    )
    assert (
        _toy_proxy_server.parse_args(["--render-completions"]).render_completions
        is True
    )

    with pytest.raises(ValueError, match="dp_size"):
        _toy_proxy_server.parse_args(["--decoder-dp-size", "0"])


@pytest.mark.cpu_test
def test_handle_completions_routes_asymmetric_prefill_and_decode_dp() -> None:
    prefill_entry = {
        "client": MagicMock(),
        "host": "prefill",
        "port": 8400,
        "id": 0,
        "dp_size": 1,
        "dp_inflight": [0],
        "dp_cursor": 0,
    }
    decode_entry = {
        "client": MagicMock(),
        "host": "decode",
        "port": 9400,
        "id": 0,
        "dp_size": 8,
        "dp_inflight": [1, 1, 0, 1, 1, 1, 1, 1],
        "dp_cursor": 0,
    }

    prefill_resp = MagicMock()
    prefill_resp.json.return_value = {"kv_transfer_params": {"uuid": 77}}
    prefill_resp.raise_for_status = MagicMock()
    prefill_entry["client"].post = AsyncMock(return_value=prefill_resp)

    stream_cm = MagicMock()
    stream_resp = MagicMock()
    stream_resp.status_code = 200
    stream_resp.raise_for_status = MagicMock()

    async def _aiter_bytes():
        # While streaming, rank 2's inflight count must be 1.
        assert decode_entry["dp_inflight"][2] == 1
        yield b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'

    stream_resp.aiter_bytes = _aiter_bytes
    stream_cm.__aenter__ = AsyncMock(return_value=stream_resp)
    stream_cm.__aexit__ = AsyncMock(return_value=False)
    decode_entry["client"].stream = MagicMock(return_value=stream_cm)

    fake_app = SimpleNamespace(
        state=SimpleNamespace(
            prefill_clients=[prefill_entry],
            decode_clients=[decode_entry],
            prefill_iterator=itertools.cycle([0]),
            decode_iterator=itertools.cycle([0]),
        )
    )
    fake_request = SimpleNamespace(
        app=fake_app,
        headers={},
        json=AsyncMock(
            return_value={
                "model": "deepseek",
                "messages": [{"role": "user", "content": "hi"}],
            }
        ),
    )

    async def _run():
        resp = await _toy_proxy_server._handle_completions(
            "/v1/chat/completions", fake_request
        )
        chunks = [chunk async for chunk in resp.body_iterator]
        assert b"hi" in b"".join(chunks)

    asyncio.run(_run())

    prefill_headers = prefill_entry["client"].post.call_args.kwargs["headers"]
    assert "X-data-parallel-rank" not in prefill_headers
    decode_headers = decode_entry["client"].stream.call_args.kwargs["headers"]
    assert decode_headers["X-data-parallel-rank"] == "2"
    assert decode_entry["dp_inflight"][2] == 0


@pytest.mark.cpu_test
def test_dp_rank_prefix_affinity_and_load_bounded_spillover() -> None:
    client_info = {
        "dp_size": 8,
        "dp_inflight": [0] * 8,
        "dp_cursor": 0,
        "dp_cached_ranks": {},
    }
    prefix_a = _toy_proxy_server._extract_prefix_hash(
        {"prompt": [101 + i for i in range(512)]}
    )
    prefix_b = _toy_proxy_server._extract_prefix_hash(
        {"prompt": [901 + i for i in range(512)]}
    )
    assert prefix_a is not None and prefix_b is not None and prefix_a != prefix_b

    # Sequential c=1 requests with prefix_a stay pinned to warmed rank 0.
    for _ in range(4):
        forwarded, rank = _acquire_dp_rank(client_info, prefix_hash=prefix_a)
        assert (forwarded["X-data-parallel-rank"], rank) == ("0", 0)
        _release_dp_rank(client_info, rank)

    # Sequential c=1 requests with a distinct prefix_b pick next idle rank (1).
    forwarded_b, rank_b = _acquire_dp_rank(client_info, prefix_hash=prefix_b)
    assert (forwarded_b["X-data-parallel-rank"], rank_b) == ("1", 1)
    _release_dp_rank(client_info, rank_b)

    # Concurrent c=8 requests with prefix_a spill over across all 8 DP ranks.
    active = [_acquire_dp_rank(client_info, prefix_hash=prefix_a)[1] for _ in range(8)]
    assert sorted(r for r in active if r is not None) == list(range(8))
    assert client_info["dp_inflight"] == [1] * 8
    for r in active:
        _release_dp_rank(client_info, r)
    assert client_info["dp_inflight"] == [0] * 8


@pytest.mark.cpu_test
def test_render_endpoint_bypasses_pretokenized_prompts_and_no_render_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        _toy_proxy_server._render_endpoint_for_api(
            "/v1/completions", {"prompt": "raw text"}
        )
        == "/v1/completions/render"
    )
    assert (
        _toy_proxy_server._render_endpoint_for_api(
            "/v1/completions", {"prompt": [101, 102, 103]}
        )
        is None
    )
    assert (
        _toy_proxy_server._render_endpoint_for_api(
            "/v1/completions", {"prompt": [[101, 102], [201, 202]]}
        )
        is None
    )

    monkeypatch.setenv("PROXY_RENDER_COMPLETIONS", "0")
    assert (
        _toy_proxy_server._render_endpoint_for_api(
            "/v1/completions", {"prompt": "raw text"}
        )
        is None
    )

    monkeypatch.setattr(
        _toy_proxy_server,
        "global_args",
        SimpleNamespace(render_completions=True),
        raising=False,
    )
    assert (
        _toy_proxy_server._render_endpoint_for_api(
            "/v1/completions", {"prompt": "raw text"}
        )
        == "/v1/completions/render"
    )

    monkeypatch.setenv("PROXY_RENDER_COMPLETIONS", "1")
    monkeypatch.setattr(
        _toy_proxy_server,
        "global_args",
        SimpleNamespace(render_completions=False),
        raising=False,
    )
    assert (
        _toy_proxy_server._render_endpoint_for_api(
            "/v1/completions", {"prompt": "raw text"}
        )
        is None
    )


@pytest.mark.cpu_test
def test_one_hop_return_token_ids_and_forward_prompt_token_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROXY_RENDER_COMPLETIONS", "0")
    prefill_entry = {
        "client": MagicMock(),
        "host": "prefill",
        "port": 8400,
        "id": 0,
        "dp_size": 8,
        "dp_inflight": [0] * 8,
        "dp_cursor": 0,
        "dp_cached_ranks": {},
    }
    decode_entry = {
        "client": MagicMock(),
        "host": "decode",
        "port": 9400,
        "id": 0,
        "dp_size": 8,
        "dp_inflight": [0] * 8,
        "dp_cursor": 0,
        "dp_cached_ranks": {},
    }

    prefill_resp = MagicMock()
    prefill_resp.json.return_value = {
        "kv_transfer_params": {"uuid": 99},
        "choices": [{"prompt_token_ids": [11, 22, 33, 44]}],
    }
    prefill_resp.raise_for_status = MagicMock()
    prefill_entry["client"].post = AsyncMock(return_value=prefill_resp)

    stream_cm = MagicMock()
    stream_resp = MagicMock()
    stream_resp.status_code = 200
    stream_resp.raise_for_status = MagicMock()

    async def _aiter_bytes():
        yield b'data: {"choices": [{"text": "ok"}]}\n\n'

    stream_resp.aiter_bytes = _aiter_bytes
    stream_cm.__aenter__ = AsyncMock(return_value=stream_resp)
    stream_cm.__aexit__ = AsyncMock(return_value=False)
    decode_entry["client"].stream = MagicMock(return_value=stream_cm)

    fake_app = SimpleNamespace(
        state=SimpleNamespace(
            prefill_clients=[prefill_entry],
            decode_clients=[decode_entry],
            prefill_iterator=itertools.cycle([0]),
            decode_iterator=itertools.cycle([0]),
        )
    )
    fake_request = SimpleNamespace(
        app=fake_app,
        headers={},
        json=AsyncMock(
            return_value={
                "model": "deepseek",
                "prompt": "raw text prompt",
                "min_tokens": 1024,
                "max_tokens": 1024,
                "stream": True,
            }
        ),
    )

    async def _run():
        resp = await _toy_proxy_server._handle_completions(
            "/v1/completions", fake_request
        )
        chunks = [chunk async for chunk in resp.body_iterator]
        assert b"ok" in b"".join(chunks)

    asyncio.run(_run())

    assert prefill_entry["client"].post.call_count == 1
    prefill_json = prefill_entry["client"].post.call_args.kwargs["json"]
    assert prefill_json["return_token_ids"] is True
    assert prefill_json["min_tokens"] == 1
    decode_json = decode_entry["client"].stream.call_args.kwargs["json"]
    assert decode_json["min_tokens"] == 1024
    assert decode_json["prompt"] == [11, 22, 33, 44]
    assert decode_json["kv_transfer_params"] == {"uuid": 99}
