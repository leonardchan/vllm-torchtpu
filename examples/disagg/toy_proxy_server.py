# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import codecs
import itertools
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class _ProxyLatencyTracker:
    """Small running latency summary for the proxy path."""

    def __init__(self, log_interval_s: float):
        self._log_interval_s = log_interval_s
        self._lock = threading.Lock()
        self._stats: dict[str, list[float]] = {}
        self._next_log = time.monotonic() + log_interval_s

    def record(self, phase: str, ms: float) -> None:
        if self._log_interval_s <= 0:
            return
        msg: str | None = None
        with self._lock:
            stat = self._stats.setdefault(phase, [0.0, 0.0, ms, ms])
            stat[0] += 1.0
            stat[1] += ms
            stat[2] = min(stat[2], ms)
            stat[3] = max(stat[3], ms)

            now = time.monotonic()
            if now >= self._next_log:
                self._next_log = now + self._log_interval_s
                parts = []
                for key in sorted(self._stats):
                    n, total, min_ms, max_ms = self._stats[key]
                    parts.append(
                        f"{key}: n={int(n)} avg={total / n:.2f}ms "
                        f"min={min_ms:.2f}ms max={max_ms:.2f}ms"
                    )
                msg = f"PERF PROXY latency summary | {' | '.join(parts)}"
        if msg is not None:
            print(msg, flush=True)
            logger.info(msg)


def _proxy_latency_interval() -> float:
    val = os.getenv("PROXY_LATENCY_LOG_INTERVAL", "30")
    try:
        return float(val)
    except ValueError:
        return 30.0


_PROXY_LATENCY = _ProxyLatencyTracker(_proxy_latency_interval())


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# Retries here are deliberately limited to failures that provably occur before
# any request byte is written to the socket, so the backend holds no state for
# the request and a replay is indistinguishable from a first attempt:
#   ConnectError   - the TCP/TLS connection was never established.
#   ConnectTimeout - timed out establishing that connection.
#   PoolTimeout    - timed out waiting for a free slot in the local pool, so we
#                    never even got a socket to write to.
# ConnectTimeout and PoolTimeout subclass TimeoutException, *not* ConnectError,
# so they have to be named explicitly.
#
# Anything that fails *after* the request hits the wire (ReadError, WriteError,
# ReadTimeout, 5xx) must NOT be retried from this proxy: vLLM derives its
# internal request id straight from the X-Request-Id header we send, so a
# replay is seen by the engine as the same request arriving twice. That trips
# `assert existing.streaming_queue is not None, "duplicate request id"` in
# Scheduler.add_request, which raises inside the engine loop and takes down the
# whole EngineCore rather than failing the single request. On the decode hop it
# additionally collides with the Raiden Stage-3 KV registration keyed by the
# same id. Recovering those cases requires re-running prefill under a fresh
# request id to mint a new KV uuid, which has to happen at the request-flow
# level (see _handle_completions), not inside a per-hop wrapper.
_PRE_SEND_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
_MAX_ATTEMPTS = max(1, _env_int("PROXY_MAX_ATTEMPTS", 5))
_RETRY_BASE_DELAY_S = _env_float("PROXY_RETRY_BASE_DELAY_S", 0.5)


def _log_unsafe_transport_failure(
    phase: str, request_id: str, exc: Exception, detail: str = ""
) -> None:
    """Flag a failure that cannot be retried safely at this layer."""
    msg = (
        f"[UNSAFE-{phase}] req_id={request_id} failed after the request "
        f"was already sent: {exc!r}. Not retrying: the server may have "
        f"admitted this request, and replaying it with the same "
        f"X-Request-Id would collide with the in-flight vLLM request id "
        f"(and the Raiden KV registration on the decode hop). Recovery "
        f"needs a fresh request id + new KV uuid and must be handled at "
        f"the request-flow level, not in this per-hop retry wrapper."
    )
    if detail:
        msg = f"{msg} {detail}"
    print(msg, flush=True)
    logger.error(msg)


async def _backoff_before_retry(
    attempt: int, phase: str, request_id: str, exc: Exception
) -> None:
    delay = _RETRY_BASE_DELAY_S * (2**attempt)
    msg = (
        f"[RETRY-{phase}] attempt {attempt + 1}/{_MAX_ATTEMPTS} "
        f"failed for req_id={request_id}: {exc!r}, "
        f"retrying in {delay:.1f}s"
    )
    print(msg, flush=True)
    logger.warning(msg)
    await asyncio.sleep(delay)


async def _post_with_retries(
    client_info: dict,
    endpoint: str,
    req_data: dict,
    headers: dict,
    request_id: str,
    phase: str,
):
    """POST to a backend, retrying only connection-establishment failures."""
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = await client_info["client"].post(
                endpoint, json=req_data, headers=headers
            )
            response.raise_for_status()
            return response
        except _PRE_SEND_ERRORS as e:
            if attempt == _MAX_ATTEMPTS - 1:
                raise
            await _backoff_before_retry(attempt, phase, request_id, e)
        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                _log_unsafe_transport_failure(phase, request_id, e)
            raise
        except httpx.TransportError as e:
            _log_unsafe_transport_failure(phase, request_id, e)
            raise


def _is_token_id_prompt(prompt: Any) -> bool:
    """Return True when ``prompt`` is already token IDs (int or list of int lists)."""
    if not isinstance(prompt, list) or not prompt:
        return False
    if all(type(token_id) is int and token_id >= 0 for token_id in prompt):
        return True
    return all(
        isinstance(item, list)
        and bool(item)
        and all(type(token_id) is int and token_id >= 0 for token_id in item)
        for item in prompt
    )


def _is_raw_text_prompt(prompt: Any) -> bool:
    """Return True when ``prompt`` is raw text (str or non-empty list[str])."""
    if isinstance(prompt, str):
        return True
    return (
        isinstance(prompt, list)
        and bool(prompt)
        and all(isinstance(item, str) for item in prompt)
    )


def _render_completions_enabled() -> bool:
    """Return True when completion rendering is enabled via CLI or env (default 1).

    Checks ``global_args.render_completions`` (set by ``--render-completions`` /
    ``--no-render-completions``) first, then falls back to
    ``PROXY_RENDER_COMPLETIONS`` (default ``"1"``).
    """
    cli_pref = getattr(globals().get("global_args"), "render_completions", None)
    if cli_pref is not None:
        return bool(cli_pref)
    env_val = os.getenv("PROXY_RENDER_COMPLETIONS", "1").strip().lower()
    return env_val not in ("0", "false", "no", "off")


def _render_endpoint_for_api(
    api: str, req_data: dict[str, Any] | None = None
) -> str | None:
    if api != "/v1/completions":
        return None
    if not _render_completions_enabled():
        return None
    if isinstance(req_data, dict) and _is_token_id_prompt(req_data.get("prompt")):
        return None
    return "/v1/completions/render"


def _forward_prompt_token_ids_if_present(
    api: str, req_data: dict[str, Any], prefill_json: Any
) -> None:
    """Replace raw text prompt with prefill's ``prompt_token_ids`` when present."""
    if (
        api not in ("/v1/completions", "/completions")
        or not _is_raw_text_prompt(req_data.get("prompt"))
        or not isinstance(prefill_json, dict)
    ):
        return
    prompt_token_ids = prefill_json.get("prompt_token_ids")
    if not prompt_token_ids:
        choices = prefill_json.get("choices")
        if isinstance(choices, list) and choices:
            if isinstance(req_data.get("prompt"), list) and len(choices) > 1:
                multi = [
                    c.get("prompt_token_ids")
                    for c in choices
                    if isinstance(c, dict)
                    and _is_token_id_prompt(c.get("prompt_token_ids"))
                ]
                if len(multi) == len(choices):
                    prompt_token_ids = multi
            elif isinstance(choices[0], dict):
                prompt_token_ids = choices[0].get("prompt_token_ids")
    if not prompt_token_ids:
        kv_params = prefill_json.get("kv_transfer_params")
        if isinstance(kv_params, dict):
            prompt_token_ids = kv_params.get("prompt_token_ids")
    if _is_token_id_prompt(prompt_token_ids):
        req_data["prompt"] = prompt_token_ids
        req_data.pop("prompt_embeds", None)


def _token_ids_from_completion_render(rendered: Any) -> list[int] | list[list[int]]:
    if not isinstance(rendered, list) or not rendered:
        raise ValueError("Completion render response must be a non-empty list")

    prompts: list[list[int]] = []
    for item in rendered:
        if not isinstance(item, dict):
            raise ValueError("Completion render item must be an object")
        token_ids = item.get("token_ids")
        if not isinstance(token_ids, list) or not all(
            type(token_id) is int and token_id >= 0 for token_id in token_ids
        ):
            raise ValueError("Completion render item must contain token_ids")
        prompts.append(token_ids)

    return prompts[0] if len(prompts) == 1 else prompts


def _replace_prompt_with_rendered_token_ids(req_data: dict, rendered: Any) -> dict:
    req_data = req_data.copy()
    req_data["prompt"] = _token_ids_from_completion_render(rendered)
    req_data.pop("prompt_embeds", None)
    return req_data


def _json_escape_non_ascii(text: str) -> bytes:
    """Encode text as ASCII bytes while preserving JSON string semantics."""
    if text.isascii():
        return text.encode("ascii")

    parts = []
    for char in text:
        codepoint = ord(char)
        if codepoint < 128:
            parts.append(char)
        elif codepoint <= 0xFFFF:
            parts.append(f"\\u{codepoint:04x}")
        else:
            codepoint -= 0x10000
            high_surrogate = 0xD800 + (codepoint >> 10)
            low_surrogate = 0xDC00 + (codepoint & 0x3FF)
            parts.append(f"\\u{high_surrogate:04x}\\u{low_surrogate:04x}")
    return "".join(parts).encode("ascii")


class _AsciiSafeStreamEncoder:
    """Converts UTF-8 stream chunks to ASCII-only JSON-compatible chunks."""

    def __init__(self):
        self._decoder = codecs.getincrementaldecoder("utf-8")()

    def encode(self, chunk: bytes = b"", *, final: bool = False) -> bytes:
        text = self._decoder.decode(chunk, final=final)
        if not text:
            return b""
        return _json_escape_non_ascii(text)


_MAX_PREFIX_CACHE_ENTRIES = 4096


def _extract_prefix_hash(req_data: dict[str, Any] | None) -> int | None:
    """Compute a lightweight hash of the request prefix for DP cache affinity."""
    if not isinstance(req_data, dict):
        return None
    prompt = req_data.get("prompt")
    if isinstance(prompt, list) and prompt:
        if type(prompt[0]) is int:
            return hash(tuple(prompt[:256]))
        if isinstance(prompt[0], list) and prompt[0] and type(prompt[0][0]) is int:
            return hash(tuple(prompt[0][:256]))
        if isinstance(prompt[0], str) and prompt[0]:
            return hash(prompt[0][:1024])
    elif isinstance(prompt, str) and prompt:
        return hash(prompt[:1024])

    messages = req_data.get("messages")
    if isinstance(messages, list) and messages:
        parts: list[str] = []
        total_chars = 0
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str) and content:
                    take = content[: 1024 - total_chars]
                    parts.append(take)
                    total_chars += len(take)
                    if total_chars >= 1024:
                        break
        if parts:
            return hash("".join(parts))
    return None


def _acquire_dp_rank(
    client_info: dict[str, Any],
    requested_dp_rank: str | None = None,
    prefix_hash: int | None = None,
) -> tuple[dict[str, str], int | None]:
    """Select a DP rank using prefix affinity + least-inflight with spillover.

    When ``requested_dp_rank`` is explicitly provided by the caller, preserve
    it and track its inflight count when it maps to a valid rank index. When
    ``dp_size <= 1`` and no rank was requested, return ``({}, None)``.
    When ``prefix_hash`` is provided, prefer cached DP ranks whose inflight load
    is below ``(sum(inflight) // dp_size) + 1``, spilling over to least-inflight
    with round-robin tie-breaking across all ranks as concurrency grows.
    """
    dp_size = max(1, int(client_info.get("dp_size", 1) or 1))
    inflight = client_info.get("dp_inflight")
    if not isinstance(inflight, list) or len(inflight) < dp_size:
        inflight = [0] * dp_size
        client_info["dp_inflight"] = inflight

    if requested_dp_rank is not None:
        forwarded = {"X-data-parallel-rank": requested_dp_rank}
        try:
            rank_idx = int(requested_dp_rank)
        except ValueError:
            return forwarded, None
        if 0 <= rank_idx < len(inflight):
            inflight[rank_idx] += 1
            return forwarded, rank_idx
        return forwarded, None

    if dp_size <= 1:
        return {}, None

    cursor = int(client_info.get("dp_cursor", 0) or 0) % dp_size
    all_candidates = [(cursor + offset) % dp_size for offset in range(dp_size)]
    candidates = all_candidates
    cached_map: dict[int, set[int]] | None = None
    if prefix_hash is not None:
        raw_map = client_info.get("dp_cached_ranks")
        if not isinstance(raw_map, dict):
            raw_map = {}
            client_info["dp_cached_ranks"] = raw_map
        cached_map = raw_map
        cached = cached_map.get(prefix_hash)
        if cached:
            max_load = (sum(inflight) // dp_size) + 1
            aff_candidates = [
                r for r in all_candidates if r in cached and inflight[r] < max_load
            ]
            if aff_candidates:
                candidates = aff_candidates

    best_rank = min(candidates, key=lambda r: inflight[r])
    client_info["dp_cursor"] = (best_rank + 1) % dp_size
    inflight[best_rank] += 1
    if prefix_hash is not None and cached_map is not None:
        if (
            len(cached_map) >= _MAX_PREFIX_CACHE_ENTRIES
            and prefix_hash not in cached_map
        ):
            cached_map.pop(next(iter(cached_map)), None)
        cached_map.setdefault(prefix_hash, set()).add(best_rank)
    return {"X-data-parallel-rank": str(best_rank)}, best_rank


def _release_dp_rank(client_info: dict[str, Any], tracked_rank: int | None) -> None:
    """Decrement the inflight counter for ``tracked_rank`` once a hop finishes."""
    if tracked_rank is None:
        return
    inflight = client_info.get("dp_inflight")
    if (
        isinstance(inflight, list)
        and 0 <= tracked_rank < len(inflight)
        and inflight[tracked_rank] > 0
    ):
        inflight[tracked_rank] -= 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize client pools for prefiller and decoder services
    app.state.prefill_clients = []
    app.state.decode_clients = []

    # keepalive_expiry must stay *below* the backend's keep-alive timeout,
    # otherwise the pool hands out sockets the server has already reaped and
    # the next request dies with ReadError/RemoteProtocolError -- which we
    # cannot retry safely (see _PRE_SEND_ERRORS above). vLLM runs uvicorn with
    # timeout_keep_alive=VLLM_HTTP_TIMEOUT_KEEP_ALIVE, default 5s, and httpx's
    # own default expiry is also 5.0, so out of the box the two race exactly.
    # 4.0 buys a second of margin; raise VLLM_HTTP_TIMEOUT_KEEP_ALIVE on the
    # backends if you want longer-lived connections.
    keepalive_expiry = _env_float("PROXY_KEEPALIVE_EXPIRY_S", 4.0)
    limits = httpx.Limits(
        max_connections=None,
        max_keepalive_connections=None,
        keepalive_expiry=keepalive_expiry,
    )
    prefill_dp_size = max(1, int(getattr(global_args, "prefiller_dp_size", 1)))
    decode_dp_size = max(1, int(getattr(global_args, "decoder_dp_size", 1)))

    # Create prefill clients
    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f"http://{host}:{port}"
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None, base_url=prefiller_base_url, limits=limits
                ),
                "host": host,
                "port": port,
                "id": i,
                "dp_size": prefill_dp_size,
                "dp_inflight": [0] * prefill_dp_size,
                "dp_cursor": 0,
                "dp_cached_ranks": {},
            }
        )

    # Create decode clients
    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f"http://{host}:{port}"
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None, base_url=decoder_base_url, limits=limits
                ),
                "host": host,
                "port": port,
                "id": i,
                "dp_size": decode_dp_size,
                "dp_inflight": [0] * decode_dp_size,
                "dp_cursor": 0,
                "dp_cached_ranks": {},
            }
        )

    # Initialize round-robin iterators
    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))

    print(
        f"Initialized {len(app.state.prefill_clients)} prefill clients "
        f"(dp_size={prefill_dp_size}) and {len(app.state.decode_clients)} "
        f"decode clients (dp_size={decode_dp_size})."
    )

    yield

    # Shutdown: Close all clients
    for client_info in app.state.prefill_clients:
        await client_info["client"].aclose()

    for client_info in app.state.decode_clients:
        await client_info["client"].aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")

    # For prefiller instances
    parser.add_argument(
        "--prefiller-hosts",
        "--prefiller-host",
        type=str,
        nargs="+",
        default=["localhost"],
    )
    parser.add_argument(
        "--prefiller-ports", "--prefiller-port", type=int, nargs="+", default=[8400]
    )

    # For decoder instances
    parser.add_argument(
        "--decoder-hosts", "--decoder-host", type=str, nargs="+", default=["localhost"]
    )
    parser.add_argument(
        "--decoder-ports", "--decoder-port", type=int, nargs="+", default=[9400]
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=None,
        help=(
            "Data-parallel size per backend replica; sets both "
            "--prefiller-dp-size and --decoder-dp-size unless overridden."
        ),
    )
    parser.add_argument(
        "--prefiller-dp-size",
        type=int,
        default=None,
        help=(
            "Data-parallel size per prefill replica for least-inflight "
            "X-data-parallel-rank routing."
        ),
    )
    parser.add_argument(
        "--decoder-dp-size",
        type=int,
        default=None,
        help=(
            "Data-parallel size per decode replica for least-inflight "
            "X-data-parallel-rank routing."
        ),
    )
    parser.add_argument(
        "--render-completions",
        dest="render_completions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable (--no-render-completions) the separate "
            "/v1/completions/render RPC; when disabled, rely on 1-hop "
            "prefill return_token_ids forwarding (default: "
            "PROXY_RENDER_COMPLETIONS=1)."
        ),
    )

    args = parser.parse_args(argv)

    # Validate and pair hosts with ports
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports"
        )

    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")

    default_dp_size = _env_int("PROXY_DP_SIZE", 1)
    default_prefill_dp = _env_int("PROXY_PREFILLER_DP_SIZE", default_dp_size)
    default_decode_dp = _env_int("PROXY_DECODER_DP_SIZE", default_dp_size)
    if args.dp_size is not None:
        if args.dp_size < 1:
            raise ValueError("dp_size must be >= 1")
        default_prefill_dp = args.dp_size
        default_decode_dp = args.dp_size

    args.prefiller_dp_size = (
        args.prefiller_dp_size
        if args.prefiller_dp_size is not None
        else default_prefill_dp
    )
    args.decoder_dp_size = (
        args.decoder_dp_size if args.decoder_dp_size is not None else default_decode_dp
    )
    if args.prefiller_dp_size < 1:
        raise ValueError("prefiller_dp_size must be >= 1")
    if args.decoder_dp_size < 1:
        raise ValueError("decoder_dp_size must be >= 1")

    # Create tuples of (host, port) for each service type
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))

    return args


def get_next_client(app, service_type: str):
    """
    Get the next client in round-robin fashion.

    Args:
        app: The FastAPI app instance
        service_type: Either 'prefill' or 'decode'

    Returns:
        The next client to use
    """
    if service_type == "prefill":
        client_idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[client_idx]
    elif service_type == "decode":
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")


async def send_request_to_prefill(
    client_info: dict,
    endpoint: str,
    req_data: dict,
    request_id: str,
    forwarded: dict | None = None,
):
    """
    Send a request to a service using a client from the pool.
    """
    req_data = req_data.copy()
    # Must overwrite these for prefill workers.
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "min_tokens" in req_data and req_data["min_tokens"] is not None:
        req_data["min_tokens"] = min(int(req_data["min_tokens"]), 1)
    if "stream_options" in req_data:
        del req_data["stream_options"]
    if endpoint in ("/v1/completions", "/completions") and _is_raw_text_prompt(
        req_data.get("prompt")
    ):
        req_data.setdefault("return_token_ids", True)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
        **(forwarded or {}),
    }

    return await _post_with_retries(
        client_info, endpoint, req_data, headers, request_id, phase="PREFILL"
    )


async def render_completion_prompt(
    client_info: dict,
    endpoint: str,
    req_data: dict,
    request_id: str,
    forwarded: dict | None = None,
):
    """
    Render/tokenize a completion request once so downstream P/D requests can
    use prompt token IDs instead of retokenizing raw text independently.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
        **(forwarded or {}),
    }

    return await _post_with_retries(
        client_info, endpoint, req_data, headers, request_id, phase="RENDER"
    )


async def stream_from_decode(
    client_info: dict,
    endpoint: str,
    req_data: dict,
    request_id: str,
    forwarded: dict | None = None,
):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
        **(forwarded or {}),
    }

    for attempt in range(_MAX_ATTEMPTS):
        yielded_any = False
        try:
            async with client_info["client"].stream(
                "POST", endpoint, json=req_data, headers=headers
            ) as response:
                if response.status_code >= 400:
                    # Streaming responses are not read on entry, so pull the
                    # body first or the raised error carries no vLLM detail.
                    await response.aread()
                response.raise_for_status()
                encoder = _AsciiSafeStreamEncoder()
                async for chunk in response.aiter_bytes():
                    chunk = encoder.encode(chunk)
                    if chunk:
                        yielded_any = True
                        yield chunk
                chunk = encoder.encode(final=True)
                if chunk:
                    yielded_any = True
                    yield chunk
            return
        except _PRE_SEND_ERRORS as e:
            # Never reached the decode server, so the KV handle minted by
            # prefill is still parked and unclaimed: safe to re-send as-is.
            if attempt == _MAX_ATTEMPTS - 1:
                raise
            await _backoff_before_retry(attempt, "DECODE", request_id, e)
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            if yielded_any:
                detail = (
                    "Stream had already emitted bytes to the client, "
                    "so the response is truncated."
                )
            else:
                detail = (
                    "Failed before the first byte; the client will "
                    "still see HTTP 200 followed by an empty/short "
                    "body because StreamingResponse has already "
                    "flushed headers."
                )
            _log_unsafe_transport_failure("DECODE", request_id, e, detail)
            raise


async def _handle_completions(api: str, request: Request):
    try:
        t_request_recv_perf = time.perf_counter()
        req_data = await request.json()
        request_id = str(uuid.uuid4())
        # vLLM sends a request carrying X-data-parallel-rank to that DP rank of
        # the engine. Passed to both hops when explicitly set by the caller, or
        # selected per hop via prefix-aware least-inflight routing when pool
        # dp_size > 1.
        requested_dp_rank = request.headers.get(
            "X-data-parallel-rank"
        ) or request.headers.get("x-data-parallel-rank")
        prefix_hash = _extract_prefix_hash(req_data)
        print(
            f"PERF PROXY req_received req_id={request_id} ts={time.time():.6f}",
            flush=True,
        )

        # Get the next prefill client in round-robin fashion
        prefill_client_info = get_next_client(request.app, "prefill")
        prefill_forwarded, prefill_tracked_rank = _acquire_dp_rank(
            prefill_client_info, requested_dp_rank, prefix_hash=prefix_hash
        )
        try:
            render_endpoint = _render_endpoint_for_api(api, req_data)
            if render_endpoint is not None:
                try:
                    render_response = await render_completion_prompt(
                        prefill_client_info,
                        render_endpoint,
                        req_data,
                        request_id,
                        prefill_forwarded,
                    )
                    req_data = _replace_prompt_with_rendered_token_ids(
                        req_data, render_response.json()
                    )
                except httpx.HTTPStatusError as e:
                    if e.response.status_code not in (400, 404, 405, 422, 501):
                        raise
                    logger.warning(
                        "Render endpoint %s returned HTTP %s for req_id=%s; "
                        "falling back to inline return_token_ids forwarding.",
                        render_endpoint,
                        e.response.status_code,
                        request_id,
                    )

            # Send request to prefill service
            t_prefill_send = time.time()
            t_prefill_send_perf = time.perf_counter()
            print(
                f"PERF PROXY prefill_send req_id={request_id} ts={t_prefill_send:.6f}",
                flush=True,
            )
            response = await send_request_to_prefill(
                prefill_client_info, api, req_data, request_id, prefill_forwarded
            )
        finally:
            _release_dp_rank(prefill_client_info, prefill_tracked_rank)

        t_prefill_recv = time.time()
        t_prefill_recv_perf = time.perf_counter()
        prefill_ms = (t_prefill_recv_perf - t_prefill_send_perf) * 1000.0
        _PROXY_LATENCY.record("prefill", prefill_ms)
        _PROXY_LATENCY.record(
            "prefill_from_req", (t_prefill_recv_perf - t_request_recv_perf) * 1000.0
        )

        # Extract the needed fields
        response_json = response.json()
        kv_transfer_params = response_json.get("kv_transfer_params", {})
        kv_uuid = kv_transfer_params.get("uuid") if kv_transfer_params else None
        kv_uuid_log = f" kv_uuid={kv_uuid}" if kv_uuid is not None else ""
        print(
            f"PERF PROXY prefill_recv req_id={request_id} ts={t_prefill_recv:.6f} "
            f"dur_ms={prefill_ms:.2f}{kv_uuid_log}",
            flush=True,
        )

        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params
        _forward_prompt_token_ids_if_present(api, req_data, response_json)

        # Get the next decode client in round-robin fashion
        decode_client_info = get_next_client(request.app, "decode")
        decode_forwarded, decode_tracked_rank = _acquire_dp_rank(
            decode_client_info, requested_dp_rank, prefix_hash=prefix_hash
        )

        logger.debug("Using %s %s", prefill_client_info, decode_client_info)

        # Stream response from decode service
        async def generate_stream():
            try:
                t_decode_send = time.time()
                t_decode_send_perf = time.perf_counter()
                print(
                    f"PERF PROXY decode_send req_id={request_id} "
                    f"ts={t_decode_send:.6f}{kv_uuid_log}",
                    flush=True,
                )
                first = True
                async for chunk in stream_from_decode(
                    decode_client_info, api, req_data, request_id, decode_forwarded
                ):
                    if first:
                        t_first = time.time()
                        t_first_perf = time.perf_counter()
                        decode_first_ms = (t_first_perf - t_decode_send_perf) * 1000.0
                        ttft_ms = (t_first_perf - t_request_recv_perf) * 1000.0
                        _PROXY_LATENCY.record("decode_first", decode_first_ms)
                        _PROXY_LATENCY.record("ttft", ttft_ms)
                        print(
                            f"PERF PROXY first_chunk req_id={request_id} "
                            f"ts={t_first:.6f} "
                            f"dur_from_decode_send_ms={decode_first_ms:.2f} "
                            f"ttft_ms={ttft_ms:.2f}{kv_uuid_log}",
                            flush=True,
                        )
                        first = False
                    yield chunk
            finally:
                _release_dp_rank(decode_client_info, decode_tracked_rank)

        return StreamingResponse(generate_stream(), media_type="application/json")

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/v1/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/v1/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    """Simple endpoint to check if the server is running."""
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
    }


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
