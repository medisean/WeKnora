#!/usr/bin/env python3
"""Expose a Dify External Knowledge API backed by WeKnora search."""

from __future__ import annotations

import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


LISTEN_HOST = os.getenv("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.getenv("LISTEN_PORT", "8090"))
WEKNORA_BASE_URL = os.getenv(
    "WEKNORA_BASE_URL", "http://localhost:8081/api/v1"
).rstrip("/")
WEKNORA_API_KEY = os.environ["WEKNORA_API_KEY"]
ADAPTER_API_KEY = os.environ["ADAPTER_API_KEY"]
RESOURCE_URL_MODE = os.getenv("RESOURCE_URL_MODE", "public").strip().lower()
ALLOWED_KB_IDS = {
    value.strip()
    for value in os.getenv("ALLOWED_KB_IDS", "").split(",")
    if value.strip()
}

if RESOURCE_URL_MODE not in {"handle", "public"}:
    raise RuntimeError("RESOURCE_URL_MODE must be 'handle' or 'public'")

MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)")


def write_json(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class Handler(BaseHTTPRequestHandler):
    server_version = "DifyWeKnoraBridge/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def do_GET(self) -> None:
        if self.path.rstrip("/") in {"", "/health"}:
            write_json(self, 200, {"status": "ok", "backend": "weknora"})
            return
        write_json(self, 404, {"error_code": 404, "error_msg": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/retrieval":
            write_json(self, 404, {"error_code": 404, "error_msg": "not found"})
            return

        if self.headers.get("Authorization") != f"Bearer {ADAPTER_API_KEY}":
            write_json(self, 403, {"error_code": 1001, "error_msg": "invalid API key"})
            return

        try:
            size = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(size) if size else b""
            # Dify validates an endpoint with an empty POST on some versions.
            if not raw.strip():
                write_json(self, 200, {"records": []})
                return
            body = json.loads(raw)
            query = str(body.get("query", "")).strip()
            knowledge_id = str(body.get("knowledge_id", "")).strip()
            if not query or not knowledge_id:
                write_json(
                    self,
                    400,
                    {"error_code": 1002, "error_msg": "query and knowledge_id are required"},
                )
                return
            if ALLOWED_KB_IDS and knowledge_id not in ALLOWED_KB_IDS:
                write_json(
                    self,
                    403,
                    {"error_code": 1003, "error_msg": "knowledge base is not allowed"},
                )
                return

            settings = body.get("retrieval_setting") or {}
            top_k = max(1, min(int(settings.get("top_k", 5)), 50))
            threshold = float(settings.get("score_threshold", 0) or 0)
            request_body = {
                "query": query,
                "knowledge_base_ids": [knowledge_id],
                "match_count": top_k,
                "rerank": {"enabled": False},
            }
            upstream = Request(
                f"{WEKNORA_BASE_URL}/knowledge-search?resource_urls={RESOURCE_URL_MODE}",
                data=json.dumps(request_body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": WEKNORA_API_KEY,
                },
                method="POST",
            )
            with urlopen(upstream, timeout=60) as response:
                result = json.load(response)

            records = []
            for item in result.get("data") or []:
                score = float(item.get("score", 0) or 0)
                if score < threshold:
                    continue
                metadata = dict(item.get("metadata") or {})
                content = str(item.get("content", ""))
                image_urls = MARKDOWN_IMAGE_RE.findall(content)
                metadata.update(
                    {
                        "weknora_chunk_id": item.get("id", ""),
                        "weknora_knowledge_id": item.get("knowledge_id", ""),
                        "weknora_knowledge_base_id": item.get("knowledge_base_id", knowledge_id),
                        "source": item.get("knowledge_filename")
                        or item.get("knowledge_title", ""),
                        "image_urls": image_urls,
                    }
                )
                records.append(
                    {
                        "content": content,
                        "score": score,
                        "title": item.get("knowledge_title")
                        or item.get("knowledge_filename")
                        or "WeKnora",
                        "metadata": metadata,
                    }
                )
            write_json(self, 200, {"records": records})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            write_json(self, 400, {"error_code": 1002, "error_msg": str(exc)})
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            write_json(
                self,
                502,
                {"error_code": 2001, "error_msg": f"WeKnora HTTP {exc.code}: {detail}"},
            )
        except (URLError, TimeoutError) as exc:
            write_json(self, 502, {"error_code": 2002, "error_msg": str(exc)})
        except Exception as exc:  # Keep Dify's response JSON-shaped on failures.
            write_json(self, 500, {"error_code": 2000, "error_msg": str(exc)})


if __name__ == "__main__":
    print(
        f"Dify-WeKnora bridge listening on http://{LISTEN_HOST}:{LISTEN_PORT}",
        flush=True,
    )
    ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler).serve_forever()
