"""Done check for a deployed demo. Exits non-zero on any failure.

uv run python scripts/check_demo.py --url https://<space>.hf.space
uv run python scripts/check_demo.py --url ... --rate-limit   # spends a few queries
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

PROBE = "What is the danger zone for food temperature? zzprobe"


def run_checks(client: Any, rate_limit: bool = False) -> list[str]:
    problems: list[str] = []
    r = client.get("/healthz")
    if r.status_code != 200 or r.json() != {"ok": True}:
        problems.append(f"/healthz returned {r.status_code}")
    page = client.get("/")
    if page.status_code != 200 or "chef-rag" not in page.text:
        problems.append("/ did not serve the UI")
    r = client.post("/query", json={"question": PROBE})
    if r.status_code == 429 and r.json().get("error") == "demo_budget_reached":
        problems.append("demo budget already reached (expected response, but no answer to check)")
    elif r.status_code != 200:
        problems.append(f"/query returned {r.status_code}: {r.text[:200]}")
    else:
        body = r.json()
        for key in ("answer", "refused", "citations", "scores"):
            if key not in body:
                problems.append(f"/query response missing {key!r}")
        if PROBE in r.text:
            problems.append("response echoes the question text")
        if not body.get("scores"):
            problems.append("no retrieval scores returned")
        if body.get("answer") and not body.get("refused") and not body.get("citations"):
            problems.append("answer has no citations")
    if client.post("/query", json={"question": "   "}).status_code != 422:
        problems.append("blank question was not rejected with 422")
    if rate_limit:
        codes = [
            client.post("/query", json={"question": "rate probe"}).status_code for _ in range(12)
        ]
        if 429 not in codes:
            problems.append(f"no 429 after 12 rapid queries: {codes}")
    return problems


def main(argv: list[str] | None = None) -> int:
    import httpx

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--rate-limit", action="store_true")
    args = ap.parse_args(argv)
    with httpx.Client(base_url=args.url.rstrip("/"), timeout=60.0) as client:
        problems = run_checks(client, args.rate_limit)
    for p in problems:
        print(f"FAIL {p}", file=sys.stderr)
    if not problems:
        print("OK")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
