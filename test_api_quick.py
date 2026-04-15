"""
test_api_quick.py — Quick smoke test for the econdata API.

Hits the 4 required group endpoints + /api/summary and prints one line each.
Use this after restarting the server or running fred_refresh.py.

Usage:
    python3 test_api_quick.py
    python3 test_api_quick.py --url http://localhost:8100
"""

from __future__ import annotations

import argparse
import sys

import requests

BASE_URL = "http://localhost:8100"


def check(label: str, path: str, base: str, expected_status: int = 200) -> bool:
    url = base + path
    try:
        r = requests.get(url, timeout=10)
    except requests.ConnectionError:
        print(f"[FAIL] {label:<40}  — Cannot connect to {base}. Is the server running?")
        return False

    if r.status_code != expected_status:
        print(f"[FAIL] {label:<40}  — HTTP {r.status_code} (expected {expected_status})")
        return False

    data = r.json()
    if path == "/api/summary":
        n = len(data.get("endpoints", []))
        print(f"[PASS] {label:<40}  — {n} endpoints listed")
    else:
        count = data.get("series_count", "?")
        run_at = data.get("run_at", "?")
        print(f"[PASS] {label:<40}  — {count} series, run_at={run_at}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=BASE_URL, help="Base URL of the API server")
    args = parser.parse_args()

    base = args.url.rstrip("/")

    checks = [
        ("GET /api/summary",         "/api/summary"),
        ("GET /api/business-env",    "/api/business-env"),
        ("GET /api/consumer-demand", "/api/consumer-demand"),
        ("GET /api/cost-of-capital", "/api/cost-of-capital"),
        ("GET /api/risk",            "/api/risk"),
    ]

    results = [check(label, path, base) for label, path in checks]
    passed = sum(results)
    total  = len(results)

    print()
    if all(results):
        print(f"All {total} checks passed.")
        return 0
    else:
        print(f"{passed}/{total} checks passed. {total - passed} FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
