"""
test_api.py — Comprehensive endpoint tester for the econdata API.

Tests every endpoint, prints detailed response values, verifies security
controls, and exits non-zero if any required test fails.

Usage:
    python3 test_api.py
    python3 test_api.py --url http://localhost:8100
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import requests

BASE_URL = "http://localhost:8100"

# ── Formatting helpers ────────────────────────────────────────────────────────

RESET  = "\033[0m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

def _pass(msg: str) -> str: return f"{GREEN}[PASS]{RESET} {msg}"
def _fail(msg: str) -> str: return f"{RED}[FAIL]{RESET} {msg}"
def _skip(msg: str) -> str: return f"{YELLOW}[SKIP]{RESET} {msg}"
def _head(msg: str) -> str: return f"\n{BOLD}{msg}{RESET}"
def _dim(msg: str)  -> str: return f"{DIM}{msg}{RESET}"

# ── Test result tracking ──────────────────────────────────────────────────────

_results: list[tuple[str, str]] = []   # (label, "PASS" | "FAIL" | "SKIP")


def _record(label: str, status: str) -> None:
    _results.append((label, status))


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(path: str, headers: dict | None = None, timeout: int = 10) -> requests.Response:
    return requests.get(BASE_URL + path, headers=headers or {}, timeout=timeout)


def _assert_status(label: str, r: requests.Response, expected: int) -> bool:
    if r.status_code == expected:
        return True
    print(_fail(f"{label}: expected HTTP {expected}, got {r.status_code}"))
    try:
        print(_dim(f"  detail: {r.json().get('detail', r.text[:200])}"))
    except Exception:
        pass
    _record(label, "FAIL")
    return False


# ── Series detail printer ─────────────────────────────────────────────────────

def _print_series(s: dict[str, Any], indent: str = "    ") -> None:
    vid  = s.get("series_id", "?")
    lbl  = s.get("label", "?")
    unit = s.get("unit", "?")
    ld   = s.get("last_date", "N/A")
    lv   = s.get("last_value")
    lv_s = f"{lv:.4g}" if lv is not None else "N/A"

    val  = s.get("validation") or {}
    mae  = val.get("mae")
    r2   = val.get("r2")
    val_s = (f"MAE={mae:.4g}  R²={r2:.4g}" if mae is not None else "no validation")

    fc   = s.get("forecast", [])
    fc_s = "no forecast"
    if fc:
        f0, fl = fc[0], fc[-1]
        fc_s = (
            f"{f0['month']} mid={f0['mid']:.4g} [{f0['lo']:.4g}–{f0['hi']:.4g}]"
            f"  →  {fl['month']} mid={fl['mid']:.4g} [{fl['lo']:.4g}–{fl['hi']:.4g}]"
            f"  ({len(fc)} months)"
        )

    print(f"{indent}{BOLD}{vid}{RESET}  {_dim(lbl)}")
    print(f"{indent}  unit      : {unit}")
    print(f"{indent}  last obs  : {ld} = {lv_s}")
    print(f"{indent}  validation: {val_s}")
    print(f"{indent}  forecast  : {fc_s}")


def _print_group(data: dict[str, Any]) -> None:
    print(f"  group      : {data.get('group')}")
    print(f"  run_at     : {data.get('run_at')}")
    print(f"  series_count: {data.get('series_count')}")
    for s in data.get("series", []):
        _print_series(s)


# ── /api/summary ──────────────────────────────────────────────────────────────

def test_summary() -> None:
    label = "GET /api/summary"
    print(_head(label))
    try:
        r = _get("/api/summary")
    except requests.ConnectionError:
        print(_fail(f"Cannot connect to {BASE_URL}. Is the server running?"))
        _record(label, "FAIL")
        return

    if not _assert_status(label, r, 200):
        return

    data = r.json()
    eps  = data.get("endpoints", [])
    print(f"  generated_at : {data.get('generated_at')}")
    print(f"  endpoints    : {len(eps)}")
    for ep in eps:
        avail = f"{GREEN}available{RESET}" if ep.get("available") else f"{YELLOW}not available{RESET}"
        sids  = ", ".join(ep.get("series") or []) or "—"
        print(f"    {ep['path']:<45} {avail}  [{sids}]")

    ok = len(eps) >= 25
    if ok:
        print(_pass(f"{label}: {len(eps)} endpoints listed"))
        _record(label, "PASS")
    else:
        print(_fail(f"{label}: expected ≥25 endpoints, got {len(eps)}"))
        _record(label, "FAIL")


# ── Group endpoint tester ─────────────────────────────────────────────────────

def test_group(path: str, expected_series_ids: list[str], optional: bool = False) -> None:
    label = f"GET {path}"
    print(_head(label))
    try:
        r = _get(path)
    except requests.ConnectionError:
        print(_fail(f"Cannot connect to {BASE_URL}"))
        _record(label, "FAIL")
        return

    if optional and r.status_code == 404:
        detail = r.json().get("detail", "")
        print(_skip(f"{label}: not yet generated — {detail}"))
        _record(label, "SKIP")
        return

    if not _assert_status(label, r, 200):
        return

    data = r.json()
    _print_group(data)

    actual_ids = [s["series_id"] for s in data.get("series", [])]
    if actual_ids == expected_series_ids:
        print(_pass(f"{label}: {len(actual_ids)} series, IDs match expected"))
        _record(label, "PASS")
    else:
        print(_fail(f"{label}: expected series {expected_series_ids}, got {actual_ids}"))
        _record(label, "FAIL")


# ── Single-series endpoint tester ─────────────────────────────────────────────

def test_series(path: str, expected_id: str, expected_status: int = 200) -> None:
    label = f"GET {path}"
    print(_head(label))
    try:
        r = _get(path)
    except requests.ConnectionError:
        print(_fail(f"Cannot connect to {BASE_URL}"))
        _record(label, "FAIL")
        return

    if not _assert_status(label, r, expected_status):
        return

    if expected_status == 200:
        data = r.json()
        _print_series(data)
        actual_id = data.get("series_id", "")
        if actual_id == expected_id:
            print(_pass(f"{label}: series_id={actual_id}"))
            _record(label, "PASS")
        else:
            print(_fail(f"{label}: expected series_id={expected_id}, got {actual_id}"))
            _record(label, "FAIL")
    else:
        detail = r.json().get("detail", "")
        print(_pass(f"{label}: HTTP {expected_status} — {detail}"))
        _record(label, "PASS")


# ── Health check ─────────────────────────────────────────────────────────────

def test_health() -> None:
    label = "GET /api/health"
    print(_head(label))
    try:
        r = _get("/api/health")
    except requests.ConnectionError:
        print(_fail(f"Cannot connect to {BASE_URL}"))
        _record(label, "FAIL")
        return

    if not _assert_status(label, r, 200):
        return

    data = r.json()
    status = data.get("status")
    ts     = data.get("ts")
    if status == "ok" and ts:
        print(f"  status: {status}   ts: {ts}")
        print(_pass(f"{label}: status=ok, ts present"))
        _record(label, "PASS")
    else:
        print(_fail(f"{label}: expected status='ok' and ts present, got status={status!r} ts={ts!r}"))
        _record(label, "FAIL")


# ── Optional single-series tester ─────────────────────────────────────────────

def test_series_optional(path: str, expected_id: str) -> None:
    """Like test_series() but SKIPs (rather than FAILs) when the server returns 404."""
    label = f"GET {path}"
    print(_head(label))
    try:
        r = _get(path)
    except requests.ConnectionError:
        print(_fail(f"Cannot connect to {BASE_URL}"))
        _record(label, "FAIL")
        return

    if r.status_code == 404:
        detail = r.json().get("detail", "")
        print(_skip(f"{label}: not yet generated — {detail}"))
        _record(label, "SKIP")
        return

    if not _assert_status(label, r, 200):
        return

    data = r.json()
    _print_series(data)
    actual_id = data.get("series_id", "")
    if actual_id == expected_id:
        print(_pass(f"{label}: series_id={actual_id}"))
        _record(label, "PASS")
    else:
        print(_fail(f"{label}: expected series_id={expected_id}, got {actual_id}"))
        _record(label, "FAIL")


# ── Regime endpoint tester ────────────────────────────────────────────────────

def test_regime() -> None:
    label = "GET /api/market/regime"
    print(_head(label))
    try:
        r = _get("/api/market/regime")
    except requests.ConnectionError:
        print(_fail(f"Cannot connect to {BASE_URL}"))
        _record(label, "FAIL")
        return

    if r.status_code == 404:
        detail = r.json().get("detail", "")
        print(_skip(f"{label}: not yet generated — {detail}"))
        _record(label, "SKIP")
        return

    if not _assert_status(label, r, 200):
        return

    data = r.json()
    regime  = data.get("current_regime")
    fsi     = data.get("current_fsi")
    history = data.get("history", [])
    forecast = data.get("forecast", [])
    print(f"  current_regime : {regime}")
    print(f"  current_fsi    : {fsi}")
    print(f"  history rows   : {len(history)}")
    print(f"  forecast months: {len(forecast)}")

    missing = [k for k in ("current_regime", "current_fsi", "history", "forecast") if k not in data]
    if not missing and len(history) > 0 and len(forecast) > 0:
        print(_pass(f"{label}: regime={regime}, FSI={fsi}"))
        _record(label, "PASS")
    else:
        print(_fail(f"{label}: missing or empty fields: {missing or 'history/forecast empty'}"))
        _record(label, "FAIL")


# ── History endpoint tester ───────────────────────────────────────────────────

def test_history(series_id: str) -> None:
    path  = f"/api/series/{series_id}/history"
    label = f"GET {path}"
    print(_head(label))
    try:
        r = _get(path)
    except requests.ConnectionError:
        print(_fail(f"Cannot connect to {BASE_URL}"))
        _record(label, "FAIL")
        return

    if r.status_code == 404:
        detail = r.json().get("detail", "")
        print(_skip(f"{label}: not yet generated — {detail}"))
        _record(label, "SKIP")
        return

    if not _assert_status(label, r, 200):
        return

    data = r.json()
    sid       = data.get("series_id")
    row_count = data.get("row_count", 0)
    obs       = data.get("observations", [])
    print(f"  series_id : {sid}")
    print(f"  row_count : {row_count}")
    if obs:
        first = obs[0]
        print(f"  first obs : {first.get('observation_date')} = {first.get('value')}")

    ok = (sid == series_id and row_count > 0 and len(obs) > 0
          and "observation_date" in (obs[0] if obs else {})
          and "value" in (obs[0] if obs else {}))
    if ok:
        print(_pass(f"{label}: {row_count} rows, observations well-formed"))
        _record(label, "PASS")
    else:
        print(_fail(f"{label}: validation failed — sid={sid}, rows={row_count}, obs_count={len(obs)}"))
        _record(label, "FAIL")


# ── Security checks ───────────────────────────────────────────────────────────

def test_security() -> None:
    print(_head("Security checks"))

    # 1. Required security headers present
    label = "Security: required headers present"
    try:
        r = _get("/api/risk")
    except requests.ConnectionError:
        print(_fail("Cannot connect to server"))
        _record(label, "FAIL")
        return

    required_headers = {
        "x-content-type-options": "nosniff",
        "x-frame-options":        "DENY",
        "cache-control":          "no-store",
        "content-security-policy": "default-src 'none'",
        "referrer-policy":         "no-referrer",
    }
    missing = []
    wrong   = []
    for hdr, expected_val in required_headers.items():
        actual = r.headers.get(hdr)
        if actual is None:
            missing.append(hdr)
        elif actual != expected_val:
            wrong.append(f"{hdr}={actual!r} (expected {expected_val!r})")

    if not missing and not wrong:
        print(_pass(f"{label}: all {len(required_headers)} headers correct"))
        _record(label, "PASS")
    else:
        if missing:
            print(_fail(f"{label}: missing headers: {missing}"))
        if wrong:
            print(_fail(f"{label}: wrong values: {wrong}"))
        _record(label, "FAIL")

    # 2. Server banner suppressed
    label = "Security: no 'server' header"
    server_hdr = r.headers.get("server")
    if server_hdr:
        print(_fail(f"{label}: server header present: {server_hdr!r}"))
        _record(label, "FAIL")
    else:
        print(_pass(f"{label}: server header absent"))
        _record(label, "PASS")

    # 3. Input validation — special characters
    label = "Security: series_id with special chars → 400"
    r2 = _get("/api/business-env/A!B")
    if r2.status_code == 400:
        print(_pass(f"{label}: HTTP 400 — {r2.json().get('detail', '')}"))
        _record(label, "PASS")
    else:
        print(_fail(f"{label}: expected 400, got {r2.status_code}"))
        _record(label, "FAIL")

    # 4. Input validation — too long
    label = "Security: series_id too long → 400"
    r3 = _get("/api/business-env/" + "A" * 35)
    if r3.status_code == 400:
        print(_pass(f"{label}: HTTP 400 — {r3.json().get('detail', '')}"))
        _record(label, "PASS")
    else:
        print(_fail(f"{label}: expected 400, got {r3.status_code}"))
        _record(label, "FAIL")

    # 5. Unknown series_id → 404
    label = "Security: unknown series_id → 404"
    r4 = _get("/api/business-env/DOESNOTEXIST")
    if r4.status_code == 404:
        print(_pass(f"{label}: HTTP 404 — {r4.json().get('detail', '')}"))
        _record(label, "PASS")
    else:
        print(_fail(f"{label}: expected 404, got {r4.status_code}"))
        _record(label, "FAIL")

    # 6. CORS — external origin blocked
    label = "Security: external CORS origin blocked"
    r5 = _get("/api/risk", headers={"Origin": "http://evil.com"})
    cors = r5.headers.get("access-control-allow-origin")
    if cors is None:
        print(_pass(f"{label}: no CORS header returned for external origin"))
        _record(label, "PASS")
    else:
        print(_fail(f"{label}: unexpected CORS header: {cors!r}"))
        _record(label, "FAIL")

    # 7. CORS — localhost origin allowed
    label = "Security: localhost CORS origin allowed"
    r6 = _get("/api/risk", headers={"Origin": "http://localhost:3000"})
    cors2 = r6.headers.get("access-control-allow-origin")
    if cors2 == "http://localhost:3000":
        print(_pass(f"{label}: access-control-allow-origin={cors2!r}"))
        _record(label, "PASS")
    else:
        print(_fail(f"{label}: expected localhost origin to be allowed, got {cors2!r}"))
        _record(label, "FAIL")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary() -> int:
    passed = [l for l, s in _results if s == "PASS"]
    failed = [l for l, s in _results if s == "FAIL"]
    skipped = [l for l, s in _results if s == "SKIP"]

    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}Test Summary{RESET}")
    print(f"{'─' * 60}")
    for label, status in _results:
        if status == "PASS":
            marker = f"{GREEN}PASS{RESET}"
        elif status == "FAIL":
            marker = f"{RED}FAIL{RESET}"
        else:
            marker = f"{YELLOW}SKIP{RESET}"
        print(f"  [{marker}] {label}")

    print(f"{'─' * 60}")
    print(
        f"  {GREEN}{len(passed)} passed{RESET}  "
        f"{RED}{len(failed)} failed{RESET}  "
        f"{YELLOW}{len(skipped)} skipped{RESET}"
    )
    return 1 if failed else 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    global BASE_URL
    parser = argparse.ArgumentParser(description="Comprehensive API tester for econdata")
    parser.add_argument("--url", default=BASE_URL, help="Base URL of the API server")
    args = parser.parse_args()
    BASE_URL = args.url.rstrip("/")

    print(f"{BOLD}econdata API Test Suite{RESET}  →  {BASE_URL}")

    # ── /api/summary ──────────────────────────────────────────────────────────
    test_summary()

    # ── Business Environment ──────────────────────────────────────────────────
    test_group("/api/business-env", ["INDPRO", "TCU", "PAYEMS"])
    test_series("/api/business-env/INDPRO", "INDPRO")
    test_series("/api/business-env/TCU",    "TCU")
    test_series("/api/business-env/PAYEMS", "PAYEMS")

    # ── Consumer Demand ───────────────────────────────────────────────────────
    test_group("/api/consumer-demand", ["DSPIC96", "PCE", "PCEPILFE", "RSAFS", "RRSFS", "UMCSENT"])
    test_series("/api/consumer-demand/PCE",     "PCE")
    test_series("/api/consumer-demand/UMCSENT", "UMCSENT")

    # ── Cost of Capital ───────────────────────────────────────────────────────
    test_group("/api/cost-of-capital", ["DFF", "DPRIME", "T10Y3M", "T10Y2Y"])
    test_series("/api/cost-of-capital/DFF",    "DFF")
    test_series("/api/cost-of-capital/T10Y3M", "T10Y3M")

    # ── Risk & Leading Indicators ─────────────────────────────────────────────
    test_group("/api/risk", ["RECPROUSM156N", "UMCSENT"])
    test_series("/api/risk/RECPROUSM156N", "RECPROUSM156N")

    # ── Optional: Sector ─────────────────────────────────────────────────────
    test_group("/api/sector/bls",       [], optional=True)
    test_group("/api/sector/bea",       [], optional=True)
    test_group("/api/sector/worldbank", [], optional=True)

    # ── Optional: Venture Capital ─────────────────────────────────────────────
    test_group("/api/vc/ai",         [], optional=True)
    test_group("/api/vc/fintech",    [], optional=True)
    test_group("/api/vc/healthcare", [], optional=True)

    # ── /api/health ───────────────────────────────────────────────────────────
    test_health()

    # ── Phase 0: Market Risk ──────────────────────────────────────────────────
    test_series_optional("/api/market/vix",    "VIXCLS")
    test_group(          "/api/market/spreads", ["BAMLH0A0HYM2", "BAMLC0A0CM"], optional=True)
    test_series_optional("/api/market/dollar", "DTWEXBGS")

    # ── Phase 0: Commodities ──────────────────────────────────────────────────
    test_series_optional("/api/commodities/oil",  "DCOILWTICO")
    test_series_optional("/api/commodities/gold", "GOLDAMGBD228NLBM")

    # ── Phase 0: Yield Curve ──────────────────────────────────────────────────
    test_group("/api/market/yield-curve",
               ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"],
               optional=True)
    test_series_optional("/api/market/yield-curve/DGS10", "DGS10")

    # ── Phase 0: FSI + Regime ─────────────────────────────────────────────────
    test_series_optional("/api/market/stress", "FSI")
    test_regime()

    # ── Phase 0: Raw history ──────────────────────────────────────────────────
    test_history("VIXCLS")
    test_history("DGS10")

    # ── Phase 0: Error cases ──────────────────────────────────────────────────
    test_series("/api/market/yield-curve/INVALID!!", "—", expected_status=400)
    test_series("/api/series/DOESNOTEXIST/history",  "—", expected_status=404)

    # ── Security ──────────────────────────────────────────────────────────────
    test_security()

    return print_summary()


if __name__ == "__main__":
    sys.exit(main())
