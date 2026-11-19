"""
Helpers for the flight agent.

Your Aviationstack plan only allows the `flights` endpoint, so this module
calls that one endpoint directly with `requests` and turns the result into
a compact text summary for the LLM.
"""

import os
from collections import Counter
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
BASE_URL = "https://api.aviationstack.com/v1/flights"
REQUEST_TIMEOUT_SECONDS = 20

# airportsdata (already in requirements.txt) is used to validate IATA codes.
try:
    import airportsdata

    _AIRPORTS = airportsdata.load("IATA")
except Exception:
    _AIRPORTS = None


def clean_iata(code: Any) -> str | None:
    """Return an upper-case 3-letter IATA code, or None if it is not valid."""
    if not code:
        return None

    code = str(code).strip().upper()

    if len(code) != 3 or not code.isalpha():
        return None

    if _AIRPORTS is not None and code not in _AIRPORTS:
        return None

    return code


def _safe_error(exc: Exception) -> str:
    """requests errors can contain the full URL, including the API key."""
    message = str(exc)
    if API_KEY:
        message = message.replace(API_KEY, "***")
    return message


def _time_only(value: str | None) -> str:
    # Aviationstack times look like 2026-11-19T10:05:00+00:00 and are LOCAL
    # airport times (the +00:00 suffix is not a real UTC offset).
    if not value or "T" not in value:
        return "n/a"
    return value.split("T")[1][:5]


def fetch_live_flights(
    dep_iata: str,
    arr_iata: str,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Query real-time flights between two airports.

    Returns {"ok": bool, "count": int, "text": str}. Never raises.
    """
    route = f"{dep_iata} -> {arr_iata}"

    if not API_KEY:
        return {
            "ok": False,
            "count": 0,
            "text": "Live flight lookup is not configured (missing AVIATIONSTACK_API_KEY).",
        }

    try:
        response = requests.get(
            BASE_URL,
            params={
                "access_key": API_KEY,
                "dep_iata": dep_iata,
                "arr_iata": arr_iata,
                "limit": limit,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        payload = response.json()
    except Exception as exc:
        return {
            "ok": False,
            "count": 0,
            "text": f"Live flight lookup for {route} failed: {_safe_error(exc)}",
        }

    if "error" in payload:
        error = payload["error"]
        return {
            "ok": False,
            "count": 0,
            "text": (
                f"Live flight lookup for {route} was rejected: "
                f"{error.get('code', 'unknown_error')}."
            ),
        }

    flights = payload.get("data") or []

    if not flights:
        return {
            "ok": True,
            "count": 0,
            "text": (
                f"No real-time flights were found for {route} right now. "
                "This does not mean the route does not exist; the API only "
                "lists flights operating around the current time."
            ),
        }

    airlines = Counter(
        (item.get("airline") or {}).get("name") or "Unknown airline"
        for item in flights
    )

    lines = [
        f"Real-time flights found for {route}: {len(flights)}",
        "Airlines seen: "
        + ", ".join(f"{name} ({count})" for name, count in airlines.most_common()),
        "Sample flights (times are local airport times):",
    ]

    for item in flights[:6]:
        flight_code = (item.get("flight") or {}).get("iata") or "n/a"
        airline_name = (item.get("airline") or {}).get("name") or "n/a"
        departure = item.get("departure") or {}
        arrival = item.get("arrival") or {}
        lines.append(
            f"- {flight_code} | {airline_name} | status: "
            f"{item.get('flight_status') or 'n/a'} | "
            f"departs {_time_only(departure.get('scheduled'))} | "
            f"arrives {_time_only(arrival.get('scheduled'))}"
        )

    return {"ok": True, "count": len(flights), "text": "\n".join(lines)}