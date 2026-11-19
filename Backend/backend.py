import os
import certifi
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import Any, TypedDict, Annotated
import operator
import uuid
import asyncio
import json
import re
from datetime import datetime
import psycopg
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command, interrupt
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from flight_tools import clean_iata, fetch_live_flights
from mcp_client import (
    tavily_mcp_search,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search,
)


# =========================
# State - original fields kept, new control fields added
# =========================
class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Original specialist results
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # Budget + HITL state
    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls: int


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


# =========================
# LLM
# =========================
# ChatGoogleGenerativeAI reads GOOGLE_API_KEY from the environment (.env).
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is missing. Please add it to your .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.3,
)


# =========================
# Text helpers
# =========================
def _content_to_text(content: Any) -> str:
    """
    Normalise anything an LLM or MCP tool can return into plain text.

    Gemini and MCP tools often return a list of content blocks such as
    [{"type": "text", "text": "..."}] instead of a string. Without this,
    the frontend receives lists and shows "[object Object]".
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        return json.dumps(content, indent=2, default=str)

    if isinstance(content, (list, tuple)):
        parts = [_content_to_text(item) for item in content]
        return "\n\n".join(part for part in parts if part)

    return str(content)


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return _content_to_text(response.content)


def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(text[start : end + 1])


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


# Shared formatting rules so the draft and the final plan look the same.
DAY_FORMAT_RULES = """
Format the day-by-day plan exactly like this. One block per day, no tables:

### Day 1 - Tokyo: Arrival and first impressions
- **Morning:** ...
- **Afternoon:** ...
- **Evening:** ...

Every day must start with a "### Day N - City: short theme" heading and have
Morning, Afternoon and Evening bullets. Keep each bullet to one or two lines.
"""


# =========================
# Supervisor Agent + Input Guardrail
# =========================
def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""

    # Fail open on parser/model errors so a temporary JSON-format issue does not
    # break the original travel-planning behavior.
    try:
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."

    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    try:
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        # The itinerary agent integrates whichever specialist results were selected.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        # Original workflow behavior is preserved as the fallback.
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }


# =========================
# Guardrail blocked response
# =========================
def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


# =========================
# Flight Agent
# =========================
# Only Aviationstack's real-time `flights` endpoint is available on the current
# plan, so the agent looks up live flights for the route and lets the LLM add
# general knowledge (durations, seasonal pricing, booking advice).
DEFAULT_ORIGIN_IATA = clean_iata(
    (os.getenv("DEFAULT_ORIGIN_IATA") or "DEL").strip().strip("\"'")
) or "DEL"

FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Route looked up: {route}

Live flight data (Aviationstack real-time flights):
{live_flight_data}

Rules:
- The live data only shows flights operating around the current time. It cannot
  confirm schedules for the traveler's future travel dates, and it has NO prices.
- Clearly separate "Live observations" from "General guidance (not live)".

Generate:
1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route (use the live data when available)
4. Typical flight duration
5. Estimated airfare range (approximate, not live)
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""


def _pick_airports(state: TravelState) -> tuple[str, str | None]:
    """Ask the LLM for the main origin and destination airport codes."""
    prompt = f"""
From this travel request, give the main international airport IATA code for
the origin and for the destination. If the origin is not stated, use
"{DEFAULT_ORIGIN_IATA}".

Trip constraints:
{state.get('trip_constraints', {})}

User request:
{state['user_query']}

Return strict JSON only:
{{"origin_iata": "", "destination_iata": ""}}
"""
    parsed = _json_from_llm(
        _llm_text(
            "You convert cities to airport codes. Return strict JSON only.",
            prompt,
        )
    )
    origin = clean_iata(parsed.get("origin_iata")) or DEFAULT_ORIGIN_IATA
    destination = clean_iata(parsed.get("destination_iata"))
    return origin, destination


def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")
    query = state["user_query"]

    route = "unknown"
    live_flight_data = (
        "Live flight data was not retrieved. Use general travel knowledge and "
        "clearly label it as non-live."
    )

    try:
        origin, destination = _pick_airports(state)

        if destination:
            route = f"{origin} -> {destination}"
            result = fetch_live_flights(origin, destination)
            live_flight_data = result["text"]
            print(f"LIVE FLIGHTS ({route}): ok={result['ok']} count={result['count']}")
        else:
            live_flight_data = (
                "The destination airport could not be determined, so no live "
                "lookup was made. Use general travel knowledge and label it as non-live."
            )

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            route=route,
            live_flight_data=live_flight_data,
        )
        flight_data = _llm_text("You are an expert travel flight planner.", prompt)

    except Exception as exc:
        print(f"FLIGHT AGENT ERROR: {type(exc).__name__}: {exc}", flush=True)
        flight_data = f"Flight information unavailable: {exc}"

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight recommendations generated")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def _try_json(value: Any) -> Any:
    """Parse JSON from text or MCP content blocks. Returns None if not JSON."""
    text = _content_to_text(value).strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(text[first : last + 1])
        except Exception:
            return None

    return None


def _clean_snippet(text: Any, limit: int = 300) -> str:
    """Strip web-page junk (image labels, link lines) from a search snippet."""
    skip_exact = {"image", "check availability"}
    kept = []

    for line in str(text).splitlines():
        # Keep the words of markdown headings (they often hold hotel names).
        line = line.lstrip("#").strip()
        if not line or line.lower() in skip_exact:
            continue
        if line.startswith("See "):
            continue
        # Image alt-text slugs such as "april-japan".
        if " " not in line and "-" in line:
            continue
        kept.append(line)

    snippet = re.sub(r"\s+", " ", " ".join(kept)).replace("[...]", "").strip()

    if len(snippet) > limit:
        snippet = snippet[:limit].rsplit(" ", 1)[0] + "..."

    return snippet


def _parse_search_results(raw: Any, min_score: float = 0.2, top: int = 5):
    """Return the most relevant Tavily results as a list of dicts."""
    data = _try_json(raw)
    if not isinstance(data, dict):
        return []

    results = [
        item
        for item in data.get("results", [])
        if isinstance(item, dict) and (item.get("score") or 0) >= min_score
    ]
    results.sort(key=lambda item: item.get("score") or 0, reverse=True)
    return results[:top]


def _format_weather(weather_raw: Any, forecast_raw: Any, city: str) -> str:
    """Turn the weather MCP JSON into a short, readable Markdown block."""
    current = _try_json(weather_raw)

    if not isinstance(current, dict) or "temperature_c" not in current:
        return (
            f"Live weather for {city} could not be read. "
            "Give general seasonal guidance instead."
        )

    lines = [
        f"**Today's live conditions in {current.get('city', city)}**",
        "",
        f"- Temperature: {current.get('temperature_c')}°C "
        f"(feels like {current.get('feels_like_c')}°C)",
        f"- Sky: {str(current.get('condition', 'n/a')).capitalize()}",
        f"- Humidity: {current.get('humidity')}%",
        f"- Wind: {current.get('wind_speed')} m/s",
    ]

    forecast = _try_json(forecast_raw)
    entries = forecast.get("forecast", []) if isinstance(forecast, dict) else []

    if entries:
        lines += ["", "**Next few hours**", ""]
        for item in entries:
            try:
                when = datetime.strptime(
                    item["datetime"], "%Y-%m-%d %H:%M:%S"
                ).strftime("%d %b, %H:%M")
            except Exception:
                when = str(item.get("datetime", ""))
            lines.append(
                f"- {when}: {item.get('temperature_c')}°C, "
                f"{str(item.get('condition', '')).capitalize()}"
            )

    lines += [
        "",
        "_This is today's weather, not the forecast for your travel dates. "
        "Use seasonal knowledge for the trip dates._",
    ]
    return "\n".join(lines)


# =========================
# Hotel Agent
# =========================
def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"

    try:
        raw_results = asyncio.run(tavily_mcp_search(query))
        results = _parse_search_results(raw_results)

        if results:
            evidence = "\n\n".join(
                f"{item.get('title', 'Untitled')} ({item.get('url', '')})\n"
                f"{_clean_snippet(item.get('content', ''), 700)}"
                for item in results
            )

            summary = _llm_text(
                "You are a helpful travel hotel advisor.",
                f"""
Using only the search results below (ignore irrelevant ones), write clear
hotel recommendations for this trip.

Trip request:
{state['user_query']}

Search results:
{evidence}

Format:
- A short heading per city or area, with 2-3 hotel bullets.
- For each hotel: bold name, then one line on why it suits this trip.
- Mention prices only if the results state them.
- End with one practical booking tip.
- Markdown only. No tables, no JSON.
""",
            )

            sources = "\n".join(
                f"- [{item.get('title', item.get('url', 'Source'))}]"
                f"({item.get('url', '')})"
                for item in results
                if item.get("url")
            )
            hotel_results = f"{summary}\n\n**Sources**\n\n{sources}"
        else:
            hotel_results = _content_to_text(raw_results)

    except Exception as exc:
        print(
            f"HOTEL AGENT MCP ERROR: {type(exc).__name__}: {exc}",
            flush=True,
        )

        hotel_results = (
            "Live hotel search is temporarily unavailable. "
            "Provide general accommodation and neighborhood "
            "guidance based on the destination and clearly "
            "label it as non-live advice."
        )

    return {
        "hotel_results": hotel_results,
        "messages": [AIMessage(content="Hotel information processed.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Weather Agent
# =========================
def weather_agent(state: TravelState):
    city = extract_destination(state["user_query"])

    try:
        weather_raw = asyncio.run(weather_mcp_search(city))
        forecast_raw = asyncio.run(forecast_mcp_search(city))
        weather_results = _format_weather(weather_raw, forecast_raw, city)

    except Exception as exc:
        print(
            f"WEATHER AGENT MCP ERROR: {type(exc).__name__}: {exc}",
            flush=True,
        )

        weather_results = (
            f"Live weather information for {city} "
            "is temporarily unavailable. Give general "
            "seasonal guidance and advise the traveler "
            "to verify the forecast before departure."
        )

    return {
        "weather_results": weather_results,
        "messages": [AIMessage(content="Weather information processed.")],
    }


# =========================
# Budget Agent
# =========================
def budget_agent(state: TravelState):
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Return:
1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate.
"""

    budget_text = _llm_text("You are a practical travel budget analyst.", prompt)

    return {
        "budget_results": budget_text,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Itinerary Agent
# =========================
def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Budget Results:
{state.get('budget_results', '')}

Make the itinerary practical, budget-aware, and easy to follow.
Create a clear draft that is ready for human review.

{DAY_FORMAT_RULES}
"""

    itinerary_text = _llm_text("You are an expert travel planner.", prompt)

    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the "
        "final polished plan, or provide feedback for revision."
    )

    return {
        "itinerary": itinerary_text,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Human-in-the-Loop approval
# =========================
def human_approval_agent(state: TravelState):
    # Do not wrap interrupt() in try/except. LangGraph uses it to pause execution.
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# =========================
# Final Response Agent
# =========================
def final_agent(state: TravelState):
    if state.get("approved", False):
        review_instruction = (
            "The user approved the draft. Preserve its decisions while polishing it."
        )
    else:
        review_instruction = f"""
The user requested a revision. Apply this feedback carefully:
{state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}
"""

    final_prompt = f"""
Generate the final travel response for the user.

Human Review:
{review_instruction}

User Request:
{state['user_query']}

Supervisor Constraints:
{state.get('trip_constraints', {})}

Flights:
{state.get('flight_results', '')}

Hotels:
{state.get('hotel_results', '')}

Weather:
{state.get('weather_results', '')}

Budget Analysis:
{state.get('budget_results', '')}

Draft Itinerary:
{state.get('itinerary', '')}

Format the final answer beautifully using these sections:
1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight APIs may not provide ticket prices when pricing is unavailable.
- Include weather-based travel advice.
- Keep the response useful for real travel planning.
- Incorporate the human feedback when revision was requested.
- Do NOT use a table for the day-by-day itinerary; use the day format below.
- A table is allowed only for Estimated Budget (header row, a |---|---| separator row, one row per line).
- Keep Flight, Hotel and Weather sections short and readable (bullets, no raw data).

{DAY_FORMAT_RULES}
"""

    final_text = _llm_text(
        "You are a professional AI travel booking assistant.",
        final_prompt,
    )

    return {
        "final_response": final_text,
        "messages": [AIMessage(content=final_text)],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Dynamic Supervisor Routing
# =========================
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route


# =========================
# Build Graph
# =========================
graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

graph.add_conditional_edges(
    "flight_agent", route_after_agent("flight_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "weather_agent", route_after_agent("weather_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "budget_agent", route_after_agent("budget_agent"), ROUTE_MAP
)

graph.add_edge("itinerary_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)


# =========================
# PostgreSQL Checkpointer
# =========================
DATABASE_URL = get_database_url()
_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row,
)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


# =========================
# FastAPI-facing helpers
# =========================
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = _content_to_text(messages[-1].content) if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    return {
        "thread_id": thread_id,
        "answer": _content_to_text(answer),
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": _content_to_text(result.get("flight_results", "")),
        "hotel_results": _content_to_text(result.get("hotel_results", "")),
        "weather_results": _content_to_text(result.get("weather_results", "")),
        "budget_results": _content_to_text(result.get("budget_results", "")),
        "itinerary": _content_to_text(
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else result.get("itinerary", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Start a new travel-planning run and pause at human approval."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )

    return _serialize_result(result, thread_id)


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    config = {"configurable": {"thread_id": thread_id}}
    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    return _serialize_result(result, thread_id)