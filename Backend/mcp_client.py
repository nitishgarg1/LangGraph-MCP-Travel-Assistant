import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env before anything reads environment variables.
load_dotenv()

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


# Automatically find the current project folder.
PROJECT_DIR = Path(__file__).resolve().parent
WEATHER_SERVER_PATH = PROJECT_DIR / "custom_weather_mcp_server.py"


# Pass the full environment to the local stdio MCP server,
# plus the OpenWeather key it needs.
WEATHER_ENV = os.environ.copy()
WEATHER_ENV["OPENWEATHER_API_KEY"] = OPENWEATHER_API_KEY or ""


# ==========================================
# LLM (used only for destination extraction)
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
)


# ==========================================
# MCP client configuration
# ==========================================
# Flights are no longer fetched through an MCP server: flight_tools.py calls
# the Aviationstack `flights` endpoint directly.

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": (
                "https://mcp.tavily.com/mcp/"
                f"?tavilyApiKey={TAVILY_API_KEY}"
            ),
        },

        "weather": {
            "transport": "stdio",

            # Use the same Python environment that runs app.py.
            "command": sys.executable,

            # custom_weather_mcp_server.py from the current project directory.
            "args": [str(WEATHER_SERVER_PATH)],

            "env": WEATHER_ENV,
        },
    }
)


# ==========================================
# Diagnostic function
# ==========================================

async def get_all_tools():
    """
    Load each MCP server separately.

    A broken server will not prevent the other
    working servers from loading.
    """

    all_tools = []

    for server_name in ("tavily", "weather"):
        try:
            tools = await client.get_tools(server_name=server_name)

            all_tools.extend(tools)

            print(f"\nAvailable tools from {server_name} MCP:\n")

            for tool in tools:
                print(tool.name)

        except Exception as error:
            print(f"\nCould not connect to {server_name} MCP:\n{error}\n")

    return all_tools


# ==========================================
# Tavily MCP tool
# ==========================================

search_tool = None


async def initialize_mcp():
    """Initialize only Tavily."""

    global search_tool

    if search_tool is not None:
        return

    tools = await client.get_tools(server_name="tavily")

    tools_by_name = {tool.name: tool for tool in tools}

    search_tool = tools_by_name.get("tavily_search")

    if search_tool is None:
        available_tools = ", ".join(tools_by_name.keys())

        raise RuntimeError(
            "Tavily MCP connected, but the "
            "'tavily_search' tool was not found. "
            f"Available tools: {available_tools or 'none'}"
        )


async def tavily_mcp_search(query: str):
    await initialize_mcp()

    result = await search_tool.ainvoke({"query": query})

    return result


# ==========================================
# Weather MCP tools
# ==========================================

weather_tool = None
forecast_tool = None


async def initialize_weather_tools():
    global weather_tool
    global forecast_tool

    if weather_tool is not None and forecast_tool is not None:
        return

    if not WEATHER_SERVER_PATH.exists():
        raise FileNotFoundError(
            "Weather MCP server file was not found: "
            f"{WEATHER_SERVER_PATH}"
        )

    # Load only Weather.
    tools = await client.get_tools(server_name="weather")

    tools_by_name = {tool.name: tool for tool in tools}

    weather_tool = tools_by_name.get("get_current_weather")
    forecast_tool = tools_by_name.get("get_forecast")

    missing_tools = []

    if weather_tool is None:
        missing_tools.append("get_current_weather")

    if forecast_tool is None:
        missing_tools.append("get_forecast")

    if missing_tools:
        available_tools = ", ".join(tools_by_name.keys())

        raise RuntimeError(
            "Missing Weather MCP tools: "
            f"{', '.join(missing_tools)}. "
            f"Available tools: {available_tools or 'none'}"
        )


async def weather_mcp_search(city: str):
    await initialize_weather_tools()

    result = await weather_tool.ainvoke({"city": city})

    return result


async def forecast_mcp_search(city: str):
    await initialize_weather_tools()

    result = await forecast_tool.ainvoke({"city": city})

    return result


# ==========================================
# Destination extractor
# ==========================================

def _content_to_text(content) -> str:
    """Gemini returns content as a string or a list of parts; normalise it."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, (list, tuple)):
        return "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
        )

    return str(content)


def extract_destination(query: str):
    prompt = f"""
Extract only the destination city or country.

Query:
{query}

Return only the destination name, nothing else.
"""
    response = llm.invoke(prompt)

    return _content_to_text(response.content).strip().strip('."\'')