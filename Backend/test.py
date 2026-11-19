import os
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("AVIATIONSTACK_API_KEY")

ENDPOINTS = ["flights", "airports", "airlines", "routes", "airplanes", "cities", "countries"]

for name in ENDPOINTS:
    for scheme in ("https", "http"):   # the free plan may be HTTP-only
        try:
            r = requests.get(
                f"{scheme}://api.aviationstack.com/v1/{name}",
                params={"access_key": KEY, "limit": 1},
                timeout=20,
            )
            data = r.json()
        except Exception as exc:
            print(f"{name:10} {scheme}: request failed ({exc})")
            continue

        if "error" in data:
            print(f"{name:10} {scheme}: BLOCKED -> {data['error'].get('code')}")
        else:
            print(f"{name:10} {scheme}: OK")
            break