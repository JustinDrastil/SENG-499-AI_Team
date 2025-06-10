import subprocess
import json
import sys
from onc.onc import ONC
from datetime import datetime, timedelta
import dateutil.parser
import re

OLLAMA_MODEL = "api-query-enhanced"
API_TOKEN = "c1d3ed8e-6922-4d54-bb1f-97bf97c2d37a"  # Replace with your actual ONC API token

# === Helper to resolve human-friendly time ranges ===
def resolve_time_range(time_range: str):
    today = datetime.utcnow()
    if time_range.lower() == "last week":
        start = today - timedelta(days=7)
        end = today
    elif time_range.lower() == "this month":
        start = today.replace(day=1)
        end = today
    else:
        try:
            parts = time_range.split("to")
            start = dateutil.parser.parse(parts[0].strip())
            end = dateutil.parser.parse(parts[1].strip())
        except:
            raise ValueError(f"Unsupported time range: {time_range}")
    return start.isoformat() + "Z", end.isoformat() + "Z"

# === ONC API Query Based on Structured JSON ===
def query_onc_api(query_json: dict, token: str) -> dict:
    try:
        onc = ONC(token=token)

        location = query_json["locationCode"]
        sensor = query_json["sensor"]
        stat = query_json["statistic"].lower()
        time_range = query_json["timeRange"]

        sensor_map = {
            "Temperature": "temperature",
            "Salinity": "salinity",
            "Pressure": "pressure",
            "Chlorophyll": "chlorophyll",
            "Dissolved Oxygen": "dissolvedOxygen"
        }

        property_code = sensor_map.get(sensor)
        if not property_code:
            raise ValueError(f"Unsupported sensor: {sensor}")

        date_from, date_to = resolve_time_range(time_range)

        print("📦 Calling getScalardataByLocation with:")
        print("  locationCode =", location)
        print("  deviceCategoryCode =", "ctd")
        print("  propertyCode =", property_code)
        print("  dateFrom =", date_from)
        print("  dateTo =", date_to)

        filters = {
            "locationCode": location,
            "deviceCategoryCode": "ctd",
            "propertyCode": property_code,
            "dateFrom": date_from,
            "dateTo": date_to
        }

        result = onc.getScalardataByLocation(filters)


        values = [entry['value'] for entry in result.get('data', []) if 'value' in entry]
        if not values:
            return {"error": "No data available for the given query."}

        value = {
            "average": sum(values) / len(values),
            "max": max(values),
            "min": min(values),
            "latest": values[-1],
            "trend": values[-1] - values[0] if len(values) >= 2 else 0
        }.get(stat)

        return {
            "sensor": sensor,
            "statistic": stat,
            "locationCode": location,
            "timeRange": time_range,
            "value": round(value, 3),
            "unit": result['data'][0].get('unit', 'N/A') if result['data'] else 'N/A'
        }

    except Exception as e:
        return {"error": str(e)}

# === LLM Call via Ollama ===
def call_ollama(prompt: str) -> dict:
    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            check=True
        )

        raw_output = result.stdout.strip()
        print("📤 Raw LLM Output:\n", raw_output)

        # Remove markdown/code block fences
        raw_output = re.sub(r"```(json)?", "", raw_output).strip()

        # Extract first JSON-like block (everything between first { and last })
        json_candidate = re.search(r'\{[\s\S]*?\}', raw_output)
        if not json_candidate:
            raise ValueError("❌ No JSON object found in LLM output.")

        # Clean common formatting issues
        cleaned = json_candidate.group(0)
        cleaned = cleaned.replace("'", '"')  # single to double quotes
        cleaned = re.sub(r',\s*}', '}', cleaned)  # remove trailing commas
        cleaned = re.sub(r',\s*\]', ']', cleaned)  # trailing comma in arrays

        # Load as JSON
        return json.loads(cleaned)

    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        sys.exit(1)


# === Format response into natural language ===
def format_response(data: dict) -> str:
    if "error" in data:
        return f"⚠️ Error: {data['error']}"
    return (
        f"The {data['statistic']} {data['sensor'].lower()} in Cambridge Bay "
        f"for {data['timeRange']} was {data['value']} {data['unit']}."
    )

# === Run the full flow ===
if __name__ == "__main__":
    user_prompt = input("🧠 Ask your question: ")

    print("🤖 Calling LLM to interpret the query...")
    query_json = call_ollama(user_prompt)

    print("🌊 Querying ONC data API...")
    result = query_onc_api(query_json, API_TOKEN)

    print("📝 Result:")
    print(format_response(result))
