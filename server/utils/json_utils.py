import numpy as np
import pandas as pd
from datetime import datetime

def compress_onc_json_response(json_data: dict) -> dict:
    """
    Compress an ONC scalarData JSON response into a compact summary suitable for LLM input.
    Includes summary statistics and daily/monthly/yearly averages depending on the data span.
    """
    location = json_data["parameters"]["locationCode"]
    property_code = json_data["parameters"]["propertyCode"][0]
    sensor_entry = json_data["sensorData"][0]
    unit = sensor_entry.get("unitOfMeasure", "")
    sample_times = sensor_entry["data"]["sampleTimes"]
    values = sensor_entry["data"]["values"]

    # Parse timestamps and clean values
    timestamps = [datetime.fromisoformat(ts.replace("Z", "+00:00")) for ts in sample_times]
    values_array = np.array(values, dtype=np.float64)
    clean_values = values_array[~np.isnan(values_array)]

    # Prepare DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "value": values_array
    })
    df["date"] = df["timestamp"].dt.date
    df["month"] = df["timestamp"].dt.to_period("M")
    df["year"] = df["timestamp"].dt.to_period("Y")

    # Determine averaging granularity
    start_time, end_time = timestamps[0], timestamps[-1]
    duration_days = (end_time - start_time).days
    if duration_days <= 2:
        interval = "daily"
        grouped = df.groupby("date")["value"].mean().reset_index(name="average")
    elif duration_days <= 60:
        interval = "monthly"
        grouped = df.groupby("month")["value"].mean().reset_index(name="average")
    else:
        interval = "yearly"
        grouped = df.groupby("year")["value"].mean().reset_index(name="average")

    # Format group keys as strings
    grouped = grouped.astype(str).to_dict(orient="records")

    # Final compressed structure
    return {
        "location": location,
        "property": property_code,
        "unit": unit,
        "startTime": start_time.isoformat(),
        "endTime": end_time.isoformat(),
        "sampleCount": int(len(values)),
        "summary": {
            "min": float(np.min(clean_values)),
            "max": float(np.max(clean_values)),
            "mean": float(np.mean(clean_values))
        },
        f"{interval}Averages": grouped,
        "note": f"Original dataset contains {len(values)} samples from {start_time.date()} to {end_time.date()}."
    }
