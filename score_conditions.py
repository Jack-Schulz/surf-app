import csv
import json
import math
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import shape
from shapely.ops import unary_union

GEOJSON_PATH = "morse.geojson"
FETCH_CSV = "morse_fetch.csv"
OUTPUT_CSV = "morse_conditions.csv"
OUTPUT_PNG = "morse_conditions.png"

LAT, LNG = 40.093, -86.044

LAT_MIN, LAT_MAX = 39.9, 40.2
LNG_MIN, LNG_MAX = -86.2, -85.8

DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DIR_DEGREES = [0, 45, 90, 135, 180, 225, 270, 315]

CONDITION_LABELS = {(0, 33): "Glass", (34, 66): "Moderate", (67, 100): "Rough"}
CONDITION_COLORS = {"Glass": "#4CAF50", "Moderate": "#FFC107", "Rough": "#F44336"}


def fetch_weather():
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LNG}"
        "&current_weather=true"
        "&wind_speed_unit=mph"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    cw = resp.json()["current_weather"]
    return cw["windspeed"], cw["winddirection"]


def nearest_direction(deg):
    """Return the fetch column name for the nearest compass direction."""
    deg = deg % 360
    deltas = [abs((deg - d + 180) % 360 - 180) for d in DIR_DEGREES]
    return DIRECTIONS[deltas.index(min(deltas))]


def load_fetch(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def condition_label(score):
    if score <= 33:
        return "Glass"
    elif score <= 66:
        return "Moderate"
    return "Rough"


def load_polygon(path):
    with open(path) as f:
        data = json.load(f)
    valid = []
    for feat in data["features"]:
        g = shape(feat["geometry"])
        minx, miny, maxx, maxy = g.bounds
        if miny >= LAT_MIN and maxy <= LAT_MAX and minx >= LNG_MIN and maxx <= LNG_MAX:
            valid.append(g)
    return unary_union(valid)


def main():
    print("Fetching weather from Open-Meteo...")
    wind_speed, wind_dir_deg = fetch_weather()
    dir_name = nearest_direction(wind_dir_deg)
    print(f"  Wind speed : {wind_speed} mph")
    print(f"  Wind dir   : {wind_dir_deg}° → nearest fetch direction: {dir_name}")

    print("Loading fetch data...")
    fetch_rows = load_fetch(FETCH_CSV)
    fetch_col = f"fetch_{dir_name}"

    print("Calculating roughness scores...")
    raw_scores = []
    for row in fetch_rows:
        fetch_m = float(row[fetch_col])
        roughness = fetch_m * (wind_speed ** 2)
        raw_scores.append(roughness)

    raw = np.array(raw_scores)
    rmin, rmax = raw.min(), raw.max()
    if rmax > rmin:
        normalized = ((raw - rmin) / (rmax - rmin) * 100).round(1)
    else:
        normalized = np.zeros_like(raw)

    print(f"Writing {OUTPUT_CSV}...")
    results = []
    for row, score in zip(fetch_rows, normalized):
        label = condition_label(score)
        results.append({
            "point_id": row["point_id"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "roughness_score": score,
            "condition": label,
        })

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["point_id", "latitude", "longitude", "roughness_score", "condition"])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    counts = {"Glass": 0, "Moderate": 0, "Rough": 0}
    for r in results:
        counts[r["condition"]] += 1
    total = len(results)
    print(f"\nCondition summary ({total} points):")
    for label, count in counts.items():
        print(f"  {label:<10} {count:>4} pts  ({count/total*100:.1f}%)")

    print(f"\nRendering {OUTPUT_PNG}...")
    polygon = load_polygon(GEOJSON_PATH)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw shoreline outline only — no fill
    geoms = list(polygon.geoms) if polygon.geom_type == "MultiPolygon" else [polygon]
    for poly in geoms:
        lngs, lats = zip(*[(x, y) for x, y in poly.exterior.coords])
        ax.plot(lngs, lats, color="black", linewidth=1.2, zorder=2)

    # Scatter condition points — cast to float explicitly
    for label in ["Glass", "Moderate", "Rough"]:
        pts = [r for r in results if r["condition"] == label]
        if pts:
            ax.scatter(
                [float(p["longitude"]) for p in pts],
                [float(p["latitude"]) for p in pts],
                c=CONDITION_COLORS[label],
                label=f"{label} ({counts[label]})",
                s=12, zorder=3,
            )

    ax.set_title(
        f"Morse Reservoir Conditions\n"
        f"Wind: {wind_speed} mph from {wind_dir_deg}° ({dir_name})"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    print(f"Plot saved to {OUTPUT_PNG}")
    print("\nDone.")


if __name__ == "__main__":
    main()
