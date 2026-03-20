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

FETCH_CSV    = "morse_fetch.csv"
DEPTH_CSV    = "morse_grid_depth.csv"
GEOJSON_PATH = "morse.geojson"
OUTPUT_CSV   = "morse_final_conditions.csv"
OUTPUT_PNG   = "morse_final_conditions.png"

LAT, LNG = 40.093, -86.044

LAT_MIN, LAT_MAX = 39.9, 40.2
LNG_MIN, LNG_MAX = -86.2, -85.8

DIRECTIONS   = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DIR_DEGREES  = [0, 45, 90, 135, 180, 225, 270, 315]

CONDITION_COLORS = {
    "GREEN":  "#2ECC71",
    "YELLOW": "#F1C40F",
    "ORANGE": "#E67E22",
    "RED":    "#E74C3C",
}

CONDITION_LABELS = {
    "GREEN":  "Glass + Deep (great)",
    "YELLOW": "Calm but Shallow",
    "ORANGE": "Moderate",
    "RED":    "Rough",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv_by_id(path):
    with open(path) as f:
        return {row["point_id"]: row for row in csv.DictReader(f)}


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


# ---------------------------------------------------------------------------
# Wind
# ---------------------------------------------------------------------------

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
    return float(cw["windspeed"]), float(cw["winddirection"])


def nearest_direction(deg):
    deg = deg % 360
    deltas = [abs((deg - d + 180) % 360 - 180) for d in DIR_DEGREES]
    return DIRECTIONS[deltas.index(min(deltas))]


# ---------------------------------------------------------------------------
# Condition logic
# ---------------------------------------------------------------------------

def wind_condition(roughness_score):
    if roughness_score <= 33:
        return "Glass"
    if roughness_score <= 66:
        return "Moderate"
    return "Rough"


def depth_tier(depth_ft):
    if depth_ft >= 15:
        return "Deep"
    if depth_ft >= 8:
        return "Moderate Depth"
    return "Shallow"


def final_condition(wind_cond, depth_t):
    if wind_cond == "Rough":
        return "RED"
    if depth_t == "Shallow":
        return "YELLOW"
    if wind_cond == "Glass":
        return "GREEN"
    return "ORANGE"  # Moderate + Deep or Moderate Depth


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Live wind ---
    print("Fetching live wind from Open-Meteo...")
    wind_speed, wind_dir_deg = fetch_weather()
    dir_name = nearest_direction(wind_dir_deg)
    print(f"  Wind speed : {wind_speed} mph")
    print(f"  Wind dir   : {wind_dir_deg}° → {dir_name}")

    # --- Merge fetch + depth on point_id ---
    print("Loading and merging fetch + depth data...")
    fetch_data = load_csv_by_id(FETCH_CSV)
    depth_data = load_csv_by_id(DEPTH_CSV)

    point_ids = sorted(fetch_data.keys())
    missing = [pid for pid in point_ids if pid not in depth_data]
    if missing:
        print(f"  Warning: {len(missing)} point_ids missing from depth CSV — skipping")
        point_ids = [pid for pid in point_ids if pid in depth_data]

    print(f"  {len(point_ids)} points merged")

    # --- Roughness + normalization ---
    fetch_col = f"fetch_{dir_name}"
    raw_roughness = np.array([
        float(fetch_data[pid][fetch_col]) * (wind_speed ** 2)
        for pid in point_ids
    ])
    rmin, rmax = raw_roughness.min(), raw_roughness.max()
    if rmax > rmin:
        normalized = ((raw_roughness - rmin) / (rmax - rmin) * 100).round(1)
    else:
        normalized = np.zeros_like(raw_roughness)

    # --- Build result rows ---
    results = []
    for pid, raw_score in zip(point_ids, normalized):
        frow = fetch_data[pid]
        drow = depth_data[pid]
        depth_ft = float(drow["depth_ft"])
        w_cond   = wind_condition(raw_score)
        d_tier   = depth_tier(depth_ft)
        f_cond   = final_condition(w_cond, d_tier)
        results.append({
            "point_id":       pid,
            "latitude":       float(frow["latitude"]),
            "longitude":      float(frow["longitude"]),
            "depth_ft":       depth_ft,
            "roughness_score": raw_score,
            "wind_condition": w_cond,
            "depth_tier":     d_tier,
            "final_condition": f_cond,
        })

    # --- Write CSV ---
    print(f"Writing {OUTPUT_CSV}...")
    fieldnames = ["point_id", "latitude", "longitude", "depth_ft",
                  "roughness_score", "wind_condition", "depth_tier", "final_condition"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # --- Summary ---
    counts = {k: 0 for k in CONDITION_COLORS}
    for r in results:
        counts[r["final_condition"]] += 1
    total = len(results)

    print(f"\nCondition summary ({total} points):")
    for cond in ["GREEN", "YELLOW", "ORANGE", "RED"]:
        n = counts[cond]
        print(f"  {CONDITION_LABELS[cond]:<30}  {n:>4} pts  ({n/total*100:.1f}%)")

    # --- Plot ---
    print(f"\nRendering {OUTPUT_PNG}...")
    polygon = load_polygon(GEOJSON_PATH)
    fig, ax = plt.subplots(figsize=(10, 8))

    geoms = list(polygon.geoms) if polygon.geom_type == "MultiPolygon" else [polygon]
    for poly in geoms:
        lngs, lats = zip(*[(x, y) for x, y in poly.exterior.coords])
        ax.plot(lngs, lats, color="black", linewidth=1.2, zorder=2)

    for cond in ["GREEN", "YELLOW", "ORANGE", "RED"]:
        pts = [r for r in results if r["final_condition"] == cond]
        if pts:
            ax.scatter(
                [p["longitude"] for p in pts],
                [p["latitude"] for p in pts],
                c=CONDITION_COLORS[cond],
                s=14, zorder=3,
                label=f"{CONDITION_LABELS[cond]} ({counts[cond]})",
            )

    ax.set_title(
        f"Morse Reservoir — Wake Surf Conditions\n"
        f"Wind: {wind_speed} mph from {wind_dir_deg}° ({dir_name})"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    print(f"Plot saved to {OUTPUT_PNG}")
    print("\nDone.")


if __name__ == "__main__":
    main()
