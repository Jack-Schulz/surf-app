"""
export_conditions.py
Runs the full final_conditions pipeline and writes conditions_data.json
for consumption by map.html.
"""
import csv
import json
import requests
import numpy as np

FETCH_CSV  = "morse_fetch.csv"
DEPTH_CSV  = "morse_grid_depth.csv"
OUTPUT     = "conditions_data.json"

LAT, LNG   = 40.093, -86.044

DIRECTIONS  = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DIR_DEGREES = [0, 45, 90, 135, 180, 225, 270, 315]

CONDITION_LABELS = {
    "GREEN":  "Glass + Deep (great)",
    "YELLOW": "Calm but Shallow",
    "ORANGE": "Moderate",
    "RED":    "Rough",
}


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


def wind_condition(score):
    if score <= 33: return "Glass"
    if score <= 66: return "Moderate"
    return "Rough"


def depth_tier(depth_ft):
    if depth_ft >= 15: return "Deep"
    if depth_ft >= 8:  return "Moderate Depth"
    return "Shallow"


def final_condition(wind_cond, depth_t):
    if wind_cond == "Rough":   return "RED"
    if depth_t == "Shallow":   return "YELLOW"
    if wind_cond == "Glass":   return "GREEN"
    return "ORANGE"


def load_csv_by_id(path):
    with open(path) as f:
        return {row["point_id"]: row for row in csv.DictReader(f)}


def lake_summary(counts, total, wind_speed, dir_name):
    green_pct  = counts["GREEN"]  / total * 100
    orange_pct = counts["ORANGE"] / total * 100
    red_pct    = counts["RED"]    / total * 100

    if red_pct > 40:
        mood = "choppy out there"
        detail = "wind is up and fetch is long — probably not the day"
    elif red_pct > 15 or orange_pct > 40:
        mood = "mixed conditions"
        detail = "glass on the sheltered end, rougher toward the channel"
    elif green_pct > 70:
        time_of_day = "this morning" if True else "right now"
        mood = f"looking great {time_of_day}"
        detail = "glass conditions on the main channel"
    elif green_pct > 40:
        mood = "decent out there"
        detail = "some glass water available, check the deeper sections"
    else:
        mood = "marginal today"
        detail = "light wind but shallow areas limiting options"

    return f"Morse is {mood} — {detail}."


def main():
    print("Fetching live wind...")
    wind_speed, wind_dir_deg = fetch_weather()
    dir_name = nearest_direction(wind_dir_deg)
    print(f"  {wind_speed} mph from {wind_dir_deg}° ({dir_name})")

    fetch_data = load_csv_by_id(FETCH_CSV)
    depth_data = load_csv_by_id(DEPTH_CSV)
    point_ids  = sorted(pid for pid in fetch_data if pid in depth_data)

    fetch_col     = f"fetch_{dir_name}"
    raw_roughness = np.array([
        float(fetch_data[pid][fetch_col]) * (wind_speed ** 2)
        for pid in point_ids
    ])
    rmin, rmax = raw_roughness.min(), raw_roughness.max()
    normalized = (
        ((raw_roughness - rmin) / (rmax - rmin) * 100).round(1)
        if rmax > rmin else np.zeros_like(raw_roughness)
    )

    points = []
    counts = {"GREEN": 0, "YELLOW": 0, "ORANGE": 0, "RED": 0}

    for pid, score in zip(point_ids, normalized):
        frow     = fetch_data[pid]
        depth_ft = float(depth_data[pid]["depth_ft"])
        w_cond   = wind_condition(score)
        d_tier   = depth_tier(depth_ft)
        f_cond   = final_condition(w_cond, d_tier)
        counts[f_cond] += 1
        points.append({
            "point_id":        pid,
            "latitude":        round(float(frow["latitude"]),  7),
            "longitude":       round(float(frow["longitude"]), 7),
            "depth_ft":        round(depth_ft, 2),
            "roughness_score": float(score),
            "final_condition": f_cond,
        })

    total   = len(points)
    summary = lake_summary(counts, total, wind_speed, dir_name)

    output = {
        "wind_speed":     wind_speed,
        "wind_direction": wind_dir_deg,
        "wind_dir_label": dir_name,
        "summary":        summary,
        "counts":         counts,
        "points":         points,
    }

    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {total} points to {OUTPUT}")
    print(f"Summary: {summary}")
    for cond in ["GREEN", "YELLOW", "ORANGE", "RED"]:
        n = counts[cond]
        print(f"  {CONDITION_LABELS[cond]:<30} {n:>4}  ({n/total*100:.1f}%)")


if __name__ == "__main__":
    main()
