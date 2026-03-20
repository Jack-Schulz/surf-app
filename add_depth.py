import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import unary_union

BATHY_CSV      = "bathymetry-5ft_Morse-Reservoir_2016.csv"
GRID_CSV       = "morse_grid_points.csv"
GEOJSON_PATH   = "morse.geojson"
OUTPUT_CSV     = "morse_grid_depth.csv"
OUTPUT_PNG     = "morse_depth_validation.png"

SPILLWAY_FT    = 809.44          # NAVD88 elevation of spillway crest
BATHY_CRS      = "EPSG:2965"    # Indiana State Plane East, US Survey Feet
WGS84_CRS      = "EPSG:4326"

LAT_MIN, LAT_MAX = 39.9, 40.2
LNG_MIN, LNG_MAX = -86.2, -85.8

# Project grid WGS84 → EPSG:2965 for KDTree (avoids projecting 2.5M bath points)
wgs84_to_2965 = Transformer.from_crs(WGS84_CRS, BATHY_CRS, always_xy=True)

DEPTH_CATEGORIES = [
    ("Very Shallow (<4 ft)",  "#D32F2F",  lambda d: d <  4),
    ("Shallow (4–8 ft)",      "#FBC02D",  lambda d: 4  <= d < 8),
    ("Moderate (8–15 ft)",    "#81D4FA",  lambda d: 8  <= d < 15),
    ("Deep (15+ ft)",         "#1565C0",  lambda d: d >= 15),
]


def load_bathymetry(path):
    print(f"Loading bathymetry ({path})...")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    easting   = data[:, 0]
    northing  = data[:, 1]
    elevation = data[:, 2]
    depth     = SPILLWAY_FT - elevation
    print(f"  {len(depth):,} points loaded")
    return easting, northing, depth


def load_grid(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "point_id": row["point_id"],
                "lat": float(row["latitude"]),
                "lng": float(row["longitude"]),
            })
    return rows


def load_polygon(path):
    with open(path) as f:
        data = json.load(f)
    valid = [
        shape(feat["geometry"]) for feat in data["features"]
        if (lambda b: b[1] >= LAT_MIN and b[3] <= LAT_MAX
                      and b[0] >= LNG_MIN and b[2] <= LNG_MAX)(
            shape(feat["geometry"]).bounds
        )
    ]
    return unary_union(valid)


def main():
    # --- Load bathymetry ---
    easting, northing, depth = load_bathymetry(BATHY_CSV)

    # --- Build KDTree in EPSG:2965 space ---
    print("Building KDTree...")
    bath_coords = np.column_stack([easting, northing])
    tree = KDTree(bath_coords)

    # --- Load grid points and project to EPSG:2965 ---
    print("Loading grid points...")
    grid = load_grid(GRID_CSV)
    grid_lngs = np.array([p["lng"] for p in grid])
    grid_lats = np.array([p["lat"] for p in grid])
    grid_e, grid_n = wgs84_to_2965.transform(grid_lngs, grid_lats)
    grid_coords = np.column_stack([grid_e, grid_n])

    # --- Nearest-neighbor lookup ---
    print("Finding nearest bathymetry point for each grid point...")
    _, indices = tree.query(grid_coords)
    grid_depths = depth[indices]

    # --- Write output CSV ---
    print(f"Writing {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["point_id", "latitude", "longitude", "depth_ft"])
        for pt, d in zip(grid, grid_depths):
            writer.writerow([pt["point_id"], f"{pt['lat']:.7f}", f"{pt['lng']:.7f}", f"{d:.2f}"])

    # --- Summary statistics ---
    print("\nDepth summary:")
    print(f"  Min depth   : {grid_depths.min():.2f} ft")
    print(f"  Max depth   : {grid_depths.max():.2f} ft")
    print(f"  Mean depth  : {grid_depths.mean():.2f} ft")
    print(f"\n  {'Category':<25}  {'Count':>5}  {'%':>6}")
    for label, _, test in DEPTH_CATEGORIES:
        count = int(np.sum([test(d) for d in grid_depths]))
        print(f"  {label:<25}  {count:>5}  {count/len(grid_depths)*100:>5.1f}%")

    # --- Validation plot ---
    print(f"\nRendering {OUTPUT_PNG}...")
    polygon = load_polygon(GEOJSON_PATH)
    fig, ax = plt.subplots(figsize=(10, 8))

    geoms = list(polygon.geoms) if polygon.geom_type == "MultiPolygon" else [polygon]
    for poly in geoms:
        lngs, lats = zip(*[(x, y) for x, y in poly.exterior.coords])
        ax.plot(lngs, lats, color="black", linewidth=1.2, zorder=2)

    for label, color, test in DEPTH_CATEGORIES:
        mask = np.array([test(d) for d in grid_depths])
        if mask.any():
            ax.scatter(
                grid_lngs[mask], grid_lats[mask],
                c=color, s=12, label=f"{label} ({mask.sum()})", zorder=3,
            )

    ax.set_title("Morse Reservoir — Depth (ft NAVD88, spillway = 809.44 ft)")
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
