import json
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import shape, Point, LineString
from shapely.ops import unary_union
from pyproj import Transformer

GEOJSON_PATH = "morse.geojson"
GRID_CSV = "morse_grid_points.csv"
OUTPUT_CSV = "morse_fetch.csv"
OUTPUT_PNG = "morse_fetch_validation.png"

RAY_LENGTH_M = 10_000
FETCH_CAP_M = 5_000

UTM_CRS = "EPSG:32616"
WGS84_CRS = "EPSG:4326"

to_utm = Transformer.from_crs(WGS84_CRS, UTM_CRS, always_xy=True)
to_wgs84 = Transformer.from_crs(UTM_CRS, WGS84_CRS, always_xy=True)

# Compass directions as (dx, dy) unit vectors in UTM (North = +Y, East = +X)
DIRECTIONS = {
    "N":  ( 0.0,               1.0             ),
    "NE": ( math.sqrt(2) / 2,  math.sqrt(2) / 2),
    "E":  ( 1.0,               0.0             ),
    "SE": ( math.sqrt(2) / 2, -math.sqrt(2) / 2),
    "S":  ( 0.0,              -1.0             ),
    "SW": (-math.sqrt(2) / 2, -math.sqrt(2) / 2),
    "W":  (-1.0,               0.0             ),
    "NW": (-math.sqrt(2) / 2,  math.sqrt(2) / 2),
}

LAT_MIN, LAT_MAX = 39.9, 40.2
LNG_MIN, LNG_MAX = -86.2, -85.8


def load_polygon(path):
    with open(path) as f:
        data = json.load(f)
    valid = []
    for feat in data["features"]:
        g = shape(feat["geometry"])
        minx, miny, maxx, maxy = g.bounds
        if miny >= LAT_MIN and maxy <= LAT_MAX and minx >= LNG_MIN and maxx <= LNG_MAX:
            valid.append(g)
    if not valid:
        raise ValueError("No features passed the bounding box filter.")
    return unary_union(valid)


def project_polygon_to_utm(polygon):
    """Project a WGS84 Shapely polygon to UTM using pyproj."""
    from shapely.ops import transform as shp_transform
    return shp_transform(lambda x, y: to_utm.transform(x, y), polygon)


def load_grid(path):
    points = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append({
                "point_id": row["point_id"],
                "lat": float(row["latitude"]),
                "lng": float(row["longitude"]),
            })
    return points


def fetch_distance(origin_utm, direction, boundary):
    """
    Cast a ray from origin_utm in (dx, dy) direction and return the distance
    in meters to the first intersection with boundary. Returns FETCH_CAP_M if
    no intersection is found within RAY_LENGTH_M.
    """
    ox, oy = origin_utm
    dx, dy = direction
    end = (ox + dx * RAY_LENGTH_M, oy + dy * RAY_LENGTH_M)
    ray = LineString([(ox, oy), end])

    intersection = ray.intersection(boundary)

    if intersection.is_empty:
        return FETCH_CAP_M

    # Collect all intersection points
    if intersection.geom_type == "Point":
        candidates = [intersection]
    elif intersection.geom_type in ("MultiPoint", "GeometryCollection", "LineString", "MultiLineString"):
        candidates = list(getattr(intersection, "geoms", [intersection]))
        # Flatten any non-Point geometries to their representative point
        flat = []
        for g in candidates:
            if g.geom_type == "Point":
                flat.append(g)
            else:
                # Use interpolate(0) to get the start of a line segment
                flat.append(g.interpolate(0))
        candidates = flat
    else:
        candidates = [intersection]

    origin_pt = Point(ox, oy)
    min_dist = min(origin_pt.distance(pt) for pt in candidates if not pt.is_empty)
    return round(min_dist, 2)


def main():
    print("Loading polygon...")
    polygon_wgs84 = load_polygon(GEOJSON_PATH)
    polygon_utm = project_polygon_to_utm(polygon_wgs84)
    boundary_utm = polygon_utm.boundary  # LinearRing or MultiLineString

    print("Loading grid points...")
    grid = load_grid(GRID_CSV)
    print(f"  {len(grid)} points loaded")

    print("Calculating fetch for 8 directions...")
    results = []
    dir_keys = list(DIRECTIONS.keys())

    for i, pt in enumerate(grid):
        ux, uy = to_utm.transform(pt["lng"], pt["lat"])
        row = {
            "point_id": pt["point_id"],
            "latitude": pt["lat"],
            "longitude": pt["lng"],
        }
        for name in dir_keys:
            row[f"fetch_{name}"] = fetch_distance((ux, uy), DIRECTIONS[name], boundary_utm)
        results.append(row)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(grid)} done")

    print(f"  {len(grid)}/{len(grid)} done")

    print(f"Writing {OUTPUT_CSV}...")
    fieldnames = ["point_id", "latitude", "longitude"] + [f"fetch_{d}" for d in dir_keys]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("Rendering validation plot (fetch_N)...")
    fetch_n = np.array([r["fetch_N"] for r in results])
    lats = [r["latitude"] for r in results]
    lngs = [r["longitude"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw polygon outline
    geoms = list(polygon_wgs84.geoms) if polygon_wgs84.geom_type == "MultiPolygon" else [polygon_wgs84]
    for poly in geoms:
        xs, ys = poly.exterior.xy
        ax.plot(xs, ys, color="black", linewidth=1.2, zorder=2)
        for ring in poly.interiors:
            xs, ys = ring.xy
            ax.plot(xs, ys, color="black", linewidth=0.8, zorder=2)

    sc = ax.scatter(lngs, lats, c=fetch_n, cmap="RdYlBu_r", s=12,
                    vmin=fetch_n.min(), vmax=fetch_n.max(), zorder=3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Fetch N (meters)")

    ax.set_title(f"Morse Reservoir — Fetch N  |  min={fetch_n.min():.0f}m  max={fetch_n.max():.0f}m")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    print(f"Plot saved to {OUTPUT_PNG}")

    # Summary stats
    print("\nFetch summary (meters):")
    print(f"  {'Direction':<6}  {'Min':>7}  {'Mean':>7}  {'Max':>7}")
    for d in dir_keys:
        vals = np.array([r[f"fetch_{d}"] for r in results])
        print(f"  {d:<6}  {vals.min():>7.0f}  {vals.mean():>7.0f}  {vals.max():>7.0f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
