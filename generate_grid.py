import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; remove if you want a popup window
import matplotlib.pyplot as plt
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from pyproj import Transformer

GEOJSON_PATH = "morse.geojson"
CSV_OUTPUT = "morse_grid_points.csv"
PNG_OUTPUT = "morse_grid_points.png"
GRID_SPACING_M = 100

# UTM Zone 16N — covers Indiana (~86°W)
UTM_CRS = "EPSG:32616"
WGS84_CRS = "EPSG:4326"

to_utm = Transformer.from_crs(WGS84_CRS, UTM_CRS, always_xy=True)
to_wgs84 = Transformer.from_crs(UTM_CRS, WGS84_CRS, always_xy=True)


LAT_MIN, LAT_MAX = 39.9, 40.2
LNG_MIN, LNG_MAX = -86.2, -85.8


def load_union(path):
    with open(path) as f:
        data = json.load(f)

    print("\nBounding box per feature:")
    valid_geoms = []
    for i, feat in enumerate(data["features"]):
        geom = shape(feat["geometry"])
        minx, miny, maxx, maxy = geom.bounds  # minx=lng, miny=lat, maxx=lng, maxy=lat
        name = feat.get("properties", {}) or {}
        label = name.get("name") or name.get("id") or f"feature {i}"
        inside = (
            miny >= LAT_MIN and maxy <= LAT_MAX
            and minx >= LNG_MIN and maxx <= LNG_MAX
        )
        status = "OK" if inside else "SKIP (out of bounds)"
        print(f"  [{i}] {label}: lat [{miny:.4f}, {maxy:.4f}]  lng [{minx:.4f}, {maxx:.4f}]  → {status}")
        if inside:
            valid_geoms.append(geom)

    print(f"\nKept {len(valid_geoms)} of {len(data['features'])} features.\n")
    if not valid_geoms:
        raise ValueError("No features passed the bounding box filter — check LAT/LNG bounds.")
    return unary_union(valid_geoms)


def project_polygon(polygon, transformer):
    def transform_coords(coords):
        xs, ys = zip(*coords)
        return list(zip(*transformer.transform(xs, ys)))

    if polygon.geom_type == "Polygon":
        exterior = transform_coords(polygon.exterior.coords)
        interiors = [transform_coords(ring.coords) for ring in polygon.interiors]
        from shapely.geometry import Polygon as ShapelyPolygon
        return ShapelyPolygon(exterior, interiors)
    elif polygon.geom_type == "MultiPolygon":
        from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
        from shapely.geometry import Polygon as ShapelyPolygon
        parts = []
        for poly in polygon.geoms:
            exterior = transform_coords(poly.exterior.coords)
            interiors = [transform_coords(ring.coords) for ring in poly.interiors]
            parts.append(ShapelyPolygon(exterior, interiors))
        return ShapelyMultiPolygon(parts)
    else:
        raise ValueError(f"Unsupported geometry type: {polygon.geom_type}")


def generate_grid_points(polygon_utm):
    minx, miny, maxx, maxy = polygon_utm.bounds
    xs = np.arange(minx, maxx, GRID_SPACING_M)
    ys = np.arange(miny, maxy, GRID_SPACING_M)
    points = []
    for x in xs:
        for y in ys:
            pt = Point(x, y)
            if polygon_utm.contains(pt):
                points.append((x, y))
    return points


def write_csv(points_wgs84, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["point_id", "latitude", "longitude"])
        for i, (lng, lat) in enumerate(points_wgs84):
            writer.writerow([str(i).zfill(5), f"{lat:.7f}", f"{lng:.7f}"])


def plot(polygon_wgs84, points_wgs84, path):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw polygon outline(s)
    geoms = (
        list(polygon_wgs84.geoms)
        if polygon_wgs84.geom_type == "MultiPolygon"
        else [polygon_wgs84]
    )
    for poly in geoms:
        xs, ys = poly.exterior.xy
        ax.plot(xs, ys, color="steelblue", linewidth=1.5)
        for ring in poly.interiors:
            xs, ys = ring.xy
            ax.plot(xs, ys, color="steelblue", linewidth=1)

    # Draw grid points
    lngs = [p[0] for p in points_wgs84]
    lats = [p[1] for p in points_wgs84]
    ax.scatter(lngs, lats, s=4, color="tomato", zorder=3)

    ax.set_title(f"Morse Reservoir — {len(points_wgs84)} grid points @ {GRID_SPACING_M}m spacing")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.show()


def main():
    print("Loading GeoJSON...")
    polygon_wgs84 = load_union(GEOJSON_PATH)

    print("Projecting to UTM Zone 16N...")
    polygon_utm = project_polygon(polygon_wgs84, to_utm)

    print(f"Generating {GRID_SPACING_M}m grid points...")
    points_utm = generate_grid_points(polygon_utm)
    print(f"  {len(points_utm)} points inside polygon")

    print("Projecting points back to WGS84...")
    utm_xs = [p[0] for p in points_utm]
    utm_ys = [p[1] for p in points_utm]
    wgs84_lngs, wgs84_lats = to_wgs84.transform(utm_xs, utm_ys)
    points_wgs84 = list(zip(wgs84_lngs, wgs84_lats))

    print(f"Writing CSV to {CSV_OUTPUT}...")
    write_csv(points_wgs84, CSV_OUTPUT)

    print("Rendering plot...")
    plot(polygon_wgs84, points_wgs84, PNG_OUTPUT)

    print("Done.")


if __name__ == "__main__":
    main()
