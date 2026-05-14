import pandas as pd
import geopandas as gpd
import numpy as np
from pyproj import Geod
from pathlib import Path

# ================= PATHS =================
CLIPPED = Path("/Users/gmashaka/Documents/Project Personal/papers/weners/validation-drillholes/cliped")
BASE = Path("/Users/gmashaka/Documents/Project Personal/papers/weners/validation-drillholes/whole SA/data-aus")

# ================= LOAD DATA =================
peaks = pd.read_csv(CLIPPED / "peaks.csv")
dh = pd.read_csv(BASE / "drillhole_reference_depths.csv")

# ================= CONVERT DRILLHOLES CRS =================
gdf = gpd.GeoDataFrame(
    dh,
    geometry=gpd.points_from_xy(dh["EASTING_GDA2020"], dh["NORTHING_GDA2020"]),
    crs="EPSG:7853"
).to_crs("EPSG:4283")

gdf["lon"] = gdf.geometry.x
gdf["lat"] = gdf.geometry.y

# ================= PEAK COLUMN DETECTION =================
lon_col = "x" if "x" in peaks.columns else "lon"
lat_col = "y" if "y" in peaks.columns else "lat"

# ================= DISTANCE ENGINE =================
geod = Geod(ellps="GRS80")
rows = []

# ================= LOOP =================
for _, p in peaks.iterrows():

    peak_lon = float(p[lon_col])
    peak_lat = float(p[lat_col])

    _, _, dist_m = geod.inv(
        np.full(len(gdf), peak_lon),
        np.full(len(gdf), peak_lat),
        gdf["lon"].values,
        gdf["lat"].values
    )

    tmp = gdf.copy()
    tmp["distance_km"] = dist_m / 1000

    near1 = tmp[tmp["distance_km"] <= 1]
    near2 = tmp[tmp["distance_km"] <= 2]
    near5 = tmp[tmp["distance_km"] <= 5]

    rows.append({
        "peak_index": int(p["peak_index"]),
        "peak_lon": peak_lon,
        "peak_lat": peak_lat,
        "peak_value": float(p["value"]) if "value" in p else np.nan,
        "quality_score": float(p["quality_score"]) if "quality_score" in p else np.nan,
        "nearest_distance_km": float(tmp["distance_km"].min()),
        "n_drillholes_1km": len(near1),
        "n_drillholes_2km": len(near2),
        "n_drillholes_5km": len(near5),
        "z_true_median_5km": float(near5["z_true"].median()) if len(near5) else np.nan,
        "z_true_max_5km": float(near5["z_true"].max()) if len(near5) else np.nan,
    })

# ================= SUMMARY =================
summary = pd.DataFrame(rows)

summary = summary.sort_values(
    ["n_drillholes_1km", "n_drillholes_2km", "n_drillholes_5km", "nearest_distance_km", "quality_score"],
    ascending=[False, False, False, True, False]
)

# ================= SAVE =================
summary.to_csv(CLIPPED / "peak_drillhole_ranked_summary.csv", index=False)

# ================= PRINT =================
print("\n===== PEAK RANKING =====\n")
print(summary.to_string(index=False))

# ================= BEST PEAK =================
best = summary.iloc[0]

print("\n===== RECOMMENDED PEAK =====")
print("Peak:", int(best["peak_index"]))
print("Nearest distance (km):", round(best["nearest_distance_km"], 3))
print("Drillholes within 1 km:", int(best["n_drillholes_1km"]))
print("Drillholes within 2 km:", int(best["n_drillholes_2km"]))
print("Drillholes within 5 km:", int(best["n_drillholes_5km"]))

print("\nSaved file:")
print(CLIPPED / "peak_drillhole_ranked_summary.csv")
