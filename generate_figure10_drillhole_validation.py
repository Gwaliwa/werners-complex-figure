"""
Generate Figure 10: Spatial validation map showing RTP magnetic background,
detected/ranked anomaly peaks, selected inversion target, and drillhole locations.

This version uses your existing file:
    peak_drillhole_ranked_summary.csv

Run:
    python generate_figure10_drillhole_validation_fixed.py

Outputs:
    figure10_drillhole_validation.png
    figure10_drillhole_validation.pdf
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import plotting_extent
from pyproj import Transformer, CRS


# ============================================================
# PATHS — EDIT ONLY IF YOUR FOLDERS CHANGE
# ============================================================

RTP_RASTER = Path(
    "/Users/gmashaka/Documents/Project Personal/papers/weners/validation-drillholes/whole SA/data-aus/rtp_clip_drillhole_area.tif"
)

# IMPORTANT: your folder does NOT have peaks.csv.
# It has peak_drillhole_ranked_summary.csv, so we use that.
PEAKS_CSV = Path(
    "/Users/gmashaka/Documents/Project Personal/papers/weners/validation-drillholes/cliped/peak_drillhole_ranked_summary.csv"
)

DRILLHOLES_CSV = Path(
    "/Users/gmashaka/Documents/Project Personal/papers/weners/validation-drillholes/whole SA/data-aus/drillhole_reference_depths.csv"
)

OUTPUT_PNG = Path(
    "/Users/gmashaka/Documents/Project Personal/papers/weners/validation-drillholes/figure101_drillhole_validation.png"
)
OUTPUT_PDF = OUTPUT_PNG.with_suffix(".pdf")

# Set to None to use the first row in peak_drillhole_ranked_summary.csv.
# Or set manually, for example:
# SELECTED_PEAK_INDEX = 14
#SELECTED_PEAK_INDEX = None
SELECTED_PEAK_INDEX = 4

# Validation buffer radius around selected peak.
BUFFER_KM = 2.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def first_existing_column(df, candidates):
    """Return first matching column name, case-insensitive."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_raster(path):
    """Load raster array, CRS, transform, and extent."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        crs = src.crs
        transform = src.transform
        extent = plotting_extent(src)
    return arr, crs, transform, extent


def get_peak_xy(peaks, raster_crs, transform):
    """
    Return peaks dataframe with peak_x and peak_y in raster CRS.

    Supports:
    - peak_lon / peak_lat
    - lon / lat
    - longitude / latitude
    - x / y
    - peak_col / peak_row or col / row
    """
    peaks = peaks.copy()

    lon_col = first_existing_column(peaks, ["peak_lon", "lon", "longitude", "LONGITUDE", "x"])
    lat_col = first_existing_column(peaks, ["peak_lat", "lat", "latitude", "LATITUDE", "y"])

    if lon_col and lat_col:
        x = pd.to_numeric(peaks[lon_col], errors="coerce")
        y = pd.to_numeric(peaks[lat_col], errors="coerce")

        # If values look like longitude/latitude, transform to raster CRS.
        looks_lonlat = x.between(-180, 180).all() and y.between(-90, 90).all()

        if looks_lonlat and raster_crs is not None:
            raster_epsg = CRS.from_user_input(raster_crs).to_epsg()
            if raster_epsg not in [4326, 4283]:
                transformer = Transformer.from_crs("EPSG:4283", raster_crs, always_xy=True)
                px, py = transformer.transform(x.values, y.values)
                peaks["peak_x"] = px
                peaks["peak_y"] = py
            else:
                peaks["peak_x"] = x
                peaks["peak_y"] = y
        else:
            peaks["peak_x"] = x
            peaks["peak_y"] = y

        return peaks

    # Pixel row/column fallback
    row_col = first_existing_column(peaks, ["peak_row", "row", "r"])
    col_col = first_existing_column(peaks, ["peak_col", "col", "column", "c"])

    if row_col and col_col:
        rows = pd.to_numeric(peaks[row_col], errors="coerce").values
        cols = pd.to_numeric(peaks[col_col], errors="coerce").values
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
        peaks["peak_x"] = xs
        peaks["peak_y"] = ys
        return peaks

    raise ValueError(
        "Could not identify peak coordinates. Expected peak_lon/peak_lat, lon/lat, "
        "x/y, or peak_row/peak_col columns."
    )


def get_drillhole_xy(dh, raster_crs):
    """
    Return drillhole dataframe with dh_x and dh_y in raster CRS.

    Supports:
    - LONGITUDE_GDA2020 / LATITUDE_GDA2020
    - longitude / latitude
    - lon / lat
    - EASTING_GDA2020 / NORTHING_GDA2020 in EPSG:7853
    """
    dh = dh.copy()

    lon_col = first_existing_column(
        dh,
        ["LONGITUDE_GDA2020", "LONGITUDE", "longitude", "lon", "x"]
    )
    lat_col = first_existing_column(
        dh,
        ["LATITUDE_GDA2020", "LATITUDE", "latitude", "lat", "y"]
    )

    if lon_col and lat_col:
        lon = pd.to_numeric(dh[lon_col], errors="coerce")
        lat = pd.to_numeric(dh[lat_col], errors="coerce")

        transformer = Transformer.from_crs("EPSG:4283", raster_crs, always_xy=True)
        x, y = transformer.transform(lon.values, lat.values)

        dh["dh_x"] = x
        dh["dh_y"] = y
        return dh

    east_col = first_existing_column(dh, ["EASTING_GDA2020", "EASTING", "easting"])
    north_col = first_existing_column(dh, ["NORTHING_GDA2020", "NORTHING", "northing"])

    if east_col and north_col:
        east = pd.to_numeric(dh[east_col], errors="coerce")
        north = pd.to_numeric(dh[north_col], errors="coerce")

        transformer = Transformer.from_crs("EPSG:7853", raster_crs, always_xy=True)
        x, y = transformer.transform(east.values, north.values)

        dh["dh_x"] = x
        dh["dh_y"] = y
        return dh

    raise ValueError(
        "Could not identify drillhole coordinates. Expected lon/lat or "
        "EASTING_GDA2020/NORTHING_GDA2020 columns."
    )


def add_buffer_circle(ax, x, y, radius_km, raster_crs):
    """Add validation buffer circle if raster CRS is projected in meters."""
    crs_obj = CRS.from_user_input(raster_crs)
    if crs_obj.is_projected:
        radius_m = radius_km * 1000.0
        circle = plt.Circle(
            (x, y),
            radius_m,
            fill=False,
            linestyle="--",
            linewidth=1.8,
            color="black",
        )
        ax.add_patch(circle)
        return True
    return False


# ============================================================
# MAIN
# ============================================================

def main():
    # ---------- check files ----------
    if not RTP_RASTER.exists():
        raise FileNotFoundError(f"RTP raster not found: {RTP_RASTER}")

    if not PEAKS_CSV.exists():
        raise FileNotFoundError(f"Peaks/ranked CSV not found: {PEAKS_CSV}")

    if not DRILLHOLES_CSV.exists():
        raise FileNotFoundError(f"Drillholes CSV not found: {DRILLHOLES_CSV}")

    # ---------- load ----------
    arr, raster_crs, transform, extent = load_raster(RTP_RASTER)
    peaks_raw = pd.read_csv(PEAKS_CSV)
    drillholes_raw = pd.read_csv(DRILLHOLES_CSV)

    print("\nPeak CSV columns:")
    print(list(peaks_raw.columns))

    print("\nDrillhole CSV columns:")
    print(list(drillholes_raw.columns))

    # ---------- convert coordinates ----------
    peaks = get_peak_xy(peaks_raw, raster_crs, transform)
    drillholes = get_drillhole_xy(drillholes_raw, raster_crs)

    peaks = peaks.dropna(subset=["peak_x", "peak_y"])
    drillholes = drillholes.dropna(subset=["dh_x", "dh_y"])

    if peaks.empty:
        raise ValueError("No valid peak coordinates found after conversion.")

    if drillholes.empty:
        raise ValueError("No valid drillhole coordinates found after conversion.")

    # ---------- selected peak ----------
    peak_index_col = first_existing_column(peaks, ["peak_index", "index", "id"])

    selected_peak_index = SELECTED_PEAK_INDEX

    # If None, use first row of ranked summary
    if selected_peak_index is None:
        if peak_index_col is not None:
            selected_peak_index = int(peaks.iloc[0][peak_index_col])
        else:
            selected_peak_index = None

    if selected_peak_index is not None and peak_index_col is not None:
        match = peaks[
            pd.to_numeric(peaks[peak_index_col], errors="coerce") == selected_peak_index
        ]
        if match.empty:
            warnings.warn(
                f"Selected peak {selected_peak_index} not found. Using first row instead."
            )
            selected_row = peaks.iloc[0]
        else:
            selected_row = match.iloc[0]
    else:
        warnings.warn("No peak_index column found. Using first row as selected target.")
        selected_row = peaks.iloc[0]

    sx = float(selected_row["peak_x"])
    sy = float(selected_row["peak_y"])

    # ---------- clip points to raster extent ----------
    xmin, xmax, ymin, ymax = extent

    in_dh = drillholes[
        (drillholes["dh_x"] >= xmin) &
        (drillholes["dh_x"] <= xmax) &
        (drillholes["dh_y"] >= ymin) &
        (drillholes["dh_y"] <= ymax)
    ]

    in_peaks = peaks[
        (peaks["peak_x"] >= xmin) &
        (peaks["peak_x"] <= xmax) &
        (peaks["peak_y"] >= ymin) &
        (peaks["peak_y"] <= ymax)
    ]

    print(f"\nPeaks inside raster extent: {len(in_peaks)}")
    print(f"Drillholes inside raster extent: {len(in_dh)}")
    print(f"Selected peak index: {selected_peak_index}")

    # ---------- raster display limits ----------
    vmin, vmax = np.nanpercentile(arr, [2, 98])

    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=(8.5, 8.0), dpi=300)

    im = ax.imshow(
        arr,
        extent=extent,
        origin="upper",
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
    )

    # Drillholes underneath
    ax.scatter(
        in_dh["dh_x"],
        in_dh["dh_y"],
        s=4,
        c="saddlebrown",
        alpha=0.45,
        linewidths=0,
        label="Drillholes",
    )

    # Peaks above drillholes
    ax.scatter(
        in_peaks["peak_x"],
        in_peaks["peak_y"],
        s=24,
        c="yellow",
        edgecolors="black",
        linewidths=0.4,
        label="Ranked anomaly peaks",
    )

    # Selected target
    ax.scatter(
        [sx],
        [sy],
        s=220,
        facecolors="none",
        edgecolors="red",
        linewidths=2.4,
        label="Selected inversion target",
    )

    # Buffer circle
    added_circle = add_buffer_circle(ax, sx, sy, BUFFER_KM, raster_crs)
    if added_circle:
        ax.plot(
            [],
            [],
            "k--",
            linewidth=1.8,
            label=f"{BUFFER_KM:g} km validation radius",
        )

    ax.set_title("Spatial validation of selected anomaly target", fontsize=13)
    ax.set_xlabel("Map X")
    ax.set_ylabel("Map Y")
    ax.legend(loc="lower right", frameon=True, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("RTP magnetic value")

    fig.tight_layout()

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved:")
    print(OUTPUT_PNG)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
