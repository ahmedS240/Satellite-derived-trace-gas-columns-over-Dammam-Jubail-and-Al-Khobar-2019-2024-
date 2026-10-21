"""
A01 - Data coverage diagnostics and analytical-grid geometry
(responds to postdoc comments on Sections 2.1, 2.3 and Fig. 2: GADM level, coverage, 5 km grid in EPSG:4326)

Outputs (../02_outputs):
    A01_valid_months_by_year.csv   number of valid monthly city means per city, gas and year
    A01_boundary_polygons.csv      source, administrative ID/level and area of each study-extent polygon
    A01_grid_cell_geometry.csv     actual cell size of the GEE coveringGrid and share of each cell inside the polygon
"""
import json
import numpy as np
import pandas as pd
import shapefile  # pyshp
from shapely.geometry import shape
import shapely
import pyproj
from common import ROOT, OUT, CITIES, GASES, HOTSPOT, load_monthly

# ---------------------------------------------------------------- 1. coverage
rows = []
for c in CITIES:
    df = load_monthly(c)
    for gas in GASES + ["NTL"]:
        for y, g in df.groupby("year"):
            rows.append({"City": c, "Variable": gas, "Year": y,
                         "Valid_months": int(g[gas].notna().sum()), "Total_months": len(g)})
cov = pd.DataFrame(rows)
wide = cov.pivot_table(index=["City", "Variable"], columns="Year", values="Valid_months").astype(int)
wide["Total_2019_2024"] = wide.sum(axis=1)
wide.to_csv(OUT / "A01_valid_months_by_year.csv")
print(wide)

# ---------------------------------------------------------------- 2. polygons
SHP = {"Dammam": "Dammam", "Jubail": "Al_Jubail", "Khobar": "AlKhobarfull"}
to_utm = pyproj.Transformer.from_crs(4326, 32639, always_xy=True)
polys, prow = {}, []
for c, name in SHP.items():
    r = shapefile.Reader(str(ROOT / "gadm41_SAU_shp" / name))
    fields = [f[0] for f in r.fields[1:]]
    rec = dict(zip(fields, r.records()[0]))
    geom = shape(r.shapes()[0].__geo_interface__)
    polys[c] = geom
    utm = shapely.transform(geom, lambda xy: np.column_stack(to_utm.transform(xy[:, 0], xy[:, 1])))
    if "GID_2" in rec:
        src, ident, lvl, nm = "GADM 4.1", rec["GID_2"], f"level 2 ({rec['ENGTYPE_2']})", rec["NAME_2"]
    else:
        src, ident, lvl, nm = "OpenStreetMap", rec.get("F_id"), f"admin_level {rec.get('admin_leve')}", rec.get("name_en")
    prow.append({"City": c, "Shapefile": name, "Source": src, "ID": ident, "Admin_level": lvl,
                 "Name": nm, "Area_km2": round(utm.area / 1e6, 1)})
pdf = pd.DataFrame(prow)
pdf.to_csv(OUT / "A01_boundary_polygons.csv", index=False)
print(pdf.to_string())

# ---------------------------------------------------------------- 3. grid cells
grow = []
for c in CITIES:
    h = pd.read_csv(HOTSPOT[c])
    h = h.dropna(subset=["NO2", "CO", "CH4", "NTL", "POP"])
    for _, r in h.iterrows():
        cell = shape(json.loads(r[".geo"]))
        minx, miny, maxx, maxy = cell.bounds
        latc = (miny + maxy) / 2
        w_km = (maxx - minx) * 111.320 * np.cos(np.radians(latc))
        h_km = (maxy - miny) * 110.574
        cell_utm = shapely.transform(cell, lambda xy: np.column_stack(to_utm.transform(xy[:, 0], xy[:, 1])))
        inter = cell.intersection(polys[c])
        inter_utm = shapely.transform(inter, lambda xy: np.column_stack(to_utm.transform(xy[:, 0], xy[:, 1]))) if not inter.is_empty else None
        frac = (inter_utm.area / cell_utm.area) if inter_utm is not None else 0.0
        grow.append({"City": c, "Grid_ID": r["Grid_ID"], "dlon_deg": maxx - minx, "dlat_deg": maxy - miny,
                     "width_km": w_km, "height_km": h_km, "area_km2": cell_utm.area / 1e6,
                     "fraction_inside_polygon": frac})
g = pd.DataFrame(grow)
g.to_csv(OUT / "A01_grid_cell_geometry.csv", index=False)
summ = g.groupby("City").agg(n_cells=("Grid_ID", "size"), dlon_deg=("dlon_deg", "mean"),
                             width_km=("width_km", "mean"), height_km=("height_km", "mean"),
                             area_km2=("area_km2", "mean"),
                             median_fraction_inside=("fraction_inside_polygon", "median"),
                             cells_lt50pct_inside=("fraction_inside_polygon", lambda s: int((s < 0.5).sum())))
summ.to_csv(OUT / "A01_grid_cell_geometry_summary.csv")
print(summ.round(3))
