// ============================================================================
// Spatial Hotspot Table — All Cities (Grid-Based Feature Extraction)
// ============================================================================
// Author      : Ahmed Saad Alnashwan
// Affiliation : King Fahd University of Petroleum and Minerals (KFUPM)
//               MSc Environmental Science & Engineering
// Supervisor  : Dr. Syed Rahman
// Study Area  : Dammam, Al Jubail, Al Khobar — Eastern Province, Saudi Arabia
// Period      : 2019-01-01 to 2024-12-31
// Grid Scale  : 5000 m (5 km spatial units)
// Project     : linen-epigram-474718-m9
// ============================================================================
//
// Description
// -----------
// Divides each city boundary into a 5 km regular grid and extracts the
// period mean for each variable per grid cell using reduceRegions().
// Output CSVs are used as input to the K-Means spatial clustering script.
//
// Fix applied vs original script
// --------------------------------
// Project path corrected:
//   sharp-airway-490210-p6  ->  linen-epigram-474718-m9
// All other settings kept exactly as in the original script:
//   NO2 : COPERNICUS/S5P/NRTI/L3_NO2 | NO2_column_number_density
//   CO  : COPERNICUS/S5P/NRTI/L3_CO  | CO_column_number_density
//   CH4 : COPERNICUS/S5P/OFFL/L3_CH4 | CH4_column_volume_mixing_ratio_dry_air
//   NTL : NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG | avg_rad
//   POP : JRC/GHSL/P2023A/GHS_POP/2020 | population_count
//
// Output files (3 CSVs)
// ----------------------
//   Hotspot_Dammam_FINAL.csv
//   Hotspot_Jubail_FINAL.csv
//   Hotspot_Khobar_FINAL.csv
//
// Columns per CSV
// ---------------
//   Grid_ID, Longitude, Latitude, NO2, CO, CH4, NTL, POP
//
// Usage
// -----
//   1. Paste into GEE Code Editor and click Run.
//   2. Go to Tasks panel -> click RUN on each of the 3 export tasks.
// ============================================================================


// ============================================================================
// 1. STUDY AREAS
//    Project: linen-epigram-474718-m9
// ============================================================================

var CITIES = {
  Dammam: ee.FeatureCollection(
    "projects/linen-epigram-474718-m9/assets/Dammam"
  ),
  Jubail: ee.FeatureCollection(
    "projects/linen-epigram-474718-m9/assets/Al_Jubail"
  ),
  Khobar: ee.FeatureCollection(
    "projects/linen-epigram-474718-m9/assets/AlKhobarfull"
  )
};


// ============================================================================
// 2. CONFIGURATION
// ============================================================================

var START_DATE   = "2019-01-01";
var END_DATE     = "2024-12-31";
var GRID_SCALE   = 5000;   // 5 km grid cell size (metres)
var REDUCE_SCALE = 1000;   // reduceRegions sampling scale (metres)
var TILE_SCALE   = 4;      // prevents memory overflow on large areas


// ============================================================================
// 3. SHARED DATASETS
//    Kept exactly as in the original script — only clipped per city in loop
// ============================================================================

// NO2 — NRTI L3 | total column number density
var no2_global = ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_NO2")
  .filterDate(START_DATE, END_DATE)
  .mean()
  .select("NO2_column_number_density");

// CO — NRTI L3 | column number density
var co_global = ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_CO")
  .filterDate(START_DATE, END_DATE)
  .mean()
  .select("CO_column_number_density");

// CH4 — OFFL L3 | dry air volume mixing ratio
var ch4_global = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CH4")
  .filterDate(START_DATE, END_DATE)
  .mean()
  .select("CH4_column_volume_mixing_ratio_dry_air");

// NTL — VCMCFG monthly | average radiance
var ntl_global = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG")
  .filterDate(START_DATE, END_DATE)
  .mean()
  .select("avg_rad");

// POP — JRC GHSL 2020 | population count
var pop_global = ee.Image("JRC/GHSL/P2023A/GHS_POP/2020")
  .select("population_count");


// ============================================================================
// 4. MAP CENTRING
// ============================================================================

Map.setCenter(49.9, 26.4, 8);
Map.setOptions("SATELLITE");


// ============================================================================
// 5. CORE FUNCTION
// ============================================================================

/**
 * Clips all datasets to the city boundary, builds the 5 km grid,
 * runs reduceRegions, appends Grid_ID and centroid coordinates,
 * and registers a Drive CSV export task.
 *
 * @param {string}               cityName - City label used in filenames.
 * @param {ee.FeatureCollection} cityFC   - City boundary feature collection.
 */
function exportHotspotTable(cityName, cityFC) {

  // Clip all datasets to this city boundary
  var no2 = no2_global.clip(cityFC).rename("NO2");
  var co  = co_global .clip(cityFC).rename("CO");
  var ch4 = ch4_global.clip(cityFC).rename("CH4");
  var ntl = ntl_global.clip(cityFC).rename("NTL");
  var pop = pop_global .clip(cityFC).rename("POP");

  // Combined multi-band image
  var dataset = ee.Image.cat([no2, co, ch4, ntl, pop]);

  // 5 km regular grid over the city boundary
  var grid = cityFC.geometry().coveringGrid({
    proj:  ee.Projection("EPSG:4326"),
    scale: GRID_SCALE
  });

  print(cityName + " -- grid cell count:", grid.size());

  // Map preview
  Map.addLayer(cityFC, { color: "000000", width: 2 }, cityName + " Boundary");
  Map.addLayer(grid,   { color: "555555" },            cityName + " Grid");

  // Extract mean value per grid cell
  var table = dataset.reduceRegions({
    collection: grid,
    reducer:    ee.Reducer.mean(),
    scale:      REDUCE_SCALE,
    tileScale:  TILE_SCALE
  });

  // Append Grid_ID and centroid coordinates
  var table_final = table.map(function(f) {
    var centroid = f.geometry().centroid(1);
    return f.set({
      "Grid_ID":   f.id(),
      "Longitude": centroid.coordinates().get(0),
      "Latitude":  centroid.coordinates().get(1)
    });
  });

  print(cityName + " -- preview (first 10 rows):", table_final.limit(10));

  // Export to Google Drive
  Export.table.toDrive({
    collection:  table_final,
    description: "Hotspot_" + cityName + "_FINAL",
    fileFormat:  "CSV"
  });

  print("Export task registered: Hotspot_" + cityName + "_FINAL.csv");
}


// ============================================================================
// 6. RUN FOR ALL CITIES
// ============================================================================

var cityNames = Object.keys(CITIES);

cityNames.forEach(function(cityName) {
  exportHotspotTable(cityName, CITIES[cityName]);
});

print("==============================================");
print("All hotspot export tasks registered (" +
      cityNames.length + " tasks).");
print("Grid scale   : " + GRID_SCALE + " m");
print("Reduce scale : " + REDUCE_SCALE + " m");
print("Period       : " + START_DATE + " to " + END_DATE);
print("Go to Tasks panel and click RUN on each.");
print("==============================================");
