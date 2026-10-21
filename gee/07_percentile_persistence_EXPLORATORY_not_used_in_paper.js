// ============================================================================
// Phase B — Monthly 90th-Percentile Hotspot Masks and Persistence
// ============================================================================
// Author      : Ahmed Saad Alnashwan
// Affiliation : King Fahd University of Petroleum and Minerals (KFUPM)
//               MSc Environmental Science & Engineering
// Supervisor  : Dr. Syed Masiur Rahman
// Study Area  : Dammam, Al Jubail, Al Khobar — Eastern Province, Saudi Arabia
// Period      : 2019-01 to 2024-12  (72 monthly steps)
// Project     : linen-epigram-474718-m9
// ============================================================================
//
// WHAT THIS IMPLEMENTS
// --------------------
// The Phase B rule described in the methodology chapter, which was documented
// but not present in the original export scripts:
//
//   1. For each city, each gas, each of the 72 months, compute the monthly
//      mean image clipped to the city boundary.
//   2. Compute the 90th percentile of THAT month's distribution inside THAT
//      city  ->  P90(c,g,t).
//   3. Binary hotspot mask:  H(i) = 1 if X(i) >= P90, else 0.
//   4. Persistence = (months a cell was a hotspot) / (months with valid data),
//      expressed as a percentage.
//   5. Persistent hotspot = persistence > 75 %.
//   6. Reduce the persistence surface onto the same 5 km grid used elsewhere
//      and export as CSV, joinable to the existing Hotspot_<City>_FINAL.csv
//      on Grid_ID.
//
// Months whose valid-pixel coverage falls below MIN_VALID_PIXELS are excluded
// from both the numerator and the denominator, matching the coverage rule in
// the methodology chapter.
//
// OUTPUT (3 CSVs, one per city)
// ------------------------------
//   Persistence_<City>_2019_2024.csv
//   Columns: Grid_ID, Longitude, Latitude,
//            NO2_persist_pct, CO_persist_pct, CH4_persist_pct,
//            NO2_persistent, CO_persistent, CH4_persistent
//
// USAGE
// -----
//   1. Paste into the GEE Code Editor and click Run.
//   2. Open the Tasks panel and click RUN on each of the 3 export tasks.
//   3. Join the result to Hotspot_<City>_FINAL.csv on Grid_ID.
//
// NOTE: this script has not been executed in this environment — run it once in
// the Code Editor and check the printed P90 values before exporting.
// ============================================================================


// ============================================================================
// 1. STUDY AREAS
// ============================================================================

var CITIES = {
  Dammam: ee.FeatureCollection("projects/linen-epigram-474718-m9/assets/Dammam"),
  Jubail: ee.FeatureCollection("projects/linen-epigram-474718-m9/assets/Al_Jubail"),
  Khobar: ee.FeatureCollection("projects/linen-epigram-474718-m9/assets/AlKhobarfull")
};


// ============================================================================
// 2. CONFIGURATION
// ============================================================================

var START_DATE       = "2019-01-01";
var N_MONTHS         = 72;      // 6 years x 12
var PERCENTILE       = 90;      // hotspot threshold percentile
var PERSISTENCE_PCT  = 75;      // > 75 % of valid months => persistent hotspot
var GRID_SCALE       = 5000;    // 5 km analysis grid (metres)
var REDUCE_SCALE     = 1000;    // sampling scale for reducers (metres)
var TILE_SCALE       = 4;       // memory guard
var MIN_VALID_PIXELS = 20;      // months below this are dropped entirely

/** Gas definitions — same collections and bands as the other Phase A/B scripts. */
var GASES = {
  NO2: { dataset: "COPERNICUS/S5P/NRTI/L3_NO2", band: "NO2_column_number_density" },
  CO:  { dataset: "COPERNICUS/S5P/NRTI/L3_CO",  band: "CO_column_number_density"  },
  CH4: { dataset: "COPERNICUS/S5P/OFFL/L3_CH4", band: "CH4_column_volume_mixing_ratio_dry_air" }
};

var months = ee.List.sequence(0, N_MONTHS - 1);


// ============================================================================
// 3. CORE — monthly percentile masks and persistence for one city / one gas
// ============================================================================

/**
 * Builds the persistence surface for one gas over one city.
 *
 * For every month the 90th percentile is recomputed from that month's own
 * within-city distribution, so the threshold is relative and seasonally
 * adaptive — this is what makes three cities of very different absolute
 * pollution level comparable.
 *
 * @param {ee.FeatureCollection} cityFC  City boundary.
 * @param {Object}               gasCfg  Entry from GASES.
 * @param {string}               gasName Gas label, used for band naming.
 * @return {ee.Image} Single-band image of persistence in percent.
 */
function persistenceSurface(cityFC, gasCfg, gasName) {

  var geom = cityFC.geometry();

  // One 0/1 mask per month, tagged with whether the month is usable.
  var maskCollection = ee.ImageCollection(months.map(function (m) {

    var start = ee.Date(START_DATE).advance(m, "month");
    var end   = start.advance(1, "month");

    var monthly = ee.ImageCollection(gasCfg.dataset)
      .filterDate(start, end)
      .filterBounds(cityFC)
      .select(gasCfg.band)
      .mean()
      .clip(cityFC);

    // Percentile AND valid-pixel count in a single pass.
    var stats = monthly.reduceRegion({
      reducer: ee.Reducer.percentile([PERCENTILE])
                 .combine({ reducer2: ee.Reducer.count(), sharedInputs: true }),
      geometry:  geom,
      scale:     REDUCE_SCALE,
      maxPixels: 1e13,
      bestEffort: true
    });

    var count = ee.Number(stats.get(gasCfg.band + "_count"));

    // When a month has no valid data the percentile is null, so substitute a
    // value nothing can exceed and mark the month invalid.
    var p90 = ee.Number(ee.Algorithms.If(
      count.gt(0),
      stats.get(gasCfg.band + "_p" + PERCENTILE),
      1e30
    ));

    return monthly.gte(p90)
                  .rename("hot")
                  .set("valid",   count.gte(MIN_VALID_PIXELS) ? 1 : 0)
                  .set("valid_n", count)
                  .set("p90",     p90)
                  .set("month",   start.format("YYYY-MM"));
  }));

  // Keep only months that cleared the coverage rule.
  var usable = maskCollection.filter(ee.Filter.gte("valid_n", MIN_VALID_PIXELS));

  print(gasName + " — usable months:", usable.size());

  // Per-pixel: how often was this cell a hotspot, out of the months where it
  // had valid data? .count() is per-pixel, so cloud gaps do not bias the ratio.
  return usable.sum()
               .divide(usable.count())
               .multiply(100)
               .rename(gasName + "_persist_pct");
}


// ============================================================================
// 4. PER-CITY EXPORT
// ============================================================================

/**
 * Builds the 5 km grid, attaches persistence for all three gases, flags
 * persistent hotspots, and registers a Drive export task.
 *
 * @param {string}               cityName City label used in filenames.
 * @param {ee.FeatureCollection} cityFC   City boundary.
 */
function exportPersistence(cityName, cityFC) {

  print("=== " + cityName + " ===");

  // Persistence surface per gas, stacked into one multi-band image.
  var bands = Object.keys(GASES).map(function (g) {
    return persistenceSurface(cityFC, GASES[g], g);
  });
  var stack = ee.Image.cat(bands);

  // Same 5 km covering grid as the original hotspot export, so Grid_ID matches.
  var grid = cityFC.geometry().coveringGrid({
    proj:  ee.Projection("EPSG:4326"),
    scale: GRID_SCALE
  });

  var table = stack.reduceRegions({
    collection: grid,
    reducer:    ee.Reducer.mean(),
    scale:      REDUCE_SCALE,
    tileScale:  TILE_SCALE
  });

  // Grid_ID, centroid coordinates, and the > 75 % persistent-hotspot flags.
  var table_final = table.map(function (f) {
    var centroid = f.geometry().centroid(1);
    var no2 = ee.Number(ee.Algorithms.If(f.get("NO2_persist_pct"), f.get("NO2_persist_pct"), 0));
    var co  = ee.Number(ee.Algorithms.If(f.get("CO_persist_pct"),  f.get("CO_persist_pct"),  0));
    var ch4 = ee.Number(ee.Algorithms.If(f.get("CH4_persist_pct"), f.get("CH4_persist_pct"), 0));
    return f.set({
      "Grid_ID":        f.id(),
      "Longitude":      centroid.coordinates().get(0),
      "Latitude":       centroid.coordinates().get(1),
      "NO2_persistent": no2.gt(PERSISTENCE_PCT),
      "CO_persistent":  co.gt(PERSISTENCE_PCT),
      "CH4_persistent": ch4.gt(PERSISTENCE_PCT)
    });
  });

  print(cityName + " — preview (first 10 rows):", table_final.limit(10));
  print(cityName + " — cells with NO2 persistence > " + PERSISTENCE_PCT + "%:",
        table_final.filter(ee.Filter.eq("NO2_persistent", 1)).size());

  Export.table.toDrive({
    collection:  table_final,
    description: "Persistence_" + cityName + "_2019_2024",
    fileFormat:  "CSV",
    selectors: ["Grid_ID", "Longitude", "Latitude",
                "NO2_persist_pct", "CO_persist_pct", "CH4_persist_pct",
                "NO2_persistent", "CO_persistent", "CH4_persistent"]
  });

  print("Export task registered: Persistence_" + cityName + "_2019_2024.csv");
}


// ============================================================================
// 5. RUN FOR ALL CITIES
// ============================================================================

Map.setCenter(49.9, 26.4, 8);
Map.setOptions("SATELLITE");

Object.keys(CITIES).forEach(function (cityName) {
  exportPersistence(cityName, CITIES[cityName]);
});

print("==============================================");
print("Percentile        : " + PERCENTILE + "th, recomputed per city per month");
print("Persistence rule  : > " + PERSISTENCE_PCT + "% of valid months");
print("Months evaluated  : " + N_MONTHS + " (2019-01 to 2024-12)");
print("Grid scale        : " + GRID_SCALE + " m");
print("Min valid pixels  : " + MIN_VALID_PIXELS + " per month");
print("Go to the Tasks panel and click RUN on each export.");
print("==============================================");
