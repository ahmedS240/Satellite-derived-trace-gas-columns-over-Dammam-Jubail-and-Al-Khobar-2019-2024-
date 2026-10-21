// ============================================================================
// Sentinel-5P / TROPOMI — Dammam Yearly Gas Export (2019 & 2024)
// ============================================================================
// Author      : Ahmed Saad Alnashwan
// Affiliation : King Fahd University of Petroleum and Minerals (KFUPM)
//               MSc Environmental Science & Engineering
// Supervisor  : Dr. Syed Rahman
// Study Area  : Dammam — Eastern Province, Saudi Arabia
// Satellite   : Sentinel-5P TROPOMI (OFFL Level-3)
// Years       : 2019, 2024
// Resolution  : 2000 m
// Purpose     : Export annual mean gas rasters per year for change detection
//               and spatio-temporal comparison between 2019 and 2024.
// ============================================================================
//
// Usage
// -----
//   1. Open Google Earth Engine Code Editor (code.earthengine.google.com).
//   2. Paste this script and click Run.
//   3. In the Tasks panel (top-right) click RUN next to each export task.
//   4. Files will appear in your Google Drive root folder.
//
// Output files (6 GeoTIFFs)
// --------------------------
//   NO2_Dammam_2019.tif
//   NO2_Dammam_2024.tif
//   CO_Dammam_2019.tif
//   CO_Dammam_2024.tif
//   CH4_Dammam_2019.tif
//   CH4_Dammam_2024.tif
//
// Notes
// -----
//   * OFFL = Offline processed (higher quality than NRTI near-real-time).
//   * .mean() computes the pixel-wise annual mean for each calendar year.
//   * filterBounds() pre-filters spatially before .mean() for efficiency.
//   * maxPixels set to 1e13 to accommodate large export regions.
//   * CRS set to EPSG:4326 (WGS84) — consistent with NTL exports.
// ============================================================================


// ============================================================================
// 1. STUDY AREA — Dammam Boundary
//    Project: linen-epigram-474718-m9
// ============================================================================

var Dammam = ee.FeatureCollection(
  "projects/linen-epigram-474718-m9/assets/Dammam"
);

Map.centerObject(Dammam, 8);
Map.setOptions("SATELLITE");
Map.addLayer(Dammam, { color: "000000", width: 2 }, "Dammam Boundary");


// ============================================================================
// 2. CONFIGURATION — Analysis Years, Scale & Gas Definitions
// ============================================================================

/**
 * Analysis years exported individually for change-detection comparison.
 * Add or remove years here — the export loop handles the rest automatically.
 */
var YEARS = [2019, 2024];

/** Export spatial resolution in metres (TROPOMI native ~3.5x5.5 km -> 2 km). */
var EXPORT_SCALE = 2000;

/**
 * Gas definitions.
 * Each entry contains:
 *   dataset  — GEE ImageCollection ID
 *   band     — band name to select
 *   min/max  — visualisation range
 *   palette  — colour ramp for Map preview
 */
var GASES = {

  NO2: {
    dataset: "COPERNICUS/S5P/OFFL/L3_NO2",
    band:    "NO2_column_number_density",
    min:     0,
    max:     0.00015,
    palette: ["0000ff", "00ffff", "00ff00", "ffff00", "ff8000", "ff0000"]
    // blue -> cyan -> green -> yellow -> orange -> red
  },

  CO: {
    dataset: "COPERNICUS/S5P/OFFL/L3_CO",
    band:    "CO_column_number_density",
    min:     0,
    max:     0.04,
    palette: ["0000ff", "00ffff", "00ff00", "ffff00", "ff8000", "ff0000"]
  },

  CH4: {
    dataset: "COPERNICUS/S5P/OFFL/L3_CH4",
    band:    "CH4_column_volume_mixing_ratio_dry_air",
    min:     1700,
    max:     1900,
    palette: ["0000ff", "00ffff", "00ff00", "ffff00", "ff8000", "ff0000"]
  }

};


// ============================================================================
// 3. CORE FUNCTION
// ============================================================================

/**
 * Computes the annual mean concentration for one gas over one year,
 * adds a preview layer to the map, and registers a Drive export task.
 *
 * @param {string}               gasName  - Gas label (e.g., "NO2").
 * @param {Object}               gasCfg   - Gas configuration from GASES dict.
 * @param {number}               year     - Four-digit calendar year.
 * @param {string}               cityName - City label (e.g., "Dammam").
 * @param {ee.FeatureCollection} cityFC   - City boundary feature collection.
 */
function exportAnnualGas(gasName, gasCfg, year, cityName, cityFC) {

  var startDate = year + "-01-01";
  var endDate   = year + "-12-31";

  // Compute annual mean — filterBounds reduces collection before .mean()
  var meanImage = ee.ImageCollection(gasCfg.dataset)
    .filterDate(startDate, endDate)
    .filterBounds(cityFC)
    .select(gasCfg.band)
    .mean()
    .clip(cityFC);

  // Map preview layer
  Map.addLayer(
    meanImage,
    { min: gasCfg.min, max: gasCfg.max, palette: gasCfg.palette },
    gasName + " — " + cityName + " " + year
  );

  // Register Drive export task
  var taskName = gasName + "_" + cityName + "_" + year;

  Export.image.toDrive({
    image:          meanImage,
    description:    taskName,
    fileNamePrefix: taskName,
    scale:          EXPORT_SCALE,
    region:         cityFC.geometry(),  // explicit .geometry() — more robust
                                        // than passing FeatureCollection directly
    maxPixels:      1e13,
    fileFormat:     "GeoTIFF",
    crs:            "EPSG:4326"         // WGS84 — consistent with NTL exports
  });

  print("Export task registered: " + taskName);
}


// ============================================================================
// 4. EXPORT LOOP — All Gases x All Years
// ============================================================================

// Iterates every gas-year combination automatically.
// To add a gas  : add it to GASES above.
// To add a year : add it to YEARS above.

var gasNames = Object.keys(GASES);

gasNames.forEach(function(gasName) {
  YEARS.forEach(function(year) {
    exportAnnualGas(gasName, GASES[gasName], year, "Dammam", Dammam);
  });
});

print("==============================================");
print("All export tasks registered (" +
      gasNames.length * YEARS.length + " tasks total).");
print("Go to the Tasks panel and click RUN on each.");
print("==============================================");
