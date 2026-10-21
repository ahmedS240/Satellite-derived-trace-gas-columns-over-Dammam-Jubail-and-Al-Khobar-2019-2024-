// ============================================================================
// Nighttime Lights (NTL) Export — VIIRS VCMSLCFG Monthly Composites
// ============================================================================
// Author      : Ahmed Saad Alnashwan
// Affiliation : King Fahd University of Petroleum and Minerals (KFUPM)
//               MSc Environmental Science & Engineering
// Supervisor  : Dr. Syed Rahman
// Study Area  : Dammam, Jubail, Al-Khobar — Eastern Province, Saudi Arabia
// Dataset     : NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG  (avg_rad band)
// Resolution  : 500 m
// Purpose     : Export annual mean NTL rasters (2019 & 2024) per city as
//               GeoTIFF to Google Drive for use as human activity proxies
//               in Sentinel-5P / TROPOMI atmospheric analysis.
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
//   NightLights_Dammam_2019.tif
//   NightLights_Dammam_2024.tif
//   NightLights_Jubail_2019.tif
//   NightLights_Jubail_2024.tif
//   NightLights_Khobar_2019.tif
//   NightLights_Khobar_2024.tif
//
// Notes
// -----
//   * VCMSLCFG = stray-light-corrected, cloud-masked monthly composite.
//   * avg_rad  = average radiance (nW/cm2/sr).
//   * Annual mean is computed from all 12 monthly images per year.
//   * maxPixels set to 1e13 to accommodate large export regions.
//   * CRS set to EPSG:4326 (WGS84) to match TROPOMI outputs.
// ============================================================================


// ============================================================================
// 1. STUDY AREA — City Boundaries
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

/** VIIRS monthly nighttime lights collection. */
var DATASET = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG";

/** Radiance band — average radiance, stray-light corrected. */
var BAND = "avg_rad";

/** Export spatial resolution in metres (native VIIRS DNB ~500 m). */
var EXPORT_SCALE = 500;

/**
 * Analysis years.
 * Add or remove years here — the export loop handles the rest automatically.
 */
var YEARS = [2019, 2024];

/** Visualisation parameters for Map preview. */
var VIS_PARAMS = {
  min:     0,
  max:     60,
  palette: ["000000", "3b0062", "0000ff", "00ffff", "ffff00", "ffffff"]
  // black -> purple -> blue -> cyan -> yellow -> white
};


// ============================================================================
// 3. CORE FUNCTION
// ============================================================================

/**
 * Computes the annual mean NTL radiance for a given city and year,
 * adds a preview layer to the map, and registers a Drive export task.
 *
 * @param {ee.FeatureCollection} cityFC   - City boundary feature collection.
 * @param {string}               cityName - Human-readable city label.
 * @param {number}               year     - Four-digit calendar year.
 */
function exportAnnualNTL(cityFC, cityName, year) {

  // Date range: full calendar year
  var startDate = year + "-01-01";
  var endDate   = year + "-12-31";

  // Compute annual mean from all monthly composites within the year.
  // .select(BAND) before .mean() avoids processing unused bands.
  // .clip(cityFC) constrains output to the city boundary.
  var annualMean = ee.ImageCollection(DATASET)
    .filterDate(startDate, endDate)
    .select(BAND)
    .mean()
    .clip(cityFC);

  // Add preview layer to the map
  Map.addLayer(annualMean, VIS_PARAMS, "NTL " + cityName + " " + year);

  // Register Drive export task
  var taskName = "NightLights_" + cityName + "_" + year;

  Export.image.toDrive({
    image:          annualMean,
    description:    taskName,
    fileNamePrefix: taskName,
    scale:          EXPORT_SCALE,
    region:         cityFC.geometry(),  // explicit .geometry() — more robust
                                        // than passing FeatureCollection directly
    maxPixels:      1e13,
    fileFormat:     "GeoTIFF",
    crs:            "EPSG:4326"         // WGS84 — consistent with TROPOMI outputs
  });

  print("Export task registered: " + taskName);
}


// ============================================================================
// 4. MAP CENTRING
// ============================================================================

Map.setCenter(49.9, 26.4, 9);   // Eastern Province, Saudi Arabia
Map.setOptions("SATELLITE");


// ============================================================================
// 5. EXPORT LOOP — All Cities x All Years
// ============================================================================

// Iterates every city-year combination automatically.
// To add a city: add it to CITIES above.
// To add a year: add it to YEARS above.

var cityNames = Object.keys(CITIES);

cityNames.forEach(function(cityName) {
  YEARS.forEach(function(year) {
    exportAnnualNTL(CITIES[cityName], cityName, year);
  });
});

print("==============================================");
print("All export tasks registered (" +
      cityNames.length * YEARS.length + " tasks total).");
print("Go to the Tasks panel and click RUN on each.");
print("==============================================");
