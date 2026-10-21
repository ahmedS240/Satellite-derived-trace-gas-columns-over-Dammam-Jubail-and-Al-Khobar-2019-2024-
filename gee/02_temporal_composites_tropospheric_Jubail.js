// ============================================================================
// Sentinel-5P / TROPOMI — Al Jubail Yearly Gas Export (2019 & 2024)
// ============================================================================
// Author      : Ahmed Saad Alnashwan
// Affiliation : King Fahd University of Petroleum and Minerals (KFUPM)
//               MSc Environmental Science & Engineering
// Supervisor  : Dr. Syed Rahman
// Study Area  : Al Jubail — Eastern Province, Saudi Arabia
// Satellite   : Sentinel-5P TROPOMI (OFFL Level-3)
// Years       : 2019, 2024
// Resolution  : 1113.2 m (0.01 deg -- native TROPOMI L3 grid)
// Project     : linen-epigram-474718-m9
// ============================================================================
//
// QA Strategy (verified against GEE L3 band catalogue)
// ------------------------------------------------------
// NO2  (L3_NO2) : No qa_value in GEE L3. Filter by cloud_fraction <= 0.3.
//                 Standard in TROPOMI NO2 literature (Georgoulias et al. 2019).
// CO   (L3_CO)  : No qa_value or cloud band in GEE L3.
//                 Use ESA pre-filtered OFFL L3 directly.
// CH4  (L3_CH4) : No qa_value in GEE L3.
//                 Use ESA pre-filtered OFFL L3 directly.
//
// Academic Note on Al Jubail
// ---------------------------
// Al Jubail hosts one of the world's largest petrochemical complexes
// (SABIC, Saudi Aramco affiliates). Active gas flaring produces elevated
// CH4, CO, and NO2 signals that are scientifically meaningful -- these
// are NOT noise and should NOT be masked as outliers here (unlike NTL
// where flares can bias radiance means).
// This is an important distinction to state in your thesis methodology.
//
// Thesis methodology statement:
// "TROPOMI OFFL L3 retrievals for Al Jubail were processed without
// suppressing industrial emission signals, as elevated gas concentrations
// from the Jubail Industrial City petrochemical complex represent
// genuine anthropogenic sources of scientific interest in this study."
//
// Output files (6 GeoTIFFs)
// --------------------------
//   NO2_Jubail_2019.tif    NO2_Jubail_2024.tif
//   CO_Jubail_2019.tif     CO_Jubail_2024.tif
//   CH4_Jubail_2019.tif    CH4_Jubail_2024.tif
// ============================================================================


// ============================================================================
// 1. STUDY AREA
//    Project: linen-epigram-474718-m9
// ============================================================================

var Jubail = ee.FeatureCollection(
  "projects/linen-epigram-474718-m9/assets/Al_Jubail"
);

Map.centerObject(Jubail, 9);
Map.setOptions("SATELLITE");
Map.addLayer(Jubail, { color: "000000", width: 2 }, "Al Jubail Boundary");


// ============================================================================
// 2. CONFIGURATION
// ============================================================================

var YEARS        = [2019, 2024];
var EXPORT_SCALE = 1113.2;       // native TROPOMI L3 grid (0.01 deg)

/**
 * Cloud fraction threshold for NO2 quality filtering.
 * <= 0.3 is the standard in TROPOMI NO2 literature.
 */
var NO2_CLOUD_THRESHOLD = 0.3;

/**
 * Gas definitions.
 *
 * qa_type:
 *   "cloud_fraction" -- filter by cloud_fraction <= NO2_CLOUD_THRESHOLD (NO2)
 *   null             -- no QA band; ESA pre-filtered OFFL L3 used directly
 *
 * NO2 band: tropospheric column (not total) for urban/industrial air quality.
 */
var GASES = {

  NO2: {
    dataset:  "COPERNICUS/S5P/OFFL/L3_NO2",
    band:     "tropospheric_NO2_column_number_density",
    qa_type:  "cloud_fraction",
    min:      0,
    max:      0.00015,
    unit:     "mol/m2 (tropospheric column)",
    palette:  ["0000ff", "00ffff", "00ff00", "ffff00", "ff8000", "ff0000"]
  },

  CO: {
    dataset:  "COPERNICUS/S5P/OFFL/L3_CO",
    band:     "CO_column_number_density",
    qa_type:  null,
    min:      0,
    max:      0.04,
    unit:     "mol/m2",
    palette:  ["0000ff", "00ffff", "00ff00", "ffff00", "ff8000", "ff0000"]
  },

  CH4: {
    dataset:  "COPERNICUS/S5P/OFFL/L3_CH4",
    band:     "CH4_column_volume_mixing_ratio_dry_air",
    qa_type:  null,
    min:      1700,
    max:      1900,
    unit:     "ppb (dry air volume mixing ratio)",
    palette:  ["0000ff", "00ffff", "00ff00", "ffff00", "ff8000", "ff0000"]
  }

};


// ============================================================================
// 3. QUALITY FILTER FUNCTION
// ============================================================================

/**
 * Cloud fraction filter for NO2 (only available QA proxy in GEE L3_NO2).
 * Masks pixels where cloud_fraction > NO2_CLOUD_THRESHOLD before .mean().
 */
function applyNO2CloudFilter(image) {
  var cloudMask = image.select("cloud_fraction").lte(NO2_CLOUD_THRESHOLD);
  return image.updateMask(cloudMask);
}


// ============================================================================
// 4. CORE EXPORT FUNCTION
// ============================================================================

/**
 * Computes the quality-filtered annual mean for one gas and one year,
 * adds a map preview layer, and registers a Drive export task.
 *
 * @param {string}               gasName  - Gas label (e.g., "NO2").
 * @param {Object}               gasCfg   - Gas config entry from GASES.
 * @param {number}               year     - Four-digit calendar year.
 * @param {string}               cityName - City label (e.g., "Jubail").
 * @param {ee.FeatureCollection} cityFC   - City boundary feature collection.
 */
function exportAnnualGas(gasName, gasCfg, year, cityName, cityFC) {

  var startDate = year + "-01-01";
  var endDate   = year + "-12-31";

  // Step 1: Filter by date and spatial bounds
  var collection = ee.ImageCollection(gasCfg.dataset)
    .filterDate(startDate, endDate)
    .filterBounds(cityFC);

  // Step 2: Log scene count for thesis data availability table
  print(gasName + " " + cityName + " " + year +
        " -- total scenes:", collection.size());

  // Step 3: Apply quality filter appropriate to each gas
  var filtered;
  if (gasCfg.qa_type === "cloud_fraction") {
    filtered = collection.map(applyNO2CloudFilter);
    print(gasName + ": cloud_fraction filter applied (<= " +
          NO2_CLOUD_THRESHOLD + ")");
  } else {
    filtered = collection;
    print(gasName + ": No QA band in GEE L3 -- using ESA pre-filtered OFFL L3.");
  }

  // Step 4: Select band, compute annual mean, clip to boundary
  var meanImage = filtered
    .select(gasCfg.band)
    .mean()
    .clip(cityFC);

  // Step 5: Map preview
  Map.addLayer(
    meanImage,
    { min: gasCfg.min, max: gasCfg.max, palette: gasCfg.palette },
    gasName + " -- " + cityName + " " + year
  );

  // Step 6: Register Drive export task
  var taskName = gasName + "_" + cityName + "_" + year;

  Export.image.toDrive({
    image:          meanImage,
    description:    taskName,
    fileNamePrefix: taskName,
    scale:          EXPORT_SCALE,
    region:         cityFC.geometry(),
    maxPixels:      1e13,
    fileFormat:     "GeoTIFF",
    crs:            "EPSG:4326"
  });

  print("Registered: " + taskName + "  |  Unit: " + gasCfg.unit);
}


// ============================================================================
// 5. EXPORT LOOP
// ============================================================================

var gasNames = Object.keys(GASES);

gasNames.forEach(function(gasName) {
  YEARS.forEach(function(year) {
    exportAnnualGas(gasName, GASES[gasName], year, "Jubail", Jubail);
  });
});

print("==============================================");
print("All export tasks registered (" +
      gasNames.length * YEARS.length + " tasks).");
print("NO2 : cloud_fraction filter <= " + NO2_CLOUD_THRESHOLD);
print("CO  : ESA pre-filtered OFFL L3 (no QA band in GEE)");
print("CH4 : ESA pre-filtered OFFL L3 (no QA band in GEE)");
print("NO2 band : tropospheric column (not total)");
print("Scale    : " + EXPORT_SCALE + " m (native TROPOMI L3 grid)");
print("NOTE: Industrial signals (flaring, emissions) retained -- not masked.");
print("Go to Tasks panel and click RUN on each.");
print("==============================================");
