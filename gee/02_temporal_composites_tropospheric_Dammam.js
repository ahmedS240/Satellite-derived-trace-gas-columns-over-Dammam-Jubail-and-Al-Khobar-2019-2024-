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
// Resolution  : 1113.2 m (0.01 deg -- native TROPOMI L3 grid)
// Project     : linen-epigram-474718-m9
// ============================================================================
//
// QA Bands — Verified against GEE Band Catalogue
// -----------------------------------------------
// L3_NO2  available bands:
//   NO2_column_number_density, tropospheric_NO2_column_number_density,
//   stratospheric_NO2_column_number_density, NO2_slant_column_number_density,
//   tropopause_pressure, absorbing_aerosol_index, cloud_fraction,
//   sensor_altitude, sensor/solar azimuth/zenith angles
//   --> NO qa_value band. Use cloud_fraction as quality proxy.
//
// L3_CO   available bands:
//   CO_column_number_density, H2O_column_number_density, cloud_height,
//   sensor_altitude, sensor/solar azimuth/zenith angles
//   --> NO qa_value band. No cloud fraction band either.
//       OFFL L3 CO is pre-filtered by ESA at L2->L3 stage.
//
// L3_CH4  available bands:
//   CH4_column_volume_mixing_ratio_dry_air,
//   CH4_column_volume_mixing_ratio_dry_air_bias_corrected,
//   CH4_column_volume_mixing_ratio_dry_air_uncertainty,
//   aerosol_height, aerosol_optical_depth,
//   sensor/solar azimuth/zenith angles
//   --> NO qa_value band. OFFL L3 CH4 is pre-filtered by ESA.
//
// Quality Control Strategy (per-product)
// ----------------------------------------
// NO2: Filter by cloud_fraction <= 0.3 (standard in TROPOMI literature).
//      Removes cloudy scenes while retaining enough observations for a
//      robust annual mean. Some studies use <= 0.5 for more data.
//      Reference: Georgoulias et al. (2019), doi:10.1038/s41598-019-39749-9
//
// CO:  No QA band available in GEE L3_CO. OFFL product is pre-screened
//      by ESA during L2->L3 aggregation. Use collection directly.
//      Statement for thesis: "CO retrievals from the TROPOMI OFFL L3
//      product were used without additional masking, as quality screening
//      is applied during ESA's L2-to-L3 gridding process."
//
// CH4: Same as CO -- no QA band, pre-filtered by ESA.
//
// Output files (6 GeoTIFFs)
// --------------------------
//   NO2_Dammam_2019.tif    NO2_Dammam_2024.tif
//   CO_Dammam_2019.tif     CO_Dammam_2024.tif
//   CH4_Dammam_2019.tif    CH4_Dammam_2024.tif
// ============================================================================


// ============================================================================
// 1. STUDY AREA
//    Project: linen-epigram-474718-m9
// ============================================================================

var Dammam = ee.FeatureCollection(
  "projects/linen-epigram-474718-m9/assets/Dammam"
);

Map.centerObject(Dammam, 8);
Map.setOptions("SATELLITE");
Map.addLayer(Dammam, { color: "000000", width: 2 }, "Dammam Boundary");


// ============================================================================
// 2. CONFIGURATION
// ============================================================================

var YEARS        = [2019, 2024];
var EXPORT_SCALE = 1113.2;      // native TROPOMI L3 grid (0.01 deg)

/**
 * Cloud fraction threshold for NO2 quality filtering.
 * <= 0.3 is the standard in TROPOMI NO2 literature.
 * This is the ONLY feasible QA proxy available in the GEE L3_NO2 collection.
 */
var NO2_CLOUD_THRESHOLD = 0.3;

/**
 * Gas definitions.
 *
 * qa_type options:
 *   "cloud_fraction" -- filter by cloud_fraction band (NO2 only)
 *   null             -- no QA band available; ESA pre-filtered L3 used directly
 *
 * NO2 band: tropospheric column (not total) for urban air quality.
 */
var GASES = {

  NO2: {
    dataset:  "COPERNICUS/S5P/OFFL/L3_NO2",
    band:     "tropospheric_NO2_column_number_density",
    qa_type:  "cloud_fraction",   // filter by cloud_fraction <= NO2_CLOUD_THRESHOLD
    min:      0,
    max:      0.00015,
    unit:     "mol/m2 (tropospheric column)",
    palette:  ["0000ff", "00ffff", "00ff00", "ffff00", "ff8000", "ff0000"]
  },

  CO: {
    dataset:  "COPERNICUS/S5P/OFFL/L3_CO",
    band:     "CO_column_number_density",
    qa_type:  null,               // no QA band -- ESA pre-filtered OFFL L3
    min:      0,
    max:      0.04,
    unit:     "mol/m2",
    palette:  ["0000ff", "00ffff", "00ff00", "ffff00", "ff8000", "ff0000"]
  },

  CH4: {
    dataset:  "COPERNICUS/S5P/OFFL/L3_CH4",
    band:     "CH4_column_volume_mixing_ratio_dry_air",
    qa_type:  null,               // no QA band -- ESA pre-filtered OFFL L3
    min:      1700,
    max:      1900,
    unit:     "ppb (dry air volume mixing ratio)",
    palette:  ["0000ff", "00ffff", "00ff00", "ffff00", "ff8000", "ff0000"]
  }

};


// ============================================================================
// 3. QUALITY FILTER FUNCTIONS
// ============================================================================

/**
 * Cloud fraction filter for NO2 (the only available QA proxy in L3_NO2).
 * Masks pixels where cloud_fraction > NO2_CLOUD_THRESHOLD.
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
 * @param {string}               cityName - City label (e.g., "Dammam").
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
    // NO2: cloud_fraction is the only available QA proxy in L3_NO2
    filtered = collection.map(applyNO2CloudFilter);
    print(gasName + ": cloud_fraction filter applied (<= " +
          NO2_CLOUD_THRESHOLD + ")");
  } else {
    // CO, CH4: no QA band in GEE -- OFFL L3 is ESA pre-filtered
    filtered = collection;
    print(gasName + ": No QA band in GEE L3 product. " +
          "Using ESA pre-filtered OFFL L3 directly.");
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
    exportAnnualGas(gasName, GASES[gasName], year, "Dammam", Dammam);
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
print("Go to Tasks panel and click RUN on each.");
print("==============================================");
