// ============================================================================
// Monthly Gas Time-Series Export — All Cities (2019–2024)
// ============================================================================
// Author      : Ahmed Saad Alnashwan
// Affiliation : King Fahd University of Petroleum and Minerals (KFUPM)
//               MSc Environmental Science & Engineering
// Supervisor  : Dr. Syed Rahman
// Study Area  : Dammam, Al Jubail, Al Khobar — Eastern Province, Saudi Arabia
// Period      : 2019-01-01 to 2024-12-31 (monthly means)
// Project     : linen-epigram-474718-m9
// ============================================================================
// All settings:
//   NO2 : COPERNICUS/S5P/NRTI/L3_NO2 | NO2_column_number_density
//   CO  : COPERNICUS/S5P/NRTI/L3_CO  | CO_column_number_density
//   CH4 : COPERNICUS/S5P/OFFL/L3_CH4 | CH4_column_volume_mixing_ratio_dry_air
// All 3 cities processed in one script (3 export tasks).
//
// Output files (3 CSVs)
// ----------------------
//   Dammam_AllGases_2019_2024.csv
//   Jubail_AllGases_2019_2024.csv
//   Khobar_AllGases_2019_2024.csv
//
// Columns per CSV
// ---------------
//   city, year, month, date, NO2, CO, CH4
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
// 2. DATASETS
//    Kept exactly as in the original script
// ============================================================================

var no2 = ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_NO2")
  .select("NO2_column_number_density");

var co = ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_CO")
  .select("CO_column_number_density");

var ch4 = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CH4")
  .select("CH4_column_volume_mixing_ratio_dry_air");


// ============================================================================
// 3. TIME DIMENSIONS
// ============================================================================

var years  = ee.List.sequence(2019, 2024);
var months = ee.List.sequence(1, 12);


// ============================================================================
// 4. MAP CENTRING
// ============================================================================

Map.setCenter(49.9, 26.4, 8);
Map.setOptions("SATELLITE");


// ============================================================================
// 5. CORE FUNCTION
// ============================================================================

/**
 * Generates a monthly time-series FeatureCollection for one city and
 * registers a Drive CSV export task.
 *
 * For each year-month combination the function computes the spatial mean
 * of NO2, CO, and CH4 over the city geometry and stores the result as
 * one Feature row with columns: city, year, month, date, NO2, CO, CH4.
 *
 * @param {string}               cityName - City label used in filenames.
 * @param {ee.FeatureCollection} cityFC   - City boundary feature collection.
 */
function exportMonthlyTimeSeries(cityName, cityFC) {

  var geometry = cityFC.geometry();

  var results = years.map(function(y) {
    return months.map(function(m) {

      var start = ee.Date.fromYMD(y, m, 1);
      var end   = start.advance(1, "month");

      // NO2 monthly mean over city
      var no2_val = no2.filterDate(start, end).mean()
        .reduceRegion({
          reducer:   ee.Reducer.mean(),
          geometry:  geometry,
          scale:     1000,
          maxPixels: 1e13
        }).get("NO2_column_number_density");

      // CO monthly mean over city
      var co_val = co.filterDate(start, end).mean()
        .reduceRegion({
          reducer:   ee.Reducer.mean(),
          geometry:  geometry,
          scale:     1000,
          maxPixels: 1e13
        }).get("CO_column_number_density");

      // CH4 monthly mean over city
      var ch4_val = ch4.filterDate(start, end).mean()
        .reduceRegion({
          reducer:   ee.Reducer.mean(),
          geometry:  geometry,
          scale:     1000,
          maxPixels: 1e13
        }).get("CH4_column_volume_mixing_ratio_dry_air");

      return ee.Feature(null, {
        "city":  cityName,
        "year":  y,
        "month": m,
        "date":  start.format("YYYY-MM"),
        "NO2":   no2_val,
        "CO":    co_val,
        "CH4":   ch4_val
      });
    });
  });

  var final = ee.FeatureCollection(ee.List(results).flatten());

  print(cityName + " -- monthly time-series (first 10 rows):",
        final.limit(10));

  Export.table.toDrive({
    collection:  final,
    description: cityName + "_AllGases_2019_2024",
    fileFormat:  "CSV"
  });

  print("Export task registered: " + cityName + "_AllGases_2019_2024.csv");
}


// ============================================================================
// 6. RUN FOR ALL CITIES
// ============================================================================

var cityNames = Object.keys(CITIES);

cityNames.forEach(function(cityName) {
  exportMonthlyTimeSeries(cityName, CITIES[cityName]);
});

print("==============================================");
print("All monthly export tasks registered (" +
      cityNames.length + " tasks).");
print("Period : 2019-01 to 2024-12 (72 rows per city)");
print("NO2    : NRTI L3 | NO2_column_number_density");
print("CO     : NRTI L3 | CO_column_number_density");
print("CH4    : OFFL L3 | CH4_column_volume_mixing_ratio_dry_air");
print("Go to Tasks panel and click RUN on each.");
print("==============================================");
