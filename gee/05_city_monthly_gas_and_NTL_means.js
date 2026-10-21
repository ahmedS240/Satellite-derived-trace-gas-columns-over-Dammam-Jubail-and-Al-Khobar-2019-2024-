// ============================================================================
// Monthly City-Level Proxy Export — All Cities (2019–2024)
// ============================================================================
// Author      : Ahmed Saad Alnashwan
// Affiliation : King Fahd University of Petroleum and Minerals (KFUPM)
//               MSc Environmental Science & Engineering
// Supervisor  : Dr. Syed Rahman
// Study Area  : Dammam, Al Jubail, Al Khobar — Eastern Province, Saudi Arabia
// Period      : 2019-01 to 2024-12 (monthly means)
// Project     : linen-epigram-474718-m9
// ============================================================================
// Settings:
//   NO2 : COPERNICUS/S5P/NRTI/L3_NO2 | NO2_column_number_density
//   CO  : COPERNICUS/S5P/NRTI/L3_CO  | CO_column_number_density
//   CH4 : COPERNICUS/S5P/OFFL/L3_CH4 | CH4_column_volume_mixing_ratio_dry_air
//   NTL : NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG | avg_rad
//   POP : WorldPop/GP/100m/pop | year=2020 | scale=100
// All 3 cities processed in one script (3 export tasks).
//
// Output files (3 CSVs)
// ----------------------
//   Dammam_Final.csv
//   Al_Jubail_Final.csv
//   Al_Khobar_Final.csv
//
// Columns per CSV
// ---------------
//   city, year, month, date, NO2, CO, CH4, NTL, POP
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
  Dammam: {
    fc:   ee.FeatureCollection("projects/linen-epigram-474718-m9/assets/Dammam"),
    name: "Dammam"
  },
  Al_Jubail: {
    fc:   ee.FeatureCollection("projects/linen-epigram-474718-m9/assets/Al_Jubail"),
    name: "Al_Jubail"
  },
  Al_Khobar: {
    fc:   ee.FeatureCollection("projects/linen-epigram-474718-m9/assets/AlKhobarfull"),
    name: "Al_Khobar"
  }
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

var ntl = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG")
  .select("avg_rad");

// WorldPop 2020 — kept exactly as original
var pop = ee.ImageCollection("WorldPop/GP/100m/pop")
  .filter(ee.Filter.eq("year", 2020))
  .first();


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
 * For each year-month combination computes spatial mean of NO2, CO, CH4,
 * NTL over the city geometry and a static POP mean (WorldPop 2020).
 * Each row: city, year, month, date, NO2, CO, CH4, NTL, POP.
 *
 * @param {string}               cityName - City label used in output filename.
 * @param {ee.FeatureCollection} cityFC   - City boundary feature collection.
 */
function exportCityTimeSeries(cityName, cityFC) {

  var geometry = cityFC.geometry();

  var results = years.map(function(y) {
    return months.map(function(m) {

      var start = ee.Date.fromYMD(y, m, 1);
      var end   = start.advance(1, "month");

      // Monthly mean images for gases and NTL
      var no2_img = no2.filterDate(start, end).mean();
      var co_img  = co .filterDate(start, end).mean();
      var ch4_img = ch4.filterDate(start, end).mean();
      var ntl_img = ntl.filterDate(start, end).mean();

      // Spatial mean over city geometry for all gases + NTL in one call
      var stats = ee.Image.cat([
        no2_img.rename("NO2"),
        co_img .rename("CO"),
        ch4_img.rename("CH4"),
        ntl_img.rename("NTL")
      ]).reduceRegion({
        reducer:   ee.Reducer.mean(),
        geometry:  geometry,
        scale:     1000,
        maxPixels: 1e13
      });

      // Population — static (WorldPop 2020), same value every month
      var pop_val = pop.reduceRegion({
        reducer:   ee.Reducer.mean(),
        geometry:  geometry,
        scale:     100,
        maxPixels: 1e13
      }).get("population");

      return ee.Feature(null, {
        "city":  cityName,
        "year":  y,
        "month": m,
        "date":  start.format("YYYY-MM"),
        "NO2":   stats.get("NO2"),
        "CO":    stats.get("CO"),
        "CH4":   stats.get("CH4"),
        "NTL":   stats.get("NTL"),
        "POP":   pop_val
      });
    });
  });

  var final = ee.FeatureCollection(ee.List(results).flatten());

  print(cityName + " -- preview (first 5 rows):", final.limit(5));

  Export.table.toDrive({
    collection:  final,
    description: cityName + "_Final",
    fileFormat:  "CSV"
  });

  print("Export task registered: " + cityName + "_Final.csv");
}


// ============================================================================
// 6. RUN FOR ALL CITIES
// ============================================================================

var cityKeys = Object.keys(CITIES);

cityKeys.forEach(function(key) {
  exportCityTimeSeries(CITIES[key].name, CITIES[key].fc);
});

print("==============================================");
print("All city proxy export tasks registered (" +
      cityKeys.length + " tasks).");
print("Period : 2019-01 to 2024-12 (72 rows per city)");
print("NO2    : NRTI L3 | NO2_column_number_density");
print("CO     : NRTI L3 | CO_column_number_density");
print("CH4    : OFFL L3 | CH4_column_volume_mixing_ratio_dry_air");
print("NTL    : VCMCFG  | avg_rad");
print("POP    : WorldPop 2020 | 100m");
print("Go to Tasks panel and click RUN on each.");
print("==============================================");
