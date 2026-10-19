var TEST    = true;
var YEAR    = 2025;
var QUARTER = 'NW';

var AOI_WSEN  = [0.0, 0.0, 0.0, 0.0];
var SPLIT_XY  = [0.0, 0.0];
var TEST_WSEN = [0.0, 0.0, 0.0, 0.0];
var EPSG      = 'EPSG:0';
var O = 0.02, W = AOI_WSEN[0], S = AOI_WSEN[1], E = AOI_WSEN[2], N = AOI_WSEN[3];
var X = SPLIT_XY[0], Y = SPLIT_XY[1];
var Q = {NW: [W, Y - O, X + O, N], NE: [X - O, Y - O, E, N],
         SW: [W, S, X + O, Y + O], SE: [X - O, S, E, Y + O]};
var AOI_TEST = ee.Geometry.Rectangle(TEST_WSEN);
var AOI = TEST ? AOI_TEST : ee.Geometry.Rectangle(Q[QUARTER]);
var YEARS = [YEAR];

var CSP   = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var BANDS = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'];
var WIN   = {early: ['04-01', '06-01'], peak: ['06-01', '09-01'],
             late:  ['09-01', '11-01']};

function seasonComposite(year, w) {
  var id = year < 2019 ? 'COPERNICUS/S2_HARMONIZED'
                       : 'COPERNICUS/S2_SR_HARMONIZED';
  var col = ee.ImageCollection(id)
      .filterBounds(AOI)
      .filterDate(year + '-' + WIN[w][0], year + '-' + WIN[w][1])
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
      .linkCollection(CSP, ['cs_cdf']);
  print(year + ' ' + w + ': scenes', col.size());
  var comp = col.map(function (img) {
        var clear = img.select('cs_cdf').gte(0.60);
        return img.select(BANDS).updateMask(clear).resample('bilinear');
      })
      .median().multiply(1e-4);
  var ndvi = comp.normalizedDifference(['B8A', 'B4']).rename('NDVI');
  var ndwi = comp.normalizedDifference(['B3', 'B8A']).rename('NDWI');
  return comp.addBands([ndvi, ndwi]).rename(
      BANDS.concat(['NDVI', 'NDWI']).map(function (b) { return b + '_' + w; }));
}

Map.centerObject(AOI, TEST ? 12 : 9);
Map.addLayer(ee.Image().paint(AOI, 0, 2), {palette: ['yellow']}, 'AOI');

YEARS.forEach(function (year) {
  var stack = ee.Image.cat(['early', 'peak', 'late'].map(function (w) {
    return seasonComposite(year, w);
  })).multiply(1e4).round().toInt16();
  print('Bands ' + year, stack.bandNames());
  Map.addLayer(stack, {bands: ['B8A_peak', 'B4_peak', 'B3_peak'],
                       min: 0, max: 4000}, 'False colour, peak ' + year);
  Map.addLayer(stack.select('NDVI_peak'),
               {min: 0, max: 9000, palette: ['brown', 'yellow', 'green']},
               'NDVI peak ' + year, false);
  Export.image.toDrive({
    image: stack,
    description: TEST ? 's2stack_test_' + year : 's2stack_' + year + '_' + QUARTER,
    folder: 'pivot_stacks',
    region: AOI, crs: EPSG,
    crsTransform: [10, 0, 0, 0, -10, 0],
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'});
});
