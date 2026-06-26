from datetime import date

from flask import Flask, render_template_string, request
import joblib
import pandas as pd
import geopandas as gpd
import ee
from shapely.geometry import Point

EE_PROJECT_ID = "flash-district-432315-q2"
# Same growing-season months used as training features (notebook 02), apr-oct.
MONTH_NAMES = {4: "apr", 5: "may", 6: "jun", 7: "jul", 8: "aug", 9: "sep", 10: "oct"}

# Use the current growing season instead of a fixed year, so predictions stay
# up to date without retraining (the model only keys on month names, not years).
# If we're before this year's season starts (Jan-Mar), use last year's instead.
_today = date.today()
_season_year = _today.year if _today.month >= 4 else _today.year - 1
START_DATE = f"{_season_year}-04-01"
END_DATE = f"{_season_year}-11-01"

app = Flask(__name__)

#Model is trained once in notebook 03 and saved to ../model.pkl (relative to
#notebooks/) / model.pkl (relative to this file). We just load it here, no
#retraining.
model = joblib.load("model.pkl")

#numeric_cols and medians aren't part of the saved model, so we still derive
#them from the same CSV the notebook trained on (this is just reading column
#names and medians, not training anything).
features = pd.read_csv("data/field_features.csv")
numeric_cols = features.select_dtypes("number").columns.tolist()
numeric_cols.remove("field_id")
medians = features[numeric_cols].median().to_dict()

#Requires `earthengine authenticate` to have been run locally beforehand.
#Done once at startup instead of per-request.
try:
    ee.Initialize(project=EE_PROJECT_ID)
    EE_ERROR = None
except Exception as e:
    EE_ERROR = str(e)


def fetch_satellite_features(lat, lon):
    #Re-implements notebook 02's extraction for a single point instead of a field polygon.
    #50m buffer around the point, area in hectares
    point_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
    point_utm = point_gdf.to_crs("EPSG:32631")
    buffer_utm = point_utm.buffer(50)
    buffer_wgs84 = buffer_utm.to_crs("EPSG:4326")
    area_ha = float(buffer_utm.area.iloc[0] / 10000)

    coords = [list(c) for c in buffer_wgs84.geometry.iloc[0].exterior.coords]
    ee_polygon = ee.Geometry.Polygon(coords)

    #Same date range, cloud filtering, and NDVI/NDWI formulas as notebook 02,
    #kept in sync by hand since this isn't a shared module.
    s2_sr = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(START_DATE, END_DATE)
        .filterBounds(ee_polygon)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
    )
    s2_clouds = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterDate(START_DATE, END_DATE)
        .filterBounds(ee_polygon)
    )

    join = ee.Join.saveFirst("clouds")
    join_filter = ee.Filter.equals(leftField="system:index", rightField="system:index")
    joined = ee.ImageCollection(
        join.apply(s2_sr, s2_clouds, join_filter)
    ).filter(ee.Filter.notNull(["clouds"]))

    def mask_and_index(image):
        image = ee.Image(image)
        cloud_prob = ee.Image(image.get("clouds")).select("probability")
        image = image.updateMask(cloud_prob.lt(40))
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
        return image.addBands(ndvi).addBands(ndwi).copyProperties(image, ["system:time_start"])

    collection = joined.map(mask_and_index).select(["NDVI", "NDWI"])

    #Notebook 02 loops reduceRegions per image; here we stack one band per
    #month into a single image so reduceRegion only needs one API call.
    combined = None
    for month_num, month_name in MONTH_NAMES.items():
        monthly = (
            collection.filter(ee.Filter.calendarRange(month_num, month_num, "month"))
            .mean()
            .rename([f"NDVI_{month_name}", f"NDWI_{month_name}"])
        )
        combined = monthly if combined is None else combined.addBands(monthly)

    stats = combined.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=ee_polygon, scale=10
    ).getInfo()

    #Assemble feature row (note: area_ha here is always the ~0.785 ha buffer
    #area, not a real field's actual size like in training)
    row = {"area_ha": area_ha}
    ndvi_vals, ndwi_vals = [], []
    for month_name in MONTH_NAMES.values():
        ndvi = stats.get(f"NDVI_{month_name}")
        ndwi = stats.get(f"NDWI_{month_name}")
        row[f"ndvi_{month_name}"] = ndvi
        row[f"ndwi_{month_name}"] = ndwi
        if ndvi is not None:
            ndvi_vals.append(ndvi)
        if ndwi is not None:
            ndwi_vals.append(ndwi)

    if not ndvi_vals:
        return None

    row["ndvi_peak"] = max(ndvi_vals)
    row["ndvi_season_mean"] = sum(ndvi_vals) / len(ndvi_vals)
    row["ndvi_range"] = max(ndvi_vals) - min(ndvi_vals)
    row["ndwi_peak"] = max(ndwi_vals) if ndwi_vals else None
    row["ndwi_season_mean"] = sum(ndwi_vals) / len(ndwi_vals) if ndwi_vals else None
    row["ndwi_range"] = (max(ndwi_vals) - min(ndwi_vals)) if ndwi_vals else None

    return row


PAGE = """
<!doctype html>
<html>
<head>
  <title>Crop Prediction</title>
  <style>
    body { font-family: sans-serif; max-width: 480px; margin: 40px auto; }
    label { display: block; margin-top: 10px; }
    input { padding: 6px; width: 100%; box-sizing: border-box; }
    button { margin-top: 16px; padding: 8px 16px; }
    .error { color: #b00020; }
    .bar-row { margin: 6px 0; }
    .bar-bg { background: #eee; border-radius: 4px; overflow: hidden; }
    .bar-fill { background: #4caf50; color: white; padding: 2px 6px; white-space: nowrap; }
  </style>
</head>
<body>
  <h1>Crop Prediction</h1>
  <p>Enter field coordinates to predict the crop type</p>

  <form method="post">
    <label>Latitude
      <input type="number" step="0.000001" name="lat" value="{{ lat }}">
    </label>
    <label>Longitude
      <input type="number" step="0.000001" name="lon" value="{{ lon }}">
    </label>
    <button type="submit">Predict</button>
  </form>

  {% if error %}
    <p class="error">{{ error }}</p>
  {% endif %}

  {% if crop %}
    <h2>Predicted crop: {{ crop }}</h2>
    <p>Confidence</p>
    {% for name, p in top3 %}
      <div class="bar-row">
        <div class="bar-bg">
          <div class="bar-fill" style="width: {{ (p * 100) | round(0) }}%;">
            {{ name }}: {{ (p * 100) | round(0) | int }}%
          </div>
        </div>
      </div>
    {% endfor %}
  {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    lat, lon = 51.55, 5.35
    crop, top3, error = None, None, None

    if request.method == "POST":
        lat = float(request.form["lat"])
        lon = float(request.form["lon"])

        if EE_ERROR:
            error = f"Earth Engine not authenticated. Run `earthengine authenticate` in your terminal.\n\n{EE_ERROR}"
        else:
            try:
                row = fetch_satellite_features(lat, lon)
            except Exception as e:
                error = f"Could not fetch satellite data: {e}"
                row = None

            if error is None:
                if row is None:
                    error = "No satellite data found for this location."
                else:
                    X_pred = pd.DataFrame([row])[numeric_cols].fillna(pd.Series(medians))
                    crop = model.predict(X_pred)[0]
                    proba = dict(zip(model.classes_, model.predict_proba(X_pred)[0]))
                    top3 = sorted(proba.items(), key=lambda x: x[1], reverse=True)[:3]

    return render_template_string(PAGE, lat=lat, lon=lon, crop=crop, top3=top3, error=error)


if __name__ == "__main__":
    app.run(debug=True)
