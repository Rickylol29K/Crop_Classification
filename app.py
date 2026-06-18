import streamlit as st
import pandas as pd
import geopandas as gpd
import ee
from sklearn.ensemble import RandomForestClassifier
from shapely.geometry import Point

EE_PROJECT_ID = "flash-district-432315-q2"
MONTH_NAMES = {4: "apr", 5: "may", 6: "jun", 7: "jul", 8: "aug", 9: "sep", 10: "oct"}


@st.cache_resource
def load_model():
    features = pd.read_csv("data/field_features.csv")
    numeric_cols = features.select_dtypes("number").columns.tolist()
    numeric_cols.remove("field_id")
    X = features[numeric_cols].fillna(features[numeric_cols].median())
    y = features["crop_class"]
    medians = X.median().to_dict()
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, numeric_cols, medians


@st.cache_resource
def init_ee():
    try:
        ee.Initialize(project=EE_PROJECT_ID)
        return None
    except Exception as e:
        return str(e)


def fetch_satellite_features(lat, lon):
    # 50m buffer around the point, area in hectares
    point_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
    point_utm = point_gdf.to_crs("EPSG:32631")
    buffer_utm = point_utm.buffer(50)
    buffer_wgs84 = buffer_utm.to_crs("EPSG:4326")
    area_ha = float(buffer_utm.area.iloc[0] / 10000)

    coords = [list(c) for c in buffer_wgs84.geometry.iloc[0].exterior.coords]
    ee_polygon = ee.Geometry.Polygon(coords)

    # Sentinel-2 collection for 2025 growing season (same date range as training)
    s2_sr = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate("2025-04-01", "2025-11-01")
        .filterBounds(ee_polygon)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
    )
    s2_clouds = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterDate("2025-04-01", "2025-11-01")
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

    # Build one combined image: monthly mean per index (single API call)
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

    # Assemble feature row
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


st.title("Crop Prediction")
st.caption("Enter field coordinates to predict the crop type")

col1, col2 = st.columns(2)
lat = col1.number_input("Latitude", value=51.55, format="%.6f", step=0.001)
lon = col2.number_input("Longitude", value=5.35, format="%.6f", step=0.001)

if st.button("Predict", type="primary"):
    model, numeric_cols, medians = load_model()

    ee_error = init_ee()
    if ee_error:
        st.error(f"Earth Engine not authenticated. Run `earthengine authenticate` in your terminal.\n\n{ee_error}")
        st.stop()

    with st.spinner("Fetching satellite data from Earth Engine..."):
        try:
            row = fetch_satellite_features(lat, lon)
        except Exception as e:
            st.error(f"Could not fetch satellite data: {e}")
            st.stop()

    if row is None:
        st.info("No satellite data found for this location.")
        st.stop()

    X_pred = pd.DataFrame([row])[numeric_cols].fillna(pd.Series(medians))
    crop = model.predict(X_pred)[0]
    proba = dict(zip(model.classes_, model.predict_proba(X_pred)[0]))

    st.success(f"**Predicted crop: {crop}**")
    top3 = sorted(proba.items(), key=lambda x: x[1], reverse=True)[:3]
    st.caption("Confidence")
    for name, p in top3:
        st.progress(p, text=f"{name}: {p:.0%}")
