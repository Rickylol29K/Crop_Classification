# Crop Classification

This project builds a crop classification dataset for Dutch agricultural fields.

The workflow uses:

- BRP field boundaries and crop labels
- Sentinel-2 satellite imagery from Google Earth Engine
- Monthly NDVI and NDWI features for the 2025 growing season

## Notebooks

1. `notebooks/01_brp_exploration.ipynb`
   - loads and explores the BRP field data
   - filters to a Noord-Brabant study region
   - keeps crop fields above 0.5 hectares
   - groups detailed crop names into broader classes
   - saves `data/sampled_fields.gpkg`

2. `notebooks/02_satellite_extraction.ipynb`
   - loads the sampled fields
   - extracts Sentinel-2 NDVI and NDWI data using Google Earth Engine
   - combines the exported batch CSV files
   - saves `data/field_features.csv`

## Data

The raw BRP GeoPackage is not included in Git because it is too large for GitHub.

To rerun the first notebook, place the BRP file here:

```text
data/brp_2025.gpkg
```

The current derived files are included:

- `data/sampled_fields.gpkg`
- `data/field_features.csv`
- `data/EarthEngineExports/`

## Final Dataset

The final training dataset is:

```text
data/field_features.csv
```

Each row is one field. The target label is `crop_class`, and the satellite features are monthly and seasonal NDVI/NDWI columns.
