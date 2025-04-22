import geojson
from shapely.geometry import Polygon

lon_start, lon_end = 60.5, 77.5
lat_start, lat_end = 37.0, 23.5
lon_step, lat_step = 0.25, 0.25

features = []

lat = lat_start
while lat > lat_end:
    lon = lon_start
    while lon < lon_end:
        polygon = Polygon([
            [lon, lat],
            [lon + lon_step, lat],
            [lon + lon_step, lat - lat_step],
            [lon, lat - lat_step],
            [lon, lat]
        ])
        feature = geojson.Feature(
            geometry=polygon,
            properties={
                "id": f"cell-{round(lat,2)}-{round(lon,2)}",
                "risk": "Unknown",
                "prediction": None
            }
        )
        features.append(feature)
        lon += lon_step
    lat -= lat_step

feature_collection = geojson.FeatureCollection(features)

with open("pakistan-grid.geojson", "w") as f:
    geojson.dump(feature_collection, f)
