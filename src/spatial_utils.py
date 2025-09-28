import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import h3
import numpy as np

def latlng_to_h3(lat, lng, h3_level=7):
    """Convert latitude/longitude to H3 cell."""
    return h3.geo_to_h3(lat, lng, h3_level)

def assign_h3_cells(df, lat_col='Start_Lat', lng_col='Start_Lng', h3_level=7):
    """Assign H3 cells to each row in the dataframe."""
    df['h3_cell'] = df.apply(lambda row: latlng_to_h3(row[lat_col], row[lng_col], h3_level), axis=1)
    return df

def aggregate_by_h3(df, cell_col='h3_cell'):
    """Aggregate accident counts by H3 cell."""
    counts = df.groupby(cell_col).size().to_frame('accident_count').reset_index()
    return counts

def h3_to_polygon(h3_index):
    """Convert H3 cell to polygon geometry."""
    boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)
    return Polygon(boundary)

def add_h3_geometry(df, cell_col='h3_cell'):
    """Add geometry column to H3 aggregated data."""
    df['geometry'] = df[cell_col].apply(h3_to_polygon)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    return gdf

def calculate_spatial_features(df, lat_col='Start_Lat', lng_col='Start_Lng'):
    """Calculate spatial features like distance to city center."""
    # Calculate distance to city center (example: Columbus, OH)
    city_center_lat, city_center_lng = 39.9612, -82.9988

    df['distance_to_center'] = np.sqrt(
        (df[lat_col] - city_center_lat)**2 + 
        (df[lng_col] - city_center_lng)**2
    ) * 111.32  # Convert to km

    return df

def save_gdf(gdf, path='hotspot_grid.geojson'):
    """Save GeoDataFrame to file."""
    gdf.to_file(path, driver='GeoJSON')
    print(f"GeoDataFrame saved to {path}")

def create_spatial_grid(bounds, grid_size=0.01):
    """Create a spatial grid for analysis."""
    min_lat, max_lat = bounds['lat_min'], bounds['lat_max']
    min_lng, max_lng = bounds['lng_min'], bounds['lng_max']

    lats = np.arange(min_lat, max_lat, grid_size)
    lngs = np.arange(min_lng, max_lng, grid_size)

    grid_cells = []
    for lat in lats:
        for lng in lngs:
            grid_cells.append({
                'lat': lat,
                'lng': lng,
                'cell_id': f"{lat:.3f}_{lng:.3f}"
            })

    return pd.DataFrame(grid_cells)