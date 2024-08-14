# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 20:02:30 2024

@author: sarp

Note: unused, since I deceided to do this in ArcGIS rather than spend more time getting it working
"""

import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd

plumeFile = r'C:\Users\sarp\Documents\sarp_research\data\Carbon Mapper\plumes_2022-01-01_2023-01-01.csv'
polygonFile = r'C:\Users\sarp\Documents\sarp_research\code\CAFOs.gpkg'


plumes = pd.read_csv(plumeFile)
polygons = gpd.read_file(polygonFile, layer="CAFOs_Corrected")


points_gdf = gpd.GeoDataFrame(
    plumes, 
    geometry=gpd.points_from_xy(plumes.plume_longitude, plumes.plume_latitude),
    crs='EPSG:32611'  # Assuming WGS84 coordinate system
)

polygons_gdf = gpd.GeoDataFrame(
    polygons, 
    geometry='geometry',
    crs='EPSG:32611'
)

# Perform a spatial join to find which points fall within which polygons
joined_gdf = gpd.sjoin(points_gdf, polygons_gdf, how='right', op='within')
print(joined_gdf.iloc[5])

#print(plumes.columns)