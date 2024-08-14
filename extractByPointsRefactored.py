# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:33:31 2024

@author: sarp
"""

import geopandas as gpd
import rasterio
import os
import pandas as pd

def extract(pointFile, pointLayer, fileDirectory, rasterNames):
    gdf = gpd.read_file(pointFile, layer=pointLayer)
    
    # Check if 'id' column exists, otherwise create a default one
    if 'id' not in gdf.columns:
        gdf['id'] = gdf.index + 1  # Create a default ID if not present

    points = gdf[['id', 'geometry']].copy()
    points['x'] = points.geometry.x
    points['y'] = points.geometry.y
    points = points[['id', 'x', 'y']]

    # Initialize a DataFrame to collect results
    results = []

    # Dictionary to cache raster data
    raster_cache = {}

    for index, row in points.iterrows():
        point_id = row['id']
        x, y = row['x'], row['y']
        result = {'PointID': point_id}

        # Loop through each raster file
        for raster in rasterNames:
            raster_path = os.path.join(fileDirectory, raster)

            # Cache raster file to avoid reopening
            if raster not in raster_cache:
                raster_cache[raster] = rasterio.open(raster_path)
            
            src = raster_cache[raster]
            num_bands = src.count
            
            # Extract values for each band
            for band in range(1, num_bands + 1):
                row_idx, col_idx = src.index(x, y)
                value = src.read(band)[row_idx, col_idx]
                if num_bands > 1:
                    result[f'{raster}_band_{band}'] = value
                else: 
                    result[f'{raster}'] = value
        
        results.append(result)

    # Convert results list to DataFrame
    return pd.DataFrame(results)

def extractByPoints():
    fileDirectory = r'C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes'
    
    allFiles = os.listdir(fileDirectory)
    rasterNames = [f for f in allFiles if os.path.isfile(os.path.join(fileDirectory, f)) and '.' not in f]

    # Process random points
    pointFile = r'C:\Users\sarp\Documents\sarp_research\data\RandomPoints.gpkg'
    pointLayer = "RandomPointsV2"
    random_df = extract(pointFile, pointLayer, fileDirectory, rasterNames)
    random_df['CAFO'] = False
    
    # Process CAFO points
    pointFile = r'C:\Users\sarp\Documents\sarp_research\data\RandomPoints.gpkg'
    pointLayer = "CAFOPointsV4"
    cafo_df = extract(pointFile, pointLayer, fileDirectory, rasterNames)
    cafo_df['CAFO'] = True
        
    # Combine DataFrames and save to CSV
    df_combined = pd.concat([random_df, cafo_df], ignore_index=True)
    df_combined.to_csv('pointSpectra.csv', index=False)
    
    return df_combined

# Run the function
df = extractByPoints()