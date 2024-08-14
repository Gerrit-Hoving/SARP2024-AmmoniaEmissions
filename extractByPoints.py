# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:34:56 2024

@author: sarp
"""

import rasterio
import pandas as pd
import geopandas as gpd
import os

def extract(pointFile, pointLayer, fileDirectory, rasterNames):
    gdf = gpd.read_file(pointFile, layer=pointLayer)
    
    # Check if 'id' column exists, otherwise create a default one
    if 'id' not in gdf.columns:
        gdf['id'] = gdf.index + 1  # Create a default ID if not present

    points = gdf[['id', 'geometry']].copy()
    points['x'] = points.geometry.x
    points['y'] = points.geometry.y
    points = points[['id', 'x', 'y']].values.tolist()

    all_results = []

    # Initialize a DataFrame to collect results
    for point_id, x, y in points:
        result = {'PointID': point_id}
        
        # Loop through each raster file
        for raster in rasterNames:
            with rasterio.open(os.path.join(fileDirectory, raster)) as src:
                num_bands = src.count
                # Extract values for each band
                for band in range(1, num_bands + 1):
                    row, col = src.index(x, y)
                    value = src.read(band)[row, col]
                    if num_bands > 1:
                        result[f'{raster}_band_{band}'] = value
                    else: 
                        result[f'{raster}'] = value
        
        all_results.append(result)
        
    return pd.DataFrame(all_results)




def extractByPoints():

    fileDirectory = r'C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes'
    
    allFiles = os.listdir(fileDirectory)
    rasterNames = [f for f in allFiles if os.path.isfile(os.path.join(fileDirectory, f)) and '.' not in f]
    #rasterPaths = [os.path.join(fileDirectory, raster) for raster in rasterNames]
    #rasterStats = ['median', 'mean', 'sum']    

    pointFile = r'C:\Users\sarp\Documents\sarp_research\data\RandomPoints.gpkg'
    pointLayer = "RandomPointsV2"
    random_df = extract(pointFile, pointLayer, fileDirectory, rasterNames)
    random_df['CAFO'] = False
    
    print(random_df.columns)
    
    pointFile = r'C:\Users\sarp\Documents\sarp_research\data\RandomPoints.gpkg'
    pointLayer = "CAFOPointsV4"
    cafo_df = extract(pointFile, pointLayer, fileDirectory, rasterNames)
    cafo_df['CAFO'] = True
        
    print(cafo_df.columns)
    
    df_combined = pd.concat([random_df, cafo_df], ignore_index=True)
    
    print(df_combined.columns)
    print(df_combined.head)
    
    df_combined.to_csv('pointSpectra.csv')
    
    return df_combined
    


df = extractByPoints()

# Display the DataFrame
print(df.columns)
print(df.head)

