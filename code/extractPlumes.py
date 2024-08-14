# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:33:25 2024

@author: sarp
"""

import fiona
from rasterstats import *
import pandas as pd



def extractPlumes():

    
    nh3raster = r'C:\Users\sarp\Documents\sarp_research\data\HyTES\output\reprocessing\NH3Mosaic.tif'
    ch4raster = r'C:\Users\sarp\Documents\sarp_research\data\HyTES\output\reprocessing\CH4Mosaic.tif'
    
    nh3plumes = fiona.open(r'C:\Users\sarp\Documents\sarp_research\code\Plumes.gpkg', layer="NH3")
    ch4plumes = fiona.open(r'C:\Users\sarp\Documents\sarp_research\code\Plumes.gpkg', layer="CH4")

    # Do it all with NH3
    results = []
    attributes = []
    for feature in nh3plumes:
        attributes.append(fiona.model.to_dict(feature['properties']))
    results.append(attributes)
    
    results.append(zonal_stats(nh3plumes, nh3raster, stats=['median', 'mean', 'sum']))
    
    
    list_names = ['Lot', 'NH3']
    
    # Combine lists into a dictionary of DataFrames
    dfs = {}
    for list_name, data in zip(list_names, results):
        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(data)
        # Rename columns to include the list name
        df.columns = [f'{list_name}_{col}' for col in df.columns]
        # Add DataFrame to dictionary
        dfs[list_name] = df
    
    # Concatenate all DataFrames horizontally (axis=1)
    nh3df = pd.concat(dfs.values(), axis=1)
    
    # Drop rows where the value in the first column is null
    nh3df = nh3df.dropna(subset=[nh3df.columns[0]])
    print(nh3df)
    
    nh3df.to_csv('nh3.csv')
    
    
    
    
    # Do it all with CH4
    results = []
    attributes = []
    for feature in ch4plumes:
        attributes.append(fiona.model.to_dict(feature['properties']))
    results.append(attributes)
    results.append(zonal_stats(ch4plumes, ch4raster, stats=['median', 'mean', 'sum']))
    
    
    list_names = ['Lot', 'CH4']
    
    # Combine lists into a dictionary of DataFrames
    dfs = {}
    for list_name, data in zip(list_names, results):
        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(data)
        # Rename columns to include the list name
        df.columns = [f'{list_name}_{col}' for col in df.columns]
        # Add DataFrame to dictionary
        dfs[list_name] = df
    
    # Concatenate all DataFrames horizontally (axis=1)
    ch4df = pd.concat(dfs.values(), axis=1)

    # Drop rows where the value in the first column is null
    ch4df = ch4df.dropna(subset=[ch4df.columns[0]])
    print(ch4df)
    
    ch4df.to_csv('ch4.csv')

    return nh3df, ch4df

extractPlumes()
