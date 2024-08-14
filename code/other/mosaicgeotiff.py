# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:48:40 2024

@author: sarp
"""

import os
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import glob

def mosaic_geotiffs(input_folder, output_file):
    # Get list of all GeoTIFF files in the input folder
    geotiff_files = glob.glob(os.path.join(input_folder, '*NH3.tiff'))
    
    if not geotiff_files:
        raise ValueError("No GeoTIFF files found in the specified folder.")
    
    # Read each GeoTIFF file and add to a list
    rasters = [rasterio.open(f) for f in geotiff_files]
    
    # Merge the rasters
    mosaic_array, out_transform = merge(rasters)
    
    # Get metadata from the first raster
    out_meta = rasters[0].meta.copy()
    
    # Update the metadata with the new dimensions, transform, and CRS
    out_meta.update({
        "driver": "GTiff",
        "count": mosaic_array.shape[0],
        "height": mosaic_array.shape[1],
        "width": mosaic_array.shape[2],
        "transform": out_transform
    })
    
    # Write the mosaic to the output file
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic_array)
    
    print(f"Mosaic saved to {output_file}")

# Usage
input_folder = r'C:\Users\sarp\Documents\sarp_research\data\HyTES\output\reprocessing'  # Folder containing GeoTIFF files
output_file = r'C:\Users\sarp\Documents\sarp_research\data\HyTES\output\reprocessing\NH3Mosaic.tif'  # Path to save the output mosaic TIFF file

mosaic_geotiffs(input_folder, output_file)