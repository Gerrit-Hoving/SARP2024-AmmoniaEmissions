# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 19:45:47 2024

@author: sarp
"""

import zipfile as z
import os
#from hytesinputs import *
#from writegeotiff import *

folder = r'C:\Users\sarp\Downloads\HyTES'
dataDest = r'C:\Users\sarp\Documents\sarp_research\data\raw'
#imgDest = r'C:\Users\sarp\Documents\sarp_research\data\test'

directory = os.fsencode(folder)
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".zip"): 
        with z.ZipFile(os.path.join(folder, filename), 'r') as zObject:
            fileList = zObject.namelist()
            
            for f in fileList:
                target = os.fsdecode(f)
                if target.endswith(".hdf5"):
                    batchHDF5 = target
                    #batchName = target.removesuffix(".hdf5")
                    zObject.extract(target, path=dataDest)
                    
                if target.endswith("geo.hdr"):
                    batchHDR = target
                    zObject.extract(target, path=dataDest)
                    
                if  target.endswith("geo.dat"):
                    batchDAT = target
                    zObject.extract(target, path=dataDest)
                    
            listLoc = os.path.join(dataDest,'fileList.txt')
            
            with open(listLoc, 'a') as f:
                f.write(batchHDF5)
                f.write('\n')
                f.write(batchHDR)
                f.write('\n')
                f.write(batchDAT)
                f.write('\n')
            
            
            
            '''raw_hy_dir_path = dataDest # the directory that contains the hdf5 file, the geo.dat and the geo.hdr.
            hdf_f_path = os.path.join(raw_hy_dir_path,batchHDF5)
            geo_dat_f_path = os.path.join(raw_hy_dir_path,batchDAT)
            geo_hdr_f_path = os.path.join(raw_hy_dir_path,batchHDR)

            dst_path = os.path.join(imgDest,batchName + '_NH3.tif')
            print(hdf_f_path,geo_dat_f_path,geo_hdr_f_path,'NH3')
            data,lat,lon,status = hytesinputs(hdf_f_path,geo_dat_f_path,geo_hdr_f_path,'NH3')
            
            if status:
                writegeotiff(data,lon,lat,dst_path)
            
            dst_path = os.path.join(imgDest,batchName + '_CH4.tif')
            data,lat,lon,status = hytesinputs(hdf_f_path,geo_dat_f_path,geo_hdr_f_path,'CH4')
            if status:
                writegeotiff(data,lon,lat,dst_path)
            
            dst_path = os.path.join(imgDest,batchName + '_H2S.tif')
            data,lat,lon,status = hytesinputs(hdf_f_path,geo_dat_f_path,geo_hdr_f_path,'H2S')
            if status:
                writegeotiff(data,lon,lat,dst_path)
            '''      
        continue
    else:
        continue


