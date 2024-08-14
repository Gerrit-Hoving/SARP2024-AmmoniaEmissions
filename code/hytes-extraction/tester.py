# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:00:41 2024

@author: sarp
"""

import h5py
import numpy as np

fL2 = "2015-05-01.202021.Tipton.Line11-Run1-Segment01.L3-CMF.hdf5"

f = h5py.File(fL2, 'r')

hyteskey=list(f.keys())

print(hyteskey)

if 'NH3' in hyteskey:
    NH3 = f['NH3']
    
    
subkey=list(NH3.keys())
print(subkey)

NH3_CMF2 = NH3['NH3_CMF2']
    
NH3_CMF2 = np.asarray(NH3_CMF2)

print(NH3_CMF2)