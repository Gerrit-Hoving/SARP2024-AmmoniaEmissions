#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 15:02:23 2022

@author: madeleip
"""
import h5py
import numpy as np
import sys


def hytesinputs(fL2, fgeo, fgeohdr, gas):
    
    #Input HDR/HDF/.dat 
    #inputs:
        #LOC     : path of file
        #fL2     : name of hdf file
        #fgeo    : name of geo.dat file 
        #fgeohdr : name of header file 
        
    #OUTPUT 
    #NP ARRAYS
        #TS     : LST (K)
        #etas   : EMISS
        #lathyf : Lat Hyf (dec deg)
        #lonhyf : Lon Hyf (dec deg)
    
    
    
    #FOR THE PURPOSE OF THE EXAMPLE .... 
    # HyTES location
    # loc='/Users/madeleip/Documents/TIR/DATASETS/HYTES/herts/20210720t105655_HertsGB/'

    # # Eg file names 
    # fL2 = loc + '20210720t105655_HertsGB_L2_B100_V01.hdf5';
    # fgeo = loc+'20210720t105655_HertsGB_L1_B100_V01.geo.dat';
    # fgeohdr = loc+'20210720t105655_HertsGB_L1_B100_V01.geo.hdr';





    #Hytes Header Info 
    fid = open(fgeohdr)
    lin=fid.readline().rstrip()
    lin=fid.readline().rstrip()
    lin=fid.readline().rstrip()
    lin=fid.readline().rstrip()
    
    tok,rem=lin.split('=')
    cols = int(rem) #get number of cols 
    
    lin=fid.readline().rstrip()
    tok,rem=lin.split('=')
    rows = int(rem) #get number of rows 
   
    lin=fid.readline().rstrip()
    lines,rem=lin.split('=')
    bands = int(rem)
    
    
    
    
    #Hytes Geo File geo.dat
    geo = open(fgeo, 'r')
    geo=np.fromfile(geo,np.float32)
    bgeo = 3
    
    geo = np.reshape(geo,(bgeo,cols,rows),order='F')
    
    geo=np.transpose(geo)
    geo = np.double(geo)
    
    
    # Sensor altitude in meters
    #Hgt = double(hdf5read(L1info.GroupHierarchy.Groups(1).Datasets(4)));
    #Hgt = mean(Hgt(:))./1000; % Average and convert to km
    
    # DEM
    DEM = geo[:,:,2]
    DEM = np.mean(DEM)/1000
     
    #Lat and Lon 
    LatHyf = geo[:,:,0]
    LonHyf = geo[:,:,1]

    #size 
    szh=np.shape(LatHyf)
    

    
    #Read in HyTES
    

    f = h5py.File(fL2, 'r')
    
    #get key of fL2
    hyteskey=list(f.keys())
    print(hyteskey)
    #import LST with either key 
   
    
    #LST=np.fillmissing(LST,'nearest');
    # if 'L2_Emissivity' in hyteskey:
    #     emis = f['L2_Emissivity']
    # if 'l2_emissivity' in hyteskey:
    #     emis=f['l2_emissivity']
    # emis =np.asarray(emis)
    

    # if 'L2_Emissivity_Wavelengths' in hyteskey:
    #     emiswave = f['L2_Emissivity_Wavelengths'];  # Read Emissivity wavelengths
    # if 'l2_emissivity_wavelengths' in hyteskey:
    #     emiswave = f['l2_emissivity_wavelengths']; 
    
    
    # emiswave = np.asarray(emiswave)
    # w = emiswave*10**-6

    # szL2 = np.shape(NH3)
    # szL2 = np.shape(NH3)
    # szL2 = np.shape(NH3)
 


    # Broadband emissivity calculation
    # Tf = 300;
    # c1 = 3.7418e-22;
    # c2 = 0.014388;
    # BBE = np.zeros(szL2);Bn1=np.sum(emis*(c1/(w**5*np.pi*(np.exp(c2/(w*Tf))-1))),2);Bn2 = sum(c1/(w**5*np.pi*(np.exp(c2/(w*Tf))-1)));BBE=Bn1/Bn2;BBEff=BBE;

  #  for i in range(0,szh[0]):
    
   #     for j in range(0,szh[1]):
        
    #        e = np.squeeze(emis[i,j,:]);
    
     #       Bnum = e*(c1/(w**5*np.pi*(np.exp(c2/(w*Tf))-1)));
      #      Bdem = c1/(w**5*np.pi*(np.exp(c2/(w*Tf))-1));
                   
       #     BBE[i,j] = sum(Bnum)/sum(Bdem);
   # BBEff = BBE;

    
    maxLat = np.max(LatHyf[:])
    minLat = np.min(LatHyf[:])
    maxLon = np.max(LonHyf[:])
    minLon = np.min(LonHyf[:])



    #Sometimes Lat/Lon and L1/L2 don't have same number of lines so resize for now
    # LatHyf = np.resize(LatHyf,szL2)
    # LonHyf = np.resize(LonHyf,szL2)
    

    if gas=='NH3':
        if 'NH3' in hyteskey:
            NH3 = f['NH3']
            NH3 = NH3['NH3_CMF2']
            NH3 = np.asarray(NH3)
            NH3 = np.where(NH3<0, np.nan,NH3)
            szL2 = np.shape(NH3)
            
            LatHyf = np.resize(LatHyf,szL2)
            LonHyf = np.resize(LonHyf,szL2)
            
            print('Pulled NH3 data')
        
            return NH3,LatHyf,LonHyf,True
     
    if gas=='CH4':
        if 'CH4' in hyteskey:
            CH4 = f['CH4']
            CH4 = CH4['CH4_CMF2']
            CH4 = np.asarray(CH4)
            CH4 = np.where(CH4<0, np.nan,CH4)
            #szL2 = np.shape(CH4)
            
            #LatHyf = np.resize(LatHyf,szL2)
            #LonHyf = np.resize(LonHyf,szL2)
        
            return CH4,LatHyf,LonHyf,True 
    
    if gas=='H2S':
        if 'H2S' in hyteskey:
            H2S = f['H2S']
            H2S = H2S['H2S_CMF2']
            H2S = np.asarray(H2S)
            H2S=np.where(H2S<0, np.nan,H2S)
            #szL2 = np.shape(H2S)
            
            #LatHyf = np.resize(LatHyf,szL2)
            #LonHyf = np.resize(LonHyf,szL2)
        
            return H2S,LatHyf,LonHyf,True  

    return NH3, LatHyf, LonHyf, False




