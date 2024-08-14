# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:54:27 2024

@author: sarp
"""

import matplotlib.pyplot as plt
import numpy as np
from pullData import pullData
from scipy import stats


def generateFigure(xin, yin, keepZero=False, title=None): 
    df = pullData()
    
    if title is None:
        title = xin + " vs " + yin
    
    print(df.columns)

    if not keepZero:
        df = df[df[xin] != 0]
        df = df[df[yin] != 0]
    
    
    x = df[xin]
    y = df[yin]

    plt.scatter(x, y)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Calculate the trendline
    trendline = slope * x + intercept

    # Plot the trendline
    plt.plot(x, trendline, color='red', label=f'Trendline: y={slope:.2f}x+{intercept:.2f}')

    # Calculate R-squared
    r_squared = r_value**2

    # Add the R-squared value to the plot
    plt.text(0.8, 0.1, f'$R^2 = {r_squared:.2f}$', transform=plt.gca().transAxes, fontsize=12, color='green')


    plt.title(title)    
    plt.xlabel(xin)                
    plt.ylabel(yin)            
    plt.xlim()
    #plt.legend()




generateFigure('mean_emission_auto', 'sum_emission_auto', True)










