#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:14:38 2024

@author: jcrismer
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def thickness(x, a, b, c, d, e):
    y = (a*np.sqrt(x) + b*x + c*x**2 + d*x**3 + e*x**4)
    return y

def mean_line(x, a, b, c, d, e):
    y = (a + b*x + c*x**2 + d*x**3 + e*x**4)
    return y

def get_intra_extra(x, c, airfoil):
    if airfoil == 'RevE-HC':
        a_ml, b_ml, c_ml, d_ml, e_ml = 0.0, 0.50511, -0.97016, 0.79535, -0.3303
        a_t, b_t, c_t, d_t, e_t = 0.39981, -0.09876, -0.6376, 0.34907, -0.01252
        
        y_intra = c*mean_line(x/c, a_ml, b_ml, c_ml, d_ml, e_ml) - c*thickness(x/c, a_t, b_t, c_t, d_t, e_t)
        y_extra = c*mean_line(x/c, a_ml, b_ml, c_ml, d_ml, e_ml) + c*thickness(x/c, a_t, b_t, c_t, d_t, e_t)
        
    else:
        print('This airfoil is unknown : ', airfoil)
        
    return y_intra, y_extra

if __name__ == "__main__":
    airfoil_data = np.loadtxt('Airfoil-RevE-HC.dat')
    c = 2
    N = 200
    # Create x discretization with a change of variable to get more points on LE and TE
    theta = np.linspace(0, np.pi, N)
    x_c = c*(0.5)*(1-np.cos(theta))
    
    # Other discretization to avoid small panels at TE
    theta_end = 0.95*np.pi
    theta = np.linspace(0, theta_end, N)
    x_c = c*(1-np.cos(theta))/(1-np.cos(theta_end))
    
    y_intra, y_extra = get_intra_extra(x_c, c, 'RevE-HC')
    airfoil_fun = np.concatenate((np.flip(y_intra), y_extra[1:]))
    x = np.concatenate((np.flip(x_c), x_c[1:]))
    
    ml, t = c*mean_line(x_c/c, 0.00215, 0.47815, -0.87927, 0.68242, -0.28378), c*thickness(x_c/c, 0.40182, -0.10894, -0.60082, 0.2902, 0.01863)
    
    ### Plot airfoil
    fig,ax = plt.subplots()
    
    # ax.plot(airfoil_data[:,0], airfoil_data[:,1], 'k', label = 'Airfoil (data)')
    ax.plot(x/c, airfoil_fun/c, 'k.', label = 'Airfoil (fun)')
    # ax.plot(x_c/c, ml/c,'k:', label = 'Mean line')
    # ax.plot(x_c/c, t/c, 'k--', label = 'Thickness')
    
    # ax.plot(x/c, airfoil_fun/c, 'k', label = 'Airfoil ($y^*_c \pm y^*_t$)')
    # ax.plot(x_c/c, ml/c,'k:', label = 'Mean line ($y^*_c$)')
    # ax.plot(x_c/c, t/c, 'k--', label = 'Thickness ($y^*_t$)')
    
    ax.set_xlabel(r'$x/c$', fontsize=13)
    ax.set_ylabel(r'$y/c$', fontsize=13)
    ax.axis('scaled')
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
    ax.grid()
    ax.legend(loc='upper right')
    
    plt.show()