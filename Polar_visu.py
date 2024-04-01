#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 13:33:10 2024

@author: jcrismer
"""


import numpy as np
import matplotlib.pyplot as plt

def read_xfoilData(filename, inviscid=False):
    data = np.loadtxt(filename, skiprows=12)
    
    aoa = data[:,0]
    CL  = data[:,1]
    if not(inviscid):
        CD  = data[:,2]
    else:
        CD  = data[:,3]
    CM  = data[:,4]
    
    return aoa, CL, CD, CM

def read_AnsysData(filename):

    data = np.loadtxt(filename)
    
    aoa = data[:,0]
    
    Cy = data[:,1]
    Cx = data[:,2]
    CM = data[:,3]
    
    CL = Cy*np.cos(aoa*np.pi/180) - Cx*np.sin(aoa*np.pi/180)
    CD = Cx*np.cos(aoa*np.pi/180) + Cy*np.sin(aoa*np.pi/180)
    
    return aoa, CL, CD, CM

NACA0012_example = read_xfoilData('./NACA0012_example_polar.txt')
RANS_data = read_AnsysData('./RevE_HC_Re6e6.txt')


plt.figure(); #draw on previous figure
plt.title('Lift coefficient')
plt.plot(NACA0012_example[0], NACA0012_example[1], label = 'Xflr - NACA0012 - Re 1e6') # , label = r'$\mathrm{present}$'
plt.plot(RANS_data[0], RANS_data[1], label = 'RANS - RevE_HC - Re 1e6') # , label = r'$\mathrm{present}$'

plt.xlabel("Angle of attack [°]")
plt.ylabel("$C_L$")
plt.grid()
plt.legend()
plt.savefig('./fig/NACA0012_example_Cl.pdf', format = 'pdf')

plt.figure(); #draw on previous figure
plt.title('Drag coefficient')
plt.plot(NACA0012_example[0], NACA0012_example[2], label = 'Xflr - NACA0012 - Re 1e6') # , label = r'$\mathrm{present}$'
plt.plot(RANS_data[0], RANS_data[2], label = 'RANS - RevE_HC - Re 1e6') # , label = r'$\mathrm{present}$'

plt.xlabel("Angle of attack [°]")
plt.ylabel("$C_D$")
plt.grid()
plt.legend()
plt.savefig('./fig/NACA0012_example_Cd.pdf', format = 'pdf')

plt.figure(); #draw on previous figure
plt.plot(NACA0012_example[0], NACA0012_example[3], label = 'Xflr - NACA0012 - Re 1e6') # , label = r'$\mathrm{present}$'
plt.plot(RANS_data[0], RANS_data[3], label = 'RANS - RevE_HC - Re 1e6') # , label = r'$\mathrm{present}$'

plt.xlabel("Angle of attack [°]")
plt.ylabel("$C_m$")
plt.grid()
plt.legend()
plt.savefig('./fig/NACA0012_example_Cm.pdf', format = 'pdf')

plt.show()
