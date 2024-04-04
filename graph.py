# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 08:52:49 2024

@author: Home
"""

import numpy as np
import matplotlib.pyplot as plt

def read_XFLR(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    alpha = []
    Cd = []
    Cl = []
    CM = []
    
    for line in lines[11:-2]:
        data = line.split()
        alpha.append(float(data[0]))
        Cl.append(float(data[1]))
        Cd.append(float(data[2]))
        CM.append(float(data[4]))
    return alpha,Cl,Cd,CM

def read_CFD(filename):
    data = np.loadtxt(filename)
    
    aoa = data[:,0]
    
    Cy = data[:,1]
    Cx = data[:,2]
    CM = data[:,3]
    
    CL = Cy*np.cos(aoa*np.pi/180) - Cx*np.sin(aoa*np.pi/180)
    CD = Cx*np.cos(aoa*np.pi/180) + Cy*np.sin(aoa*np.pi/180)
    
    return aoa, CL, CD, CM

def XFLRvsCFD():
    alpha_XFLR1,Cl_XFLR1,Cd_XFLR1, CM_XFLR1 = read_XFLR("data_XFLR5\Re6e6\T1_Re6e6_M0_N0.5_a_0-12.txt")
    alpha_XFLR2,Cl_XFLR2,Cd_XFLR2,CM_XFLR2 = read_XFLR("data_XFLR5\Re6e6\T1_Re6e6_M0_N0.5_a_0-(-12).txt")
    alpha_XFLR = np.concatenate((alpha_XFLR2,alpha_XFLR1))
    Cl_XFLR=np.concatenate((Cl_XFLR2,Cl_XFLR1))
    Cd_XFLR=np.concatenate((Cd_XFLR2,Cd_XFLR1))
    CM_XFLR=np.concatenate((CM_XFLR2,CM_XFLR1))
    
    alpha_CFD,Cl_CFD,Cd_CFD,CM_CFD = read_CFD("RevE_HC_Re6e6.txt")
    
    
    # Tracé du coefficient de traînée (Cd) en fonction de alpha
    plt.figure()
    plt.plot(alpha_XFLR, Cd_XFLR, label='XFLR')
    plt.plot(alpha_CFD, Cd_CFD, label='CFD')
    plt.xlabel('Angle of Attack[°]')
    plt.ylabel('Coefficient of Drag')
    plt.title('Cd vs. Alpha')
    plt.grid()
    plt.legend()
    plt.show()
    
    # Tracé du coefficient de portance (Cl) en fonction de alpha
    plt.figure()
    plt.plot(alpha_XFLR, Cl_XFLR, label='XFLR')
    plt.plot(alpha_CFD, Cl_CFD, label='CFD')
    plt.xlabel('Angle of Attack[°]')
    plt.ylabel('Coefficient of Lift')
    plt.title('Cl vs. Alpha')
    plt.grid()
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(alpha_XFLR, CM_XFLR, label='XFLR')
    plt.plot(alpha_CFD, CM_CFD, label='CFD')
    plt.xlabel('Angle of Attack[°]')
    plt.ylabel('$C_m$')
    plt.title('$C_m$ vs. Alpha')
    plt.grid()
    plt.legend()
    plt.show()
    
    
XFLRvsCFD()