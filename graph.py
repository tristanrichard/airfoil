# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 08:52:49 2024

@author: Home
"""

import numpy as np
import matplotlib.pyplot as plt

def read_pot():
    with open("cl_alpha_values.txt", 'r') as file:
        lines = file.readlines()
    
    alpha = []
    Cl = []
    
    for line in lines[1:]:
        data = line.split()
        alpha.append(float(data[0]))
        Cl.append(float(data[1]))

    return alpha,Cl

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

def XFLRvsCFD(file1,file2):
    alpha_XFLR1,Cl_XFLR1,Cd_XFLR1, CM_XFLR1 = read_XFLR(file1)
    alpha_XFLR2,Cl_XFLR2,Cd_XFLR2,CM_XFLR2 = read_XFLR(file2)
    alpha_XFLR = np.concatenate((alpha_XFLR2,alpha_XFLR1))
    Cl_XFLR=np.concatenate((Cl_XFLR2,Cl_XFLR1))
    Cd_XFLR=np.concatenate((Cd_XFLR2,Cd_XFLR1))
    CM_XFLR=np.concatenate((CM_XFLR2,CM_XFLR1))
    
    alpha_CFD,Cl_CFD,Cd_CFD,CM_CFD = read_CFD("RevE_HC_Re6e6.txt")
    
    
    # Tracé du coefficient de traînée (Cd) en fonction de alpha
    plt.figure()
    plt.plot(alpha_XFLR, Cd_XFLR, label='XFLR')
    plt.plot(alpha_CFD, Cd_CFD, label='RANS')
    plt.xlabel('Angle of Attack[°]')
    plt.ylabel('Coefficient of Drag')
    plt.title('Cd vs. Alpha')
    plt.grid()
    plt.legend()
    plt.savefig('./fig/XFLRvsCFD/Cd.png', format = 'png',dpi=300)
    plt.show()
    
    
    # Tracé du coefficient de portance (Cl) en fonction de alpha
    plt.figure()
    plt.plot(alpha_XFLR, Cl_XFLR, label='XFLR')
    plt.plot(alpha_CFD, Cl_CFD, label='RANS')
    plt.xlabel('Angle of Attack[°]')
    plt.ylabel('Coefficient of Lift')
    plt.title('Cl vs. Alpha')
    plt.grid()
    plt.legend()
    plt.savefig('./fig/XFLRvsCFD/Cl.png', format = 'png',dpi=300)
    plt.show()
    
    plt.figure()
    plt.plot(alpha_XFLR, CM_XFLR, label='XFLR')
    plt.plot(alpha_CFD, CM_CFD, label='RANS')
    plt.xlabel('Angle of Attack[°]')
    plt.ylabel('$C_m$')
    plt.title('$C_m$ vs. Alpha')
    plt.grid()
    plt.legend()
    plt.savefig('./fig/XFLRvsCFD/Cm.png', format = 'png',dpi=300)
    plt.show()
    
    
def CL_CDvsRe():
    
    alpha_XFLR1,Cl_XFLR1,Cd_XFLR1,CM_XFLR2 = read_XFLR("data_XFLR5\Re4.4\T1_Re4.4_M0.55_N0.5_a_0-12.txt")
    alpha_XFLR2,Cl_XFLR2,Cd_XFLR2,CM_XFLR2 = read_XFLR("data_XFLR5\Re4.4\T1_Re4.4_M0.55_N0.5_a_0-(-12).txt")
    alpha_XFLR_4 = np.concatenate((alpha_XFLR2,alpha_XFLR1))
    Cl_XFLR_4=np.concatenate((Cl_XFLR2,Cl_XFLR1))
    Cd_XFLR_4=np.concatenate((Cd_XFLR2,Cd_XFLR1))
    polar4=np.subtract(Cl_XFLR_4,Cd_XFLR_4)
    
    alpha_XFLR3,Cl_XFLR3,Cd_XFLR3,CM_XFLR3 = read_XFLR("data_XFLR5\Re3.1\T1_Re3.1_M0.37_N0.5_a_0-12.txt")
    alpha_XFLR4,Cl_XFLR4,Cd_XFLR4,CM_XFLR4 = read_XFLR("data_XFLR5\Re3.1\T1_Re3.1_M0.37_N0.5_a_0-(-12).txt")
    alpha_XFLR_3 = np.concatenate((alpha_XFLR4,alpha_XFLR3))
    Cl_XFLR_3=np.concatenate((Cl_XFLR4,Cl_XFLR3))
    Cd_XFLR_3=np.concatenate((Cd_XFLR4,Cd_XFLR3))
    
    alpha_XFLR5,Cl_XFLR5,Cd_XFLR5,CM_XFLR6 = read_XFLR("data_XFLR5\Re1.5\T1_Re1.5_M0.18_N0.5_a_0-12.txt")
    alpha_XFLR6,Cl_XFLR6,Cd_XFLR6,CM_XFLR6 = read_XFLR("data_XFLR5\Re1.5\T1_Re1.5_M0.18_N0.5_a_0-(-12).txt")
    alpha_XFLR_1 = np.concatenate((alpha_XFLR6,alpha_XFLR5))
    Cl_XFLR_1=np.concatenate((Cl_XFLR6,Cl_XFLR5))
    Cd_XFLR_1=np.concatenate((Cd_XFLR6,Cd_XFLR5))
    





    plt.figure()
    plt.plot(alpha_XFLR_4, Cd_XFLR_4, label='Re=4.4x10e6')
    plt.plot(alpha_XFLR_3, Cd_XFLR_3, label='Re=3.1x10e6')
    plt.plot(alpha_XFLR_1, Cd_XFLR_1,color='brown', label='Re=1.5x10e6')
    plt.xlabel('Angle of Attack[°]')
    plt.ylabel('Coefficient of Drag')
    plt.title('Cd vs Alpha for different Reynolds')
    plt.grid()
    plt.legend()
    plt.savefig('./fig/diff_Re/Cd.png', format = 'png',dpi=300)
    plt.show()
    
    # Tracé du coefficient de portance (Cl) en fonction de alpha
    plt.figure()
    plt.plot(alpha_XFLR_4, Cl_XFLR_4, label='Re=4.4x10e6')
    plt.plot(alpha_XFLR_3, Cl_XFLR_3, label='Re=3.1x10e6')
    plt.plot(alpha_XFLR_1, Cl_XFLR_1,color='brown', label='Re=1.5x10e6')
    plt.xlabel('Angle of Attack[°]')
    plt.ylabel('Coefficient of Lift')
    plt.title('Cl vs Alpha for different Reynolds')
    plt.grid()
    plt.legend()
    plt.savefig('./fig/diff_Re/Cl.png', format = 'png',dpi=300)
    plt.show()
    
        # Définir la pente de la ligne
    pente = 47  # Modifiez la pente selon vos besoins
    
    # Coordonnées du point final de la ligne
    x_end = 0.04
    y_end = pente * x_end
    
    plt.figure()
    plt.plot(Cd_XFLR_4,Cl_XFLR_4, label='Re=4.4x10e6')
    plt.plot(Cd_XFLR_3,Cl_XFLR_3, label='Re=3.1x10e6')
    plt.plot(Cd_XFLR_1,Cl_XFLR_1,color='brown', label='Re=1.5x10e6')
    plt.plot([0, x_end], [0, y_end], color='red', linestyle='--', label='(Cl/Cd)max')
    plt.xlabel('Cd')
    plt.ylabel('Cl')
    plt.title('airfoil polar for different Reynolds')
    plt.grid()
    plt.legend()
    plt.savefig('./fig/diff_Re/Cl-Cd.png', format = 'png',dpi=300)
    plt.show()
    
    
def XFLRvspot():
    alpha,Cl = read_pot()
    alpha_XFLR1,Cl_XFLR1,a,b = read_XFLR("data_XFLR5\Re4.4\T1_Re4.4_M0.55_N0.5_a_0-12.txt")
    alpha_XFLR2,Cl_XFLR2,c,d = read_XFLR("data_XFLR5\Re4.4\T1_Re4.4_M0.55_N0.5_a_0-(-12).txt")
    alpha_XFLR = np.concatenate((alpha_XFLR2,alpha_XFLR1))
    Cl_XFLR=np.concatenate((Cl_XFLR2,Cl_XFLR1))
    
    
    plt.figure()
    plt.plot(alpha_XFLR, Cl_XFLR, label='XFLR')
    #plt.plot(alpha, np.add(np.divide(Cl,50000),0.75), label='Potential')
    plt.plot(alpha,Cl, label='Potential')
    plt.xlabel('Angle of Attack[°]')
    plt.ylabel('Coefficient of Lift')
    plt.title('Cl vs. Alpha')
    plt.grid()
    plt.legend()
    plt.savefig('./fig/XFLRvsPOT/Cl.png', format = 'png',dpi=300)
    plt.show()
    
    
    
    
#XFLRvsCFD("data_XFLR5\Re6e6\T1_Re6e6_M0_N0.5_a_0-12.txt","data_XFLR5\Re6e6\T1_Re6e6_M0_N0.5_a_0-(-12).txt")
#XFLRvsCFD("data_XFLR5\Re6e6\T1_Re6e6_M0.45_N0.5_a_0-12.txt","data_XFLR5\Re6e6\T1_Re6e6_M0.45_N0.5_0-(-12).txt")
#CL_CDvsRe()
XFLRvspot()