#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:14:38 2024

@author: jcrismer
"""

import scipy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Panel():
    def __init__(self, x_start, y_start, x_end, y_end):
        self.start = (x_start, y_start)
        self.end = (x_end, y_end)
        self.control = (1/2 * (x_start+x_end), 1/2 * (y_start+y_end))
        self.length = (np.abs(x_end-x_start), np.abs(y_end-y_start))
        self.angle = np.arctan2(self.length[1], self.length[0])
        self.longueur = np.sqrt(((y_end-y_start)**2)+((y_end-y_start)**2))
        
        
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

def influence_coefficients(panel_i,panel_j,panel_j_avant,same):
    # Calculate the influence of the panel j on the panel i
    A=0
    #panel_j
    if(panel_j!=None):
        b1= panel_j.longueur/2
        alpha1 = -1/(2*b1)
        beta1 = 1/2
        x1=np.abs(panel_j.start[0]-panel_i.control[0])
        y1=np.abs(panel_j.start[1]-panel_i.control[1])
    
    #panel j-1
    if(panel_j_avant!=None):
        b2= panel_j_avant.longueur/2
        alpha2 = 1/(2*b2) #le signe change car devient yR 
        beta2 = 1/2
        x2=np.abs(panel_j_avant.end[0]-panel_i.control[0])
        y2=np.abs(panel_j_avant.end[1]-panel_i.control[1])
    
    if (same):
        if(panel_j!=None):
            A+= funct2(alpha1,beta1,b1)*np.cos(panel_i.angle-panel_j.angle)
        if(panel_j_avant!=None):
            A+=funct2(alpha2,beta2,b2)*np.cos(panel_i.angle-panel_j_avant.angle)
    if(not same):
        #panel_j
        if(panel_j!=None):
            A+= funct1(alpha1,beta1,x1,y1,x1-b1)*np.cos(panel_i.angle-panel_j.angle) + funct3(alpha1,beta1,x1,y1,x1+b1)*np.cos(panel_i.angle-panel_j.angle)
        
        #panel j-1
        if(panel_j_avant!=None):
            A+= funct1(alpha2,beta2,x2,y2,x2-b2)*np.cos(panel_i.angle-panel_j_avant.angle) + funct3(alpha2,beta2,x2,y2,x2+b2)*np.cos(panel_i.angle-panel_j_avant.angle)
        
    return A

def funct1(alpha,beta,x,y,s):
    return (-1/(2*np.pi))*((alpha*(-s+(y*np.arctan2(s,y))))+((1/2)*(alpha*x+beta)*np.log((s**2)+(y**2))))

def funct3(alpha,beta,x,y,s):
    return (1/(2*np.pi))*((alpha*(-s+(y*np.arctan2(s,y))))+((1/2)*(alpha*x+beta)*np.log((s**2)+(y**2))))

def funct2(alpha,beta,b):
    return (-1/(2*np.pi))*((2*alpha*b))

if __name__ == "__main__":
    airfoil_data = np.loadtxt('Airfoil-RevE-HC.dat')
    c = 2
    N = 200
    AoA = 5
    
    # Create x discretization with a change of variable to get more points on LE and TE
    #theta = np.linspace(0, np.pi, N)
    #x_c = c*(0.5)*(1-np.cos(theta))
    
    # Other discretization to avoid small panels at TE
    theta_end = 0.95*np.pi
    theta = np.linspace(0, theta_end, (N//2)+1)
    x_c = c*(1-np.cos(theta))/(1-np.cos(theta_end))
    
    y_intra, y_extra = get_intra_extra(x_c, c, 'RevE-HC')
    airfoil_fun = np.concatenate((np.flip(y_intra), y_extra[1:]))
    x = np.concatenate((np.flip(x_c), x_c[1:]))
    
    ml, t = c*mean_line(x_c/c, 0.00215, 0.47815, -0.87927, 0.68242, -0.28378), c*thickness(x_c/c, 0.40182, -0.10894, -0.60082, 0.2902, 0.01863)
    
    ### Setup of the influence matrix and the RHS
    b = np.zeros(N+1)
    A = np.zeros((N+1,N+1))
    for i in range(N):          # Vérifier le N-1, mais je crois qu'on peut juste l'imposer avec la condition au TE
        panel_i = Panel(x[i], airfoil_fun[i], x[i+1], airfoil_fun[i+1])
        for j in range(N+1):
            if(j!=N):   
                panel_j = Panel(x[j], airfoil_fun[j], x[j+1], airfoil_fun[j+1])
            if(j==N):
                panel_j=None
            if(j!=0):
                panel_j_avant = Panel(x[j-1], airfoil_fun[j-1], x[j], airfoil_fun[j])
            if(j==0):
                panel_j_avant=None
            if(i==j):
                A[i,j] = influence_coefficients(panel_i, panel_j,panel_j_avant,True)
            if(i!=j):
                A[i,j] = influence_coefficients(panel_i, panel_j,panel_j_avant,False)
        b[i] = np.cos(AoA)*np.sin(panel_i.angle) - np.sin(AoA)*np.cos(panel_i.angle)
    
    for k in range(N+1):
        if(k==0 or k==N):
            A[N,k] = 1
        else:
            A[N,k] = 0

        
    ###construction de s
    s=np.zeros(N+1)
    s[0]=0
    for i in range(N):
        panel_i = Panel(x[i], airfoil_fun[i], x[i+1], airfoil_fun[i+1])
        s[i+1]= s[i]+ panel_i.longueur 
    
    gamma = scipy.linalg.solve(A,b)

            
    ### Plot airfoil
    fig,ax = plt.subplots()
    
    # ax.plot(airfoil_data[:,0], airfoil_data[:,1], 'k', label = 'Airfoil (data)')
    #ax.plot(x/c, airfoil_fun/c, 'k.', label = 'Airfoil (fun)')
    # ax.plot(x_c/c, ml/c,'k:', label = 'Mean line')
    # ax.plot(x_c/c, t/c, 'k--', label = 'Thickness')
    print(A)
    ax.plot(s/c,gamma)
    
    # ax.plot(x/c, airfoil_fun/c, 'k', label = 'Airfoil ($y^*_c \pm y^*_t$)')
    # ax.plot(x_c/c, ml/c,'k:', label = 'Mean line ($y^*_c$)')
    # ax.plot(x_c/c, t/c, 'k--', label = 'Thickness ($y^*_t$)')
    
    ax.set_xlabel('s/c')
    ax.set_ylabel('gamma')  
    plt.show()
    
    ### Plot gamma
    
    # TODO