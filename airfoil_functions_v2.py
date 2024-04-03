# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:55:46 2024

@author: Home
"""

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
        self.angle = np.arctan2(y_end-y_start,x_end-x_start)
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

def influence_coefficients(panel_i, panel_j,i,j):
    X=panel_i.control[0]-panel_j.start[0]
    Y=panel_i.control[1]-panel_j.start[1]
    
    X1= X*np.cos(panel_j.angle) + Y*np.sin(panel_j.angle)
    Y1= -X*np.sin(panel_j.angle) + Y*np.cos(panel_j.angle)
    X2 = (panel_j.end[0]-panel_j.start[0])*np.cos(panel_j.angle) + (panel_j.end[1]-panel_j.start[1])*np.sin(panel_j.angle)
    
    D1 = np.sqrt(np.abs(X1**2 + Y1**2))
    D2 = np.sqrt(np.abs((X1-X2)**2 + Y1**2))
    angle1 = np.arctan2(Y1,X1)
    angle2 = np.arctan2(Y1,(X1-X2))
    if (i==j):
        Ua=-(X1-X2)/(2*X2)
        Ub=X1/(2*X2)
        Wa= -0.15916
        Wb= 0.15916
    else:
        Ua = -((Y1 * np.log(D2/D1)) + (X1 * (angle2-angle1)) - (X2* (angle2-angle1)))/(2*np.pi*X2)
        Ub = ((Y1 * np.log(D2/D1)) + (X1 * (angle2-angle1)))/(2*np.pi*X2)
        Wa = -((X2 - (Y1*(angle2-angle1))) - (X1 * np.log(D1/D2)) + (X2 * np.log(D1/D2))) / (2*np.pi*X2)
        Wb = ((X2 - (Y1*(angle2-angle1))) - (X1 * np.log(D1/D2))) / (2*np.pi*X2)
        
    U1 = Ua * np.cos(-panel_j.angle) + Wa*np.sin(-panel_j.angle)
    U2 = Ub * np.cos(-panel_j.angle) + Wb*np.sin(-panel_j.angle)
    W1 = -Ua * np.sin(-panel_j.angle) + Wa*np.cos(-panel_j.angle)
    W2 = -Ub * np.sin(-panel_j.angle) + Wb*np.cos(-panel_j.angle)
    
    return U1,U2,W1,W2

if __name__ == "__main__":
    airfoil_data = np.loadtxt('Airfoil-RevE-HC.dat')
    c = 2
    N = 200
    AoA = 5
    
    # Create x discretization with a change of variable to get more points on LE and TE
    theta = np.linspace(0, np.pi, N//2)
    x_c = c*(0.5)*(1-np.cos(theta))
    
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
    K=0
    for i in range(N):          
        panel_i = Panel(x[i], airfoil_fun[i], x[i+1], airfoil_fun[i+1])
        for j in range(N):
            panel_j = Panel(x[j], airfoil_fun[j], x[j+1], airfoil_fun[j+1])
            U1,U2,W1,W2 = influence_coefficients(panel_i, panel_j,i,j)
            if(j==0):
                A[i,j]=-U1*np.sin(panel_i.angle) + W1*np.cos(panel_i.angle)
                K = -U2*np.sin(panel_i.angle) + W2*np.cos(panel_i.angle)
            elif(j==N-1):
                A[i,j]=-U1*np.sin(panel_i.angle) + W1*np.cos(panel_i.angle) + K
                A[i,j+1] = -U2*np.sin(panel_i.angle) + W2*np.cos(panel_i.angle)
            else:
                A[i,j]=-U1*np.sin(panel_i.angle) + W1*np.cos(panel_i.angle) + K
                K = -U2*np.sin(panel_i.angle) + W2*np.cos(panel_i.angle)
                
        b[i] = np.cos(AoA)*np.sin(panel_i.angle) - np.sin(AoA)*np.cos(panel_i.angle)
    
    ##runge kutta
    A[N,0]= 1
    A[N,N]=1
    
    ##changement de variable##
    s=np.zeros(N+1)
    s[0]=0
    for i in range(N):
        panel_i = Panel(x[i], airfoil_fun[i], x[i+1], airfoil_fun[i+1])
        s[i+1]= s[i]+ panel_i.longueur 
    
    
    gamma = scipy.linalg.solve(A,-b)

            
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
    
    plt.figure()
    plt.plot(s/c,gamma,label="gamma")
    
    ### Plot gamma
    
    # TODO