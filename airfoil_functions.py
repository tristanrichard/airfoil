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

### Functions for the geometry of the airfoil
        
def thickness(x, a, b, c, d, e):
    return (a*np.sqrt(x) + b*x + c*x**2 + d*x**3 + e*x**4)

def mean_line(x, a, b, c, d, e):
    return (a + b*x + c*x**2 + d*x**3 + e*x**4)

def get_intra_extra(x, c, airfoil):
    if airfoil == 'RevE-HC':
        a_ml, b_ml, c_ml, d_ml, e_ml = 0.0, 0.50511, -0.97016, 0.79535, -0.3303
        a_t, b_t, c_t, d_t, e_t = 0.39981, -0.09876, -0.6376, 0.34907, -0.01252
        
        y_intra = c*mean_line(x/c, a_ml, b_ml, c_ml, d_ml, e_ml) - c*thickness(x/c, a_t, b_t, c_t, d_t, e_t)
        y_extra = c*mean_line(x/c, a_ml, b_ml, c_ml, d_ml, e_ml) + c*thickness(x/c, a_t, b_t, c_t, d_t, e_t)
        
    else:
        print('This airfoil is unknown : ', airfoil)
        
    return y_intra, y_extra

### Functions for the influence matrix and RHS

class Panel():
    def __init__(self, x_start, y_start, x_end, y_end):
        self.start = (x_start, y_start)
        self.end = (x_end, y_end)
        self.control = (1/2 * (x_start+x_end), 1/2 * (y_start+y_end))
        self.length = (np.abs(x_end-x_start), np.abs(y_end-y_start))
        self.norm = np.linalg.norm(np.array([self.length[0], self.length[1]]))
        self.angle = np.arctan2(self.length[1], self.length[0])
        
        # Normal and tangent vector of the panel
        self.normal = np.array([-self.length[1], self.length[0]])
        self.tangent = np.array([self.length[0], self.length[1]])
        
        # We normalize to get unit vectors
        self.normal /= np.linalg.norm(self.normal)
        self.tangent /= np.linalg.norm(self.tangent)
        
def circulation(alpha, beta, x):
    return beta + alpha * x
        
def u(x, y, b):
    coeff_alpha = y/2 * (np.log(((x-b)**2 + y**2)/((x+b)**2 + y**2))) + x * (np.arctan((x+b)/y) - np.arctan((x-b)/y))
    coeff_beta = np.arctan((x+b)/y) - np.arctan((x-b)/y)
    
    coeff_alpha *= -1/(2 * np.pi)
    coeff_beta *= -1/(2 * np.pi)
    
    return coeff_alpha, coeff_beta

def v(x, y, b):
    coeff_alpha = 2*b + y * (np.arctan((x-b)/y) - np.arctan((x+b)/y)) + 1/2 * np.log(((x-b)**2 + y**2)/((x+b)**2 + y**2))
    coeff_beta = 1/2 * np.log(((x-b)**2 + y**2)/((x+b)**2 + y**2))
    
    coeff_alpha *= -1/(2 * np.pi)
    coeff_beta *= -1/(2 * np.pi)
    
    return coeff_alpha, coeff_beta

def influence_coefficients(panel_i, panel_j):
    # Calculate the influence of the panel j on the panel i
    
    if (panel_i == panel_j):
        alpha_j = -1/(2*np.pi) * panel_i.norm + panel_i.control[0] * np.log(np.abs(panel_i.norm/2.0 - panel_i.control[0])/np.abs(panel_i.norm/2.0 + panel_i.control[0]))
        beta_j = -1/(2*np.pi) * np.log(np.abs(panel_i.norm/2.0 - panel_i.control[0])/np.abs(panel_i.norm/2.0 + panel_i.control[0]))
    
    else:
        delta_xc = np.abs(panel_i.control[0] - panel_j.control[0])
        delta_yc = np.abs(panel_i.control[1] - panel_j.control[1])
        
        theta_c = np.arctan2(delta_yc, delta_xc)
        
        x = np.linalg.norm(np.array([delta_xc, delta_yc])) * np.cos(theta_c - panel_j.angle)     # Pas sûr de ça 
        y = np.linalg.norm(np.array([delta_xc, delta_yc])) * np.sin(theta_c - panel_j.angle)
        
        alpha_u, beta_u = u(x, y, 1/2 * panel_j.norm)
        alpha_v, beta_v = v(x, y, 1/2 * panel_j.norm)
        
        alpha_j = np.dot(np.array([alpha_u, alpha_v]), panel_i.normal)
        beta_j = np.dot(np.array([beta_u, beta_v]), panel_i.normal)
    
    return alpha_j, beta_j

if __name__ == "__main__":
    airfoil_data = np.loadtxt('Airfoil-RevE-HC.dat')
    c = 2
    N = 100
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
    
    ### Initialisation of the panels
    panels = [Panel(x[i], airfoil_fun[i], x[i+1], airfoil_fun[i+1]) for i in range(N)]
    
    ### Setup of the influence matrix and the RHS
    b = np.zeros(2*N)
    A = np.zeros((2*N,2*N))
    
    ### We impose the condition on the normal velocity for each panel to be zero, the continuity equations and the Kutta condition
    for i in range(N):          
        panel_i = panels[i]
        for j in range(N-1):
            panel_j = panels[j]
            
            if (i == j):
                A[i+N,j] = 1/2 * panel_i.norm
                A[i+N,j+1] = 1/2 * panel_j.norm
                A[i+N,j+N] = 1
                A[i+N,j+N+1] = -1
                
            A[i,j] = influence_coefficients(panel_i, panel_j)[0]
            A[i,j+N] = influence_coefficients(panel_i, panel_j)[1]
        
        A[i,N-1] = influence_coefficients(panel_i, panel_j)[0]
        A[i,2*N-1] = influence_coefficients(panel_i, panel_j)[1]
            
        b[i] = np.cos(AoA)*np.sin(panel_i.angle) - np.sin(AoA)*np.cos(panel_i.angle)
        
    ### Kutta condition
    A[2*N-1,0] = -panels[0].norm
    A[2*N-1,N-1] = panels[N-1].norm
    A[2*N-1,N] = 1
    A[2*N-1,2*N-1] = 1
        
    ## Changement de variable##
    s=np.zeros(N+1)
    s[0]=0
    for i in range(N):
        panel_i = Panel(x[i], airfoil_fun[i], x[i+1], airfoil_fun[i+1])
        s[i+1]= (s[i]+ panel_i.norm) 
    
    linear_coefficients = scipy.linalg.solve(A,b)
    
    gamma = np.zeros(N+1)
    for i in range(N):
        gamma[i] = circulation(linear_coefficients[i], linear_coefficients[i+N], x[i]/c)
    gamma[-1] = -gamma[0]

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
    
    ### Plot gamma
    
    plt.plot(s,gamma)
    plt.show()