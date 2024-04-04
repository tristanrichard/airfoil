#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:14:38 2024

@author: jcrismer
"""

import math
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
    def __init__(self, i, x_start, y_start, x_end, y_end):
        self.index = i
        self.x_start, self.y_start = x_start, y_start
        self.x_end, self.y_end = x_end, y_end
        self.x_control, self.y_control = 1/2 * (x_start+x_end), 1/2 * (y_start+y_end)
        self.dx, self.dy = (x_end-x_start), (y_end-y_start)
        self.norm = math.sqrt((x_end-x_start)**2 + (y_end-y_start)**2)
        self.b = self.norm/2
        self.angle = np.arctan2(self.dy, self.dx)
        
        # Normal and tangent vector of the panel
        self.normal = np.array([-self.dy, self.dx])
        self.tangent = np.array([self.dx, self.dy])
        
        # We normalize to get unit vectors
        self.normal /= self.norm
        self.tangent /= self.norm
        
def circulation(alpha, beta, x):
    return beta + alpha * x
        
def u(x, y, b):
    coeff_gamma_i = -y/2 * 1/(2*b) * (np.log(((x-b)**2 + y**2)/((x+b)**2 + y**2))) + (-x/(2*b) + 1/2) * (np.arctan((x+b)/y) - np.arctan((x-b)/y))
    coeff_gamma_i_1 = y/2 * 1/(2*b) * (np.log(((x-b)**2 + y**2)/((x+b)**2 + y**2))) + (x/(2*b) + 1/2) * (np.arctan((x+b)/y) - np.arctan((x-b)/y))
    
    coeff_gamma_i *= -1/(2 * np.pi)
    coeff_gamma_i_1 *= -1/(2 * np.pi)
    
    return coeff_gamma_i, coeff_gamma_i_1

def v(x, y, b):
    coeff_gamma_i = -1/(2*b) * (2*b + y * (np.arctan((x-b)/y) - np.arctan((x+b)/y))) + 1/2 * (-x/(2*b) + 1/2) * np.log(((x-b)**2 + y**2)/((x+b)**2 + y**2))
    coeff_gamma_i_1 = 1/(2*b) * (2*b + y * (np.arctan((x-b)/y) - np.arctan((x+b)/y))) + 1/2 * (x/(2*b) + 1/2) * np.log(((x-b)**2 + y**2)/((x+b)**2 + y**2))
    
    coeff_gamma_i *= -1/(2 * np.pi)
    coeff_gamma_i_1 *= -1/(2 * np.pi)
    
    return coeff_gamma_i, coeff_gamma_i_1

def own_influence(x, b):
    coeff_gamma_i = -1 + (-x/(2*b) + 1/2) * np.log((b-x)/(b+x))
    coeff_gamma_i_1 = 1 + (x/(2*b) + 1/2) * np.log((b-x)/(b+x))
    
    coeff_gamma_i *= -1/(2 * np.pi)
    coeff_gamma_i_1 *= -1/(2 * np.pi)
    
    return coeff_gamma_i, coeff_gamma_i_1

def calculate_coordinates(panel_i, panel_j):
    # Calculate the coordinates of the control point of the panel_i in the frame of reference of panel_j, in order to use them in u and v
    delta_x = panel_i.x_control - panel_j.x_control
    delta_y = panel_i.y_control - panel_j.y_control
    theta_control = np.arctan2(delta_y, delta_x)
    
    distance = math.sqrt(delta_x**2 + delta_y**2)
    
    angle = panel_j.angle - theta_control
    
    x = np.cos(angle) * distance
    y = np.sin(angle) * distance
    
    return x,y

def influence_coefficients(panel_i, panel_j):
    ### Calculate the influence of the panel j on the panel i
    x,y = calculate_coordinates(panel_i, panel_j)
    
    if (panel_i.index == panel_j.index):
        coeff_u_gamma_i, coeff_u_gamma_i_1 = 0, 0
        coeff_v_gamma_i, coeff_v_gamma_i_1 = own_influence(x, panel_j.b)
    else:
        coeff_u_gamma_i, coeff_u_gamma_i_1 = u(x, y, panel_j.b)
        coeff_v_gamma_i, coeff_v_gamma_i_1 = v(x, y, panel_j.b)
        
    coeff_gamma_j = np.dot(panel_i.normal, coeff_u_gamma_i * panel_j.tangent) + np.dot(panel_i.normal, coeff_v_gamma_i * panel_j.normal)
    coeff_gamma_j_1 = np.dot(panel_i.normal, coeff_u_gamma_i_1 * panel_j.tangent) + np.dot(panel_i.normal, coeff_v_gamma_i_1 * panel_j.normal)
    
    return coeff_gamma_j, coeff_gamma_j_1

def circulation(alpha, U_inf):
    ### Setup of the influence matrix and the RHS
    b = np.zeros(N+1)
    A = np.zeros((N+1,N+1))
    
    ### We impose the condition on the normal velocity for each panel to be zero
    for i in range(N):          
        panel_i = panels[i]
        for j in range(N):
            panel_j = panels[j]
                
            A[i,j] += influence_coefficients(panel_i, panel_j)[0]
            A[i,j+1] += influence_coefficients(panel_i, panel_j)[1]
            
        b[i] = U_inf * (np.cos(alpha)*np.sin(panel_i.angle) - np.sin(alpha)*np.cos(panel_i.angle))
        
    ### Kutta condition
    A[N,0] = 1      # Gamma 1
    A[N,-1] = 1     # Gamma n+1
    b[-1] = 0
    
    gamma = scipy.linalg.solve(A,b)
    
    return gamma/(U_inf*50000)      # TODO check le facteur

if __name__ == "__main__":
    airfoil_data = np.loadtxt('Airfoil-RevE-HC.dat')
    c = 2
    N = 200
    U_inf = 1.0
    
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
            
    ### Initialization of the panels
    panels = [Panel(i, x[i], airfoil_fun[i], x[i+1], airfoil_fun[i+1]) for i in range(N)]       
    
    ## Changement de variable##
    s=np.zeros(N+1)
    s[0]=0
    for i in range(N):
        panel_i = panels[i]
        s[i+1]= (s[i]+ panel_i.norm/c) 

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
    
    ### Results
    
    gamma_5 = circulation(5.0,U_inf)
    print(gamma_5[0])
    print(gamma_5[-1])
    gamma_10 = circulation(10.0,U_inf)
    gamma_12_5 = circulation(12.5,U_inf)
    gamma_15 = circulation(15.0,U_inf)
    gamma_20 = circulation(20.0,U_inf)
    
    ### Plot gamma
    
    plt.plot(s, gamma_5, label="alpha = 5°")            # TODO: vérifier le facteur
    # plt.plot(s,gamma_10, label="alpha = 10°")
    # plt.plot(s,gamma_12_5, label="alpha = 12.5°")
    # plt.plot(s,gamma_15, label="alpha = 15°")
    # plt.plot(s,gamma_20, label="alpha = 20°")
    plt.legend(loc='upper right')
    plt.show()
    
    ### Plot C_p
    
    plt.plot(x[N//4:-N//4],(1-(gamma_5[N//4:-N//4])**2))     # TODO: vérifier le facteur, le signe et le s
    plt.show()
    
    ### Plot streamlines
    
    # TODO
    
    ### Lift coefficient
    
    ### We find the total circulation
    gamma_tot_5 = sum((gamma_5[i] + gamma_5[i+1]) * panels[i].b for i in range(N))
    C_l = -gamma_tot_5/(1/2 * U_inf * c)
    print(C_l)