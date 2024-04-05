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

def psi(x, y, alpha, beta, b):
    def intermediate(x, y, alpha, beta, s):
        return ((alpha * x + beta) * (s * np.log(s**2 + y**2) - 2 * s + 2 * y * np.arctan(s / y)) - (alpha / 2) * ((s**2 + y**2) * np.log(s**2 + y**2) - s**2))
    return (1 / (4 * np.pi)) * (intermediate(x, y, alpha, beta, x-b) - intermediate(x, y, alpha, beta, x+b))
        
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
    ### Setup the influence matrix and the RHS
    b = np.zeros(N+1)
    A = np.zeros((N+1,N+1))

    U_inf_array = np.array(([U_inf*np.cos(alpha), U_inf*np.sin(alpha)]))

    ### We impose the condition on the normal velocity for each panel to be zero + contribution of the freestream
    for i in range(N):          
        panel_i = panels[i]
        for j in range(N):
            panel_j = panels[j]

            A[i,j] += influence_coefficients(panel_i, panel_j)[0]
            A[i,j+1] += influence_coefficients(panel_i, panel_j)[1]

        b[i] = -np.dot(panel_i.normal, U_inf_array)

    ### Kutta condition
    A[N,0] = 1      # Gamma 1
    A[N,N] = 1     # Gamma n+1
    b[N] = 0

    return scipy.linalg.solve(A,-b)/10000     # TODO facteur et -

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
            
    # Initialization of the panels
    panels = [Panel(i, x[i], airfoil_fun[i], x[i+1], airfoil_fun[i+1]) for i in range(N)]       
    
    # Changement de variable##
    s=np.zeros(N+1)
    s[0]=0
    for i in range(N):
        panel_i = panels[i]
        s[i+1]= (s[i]+ panel_i.norm) 
    
    ### Results

    alphas_degre = [4.0*i for i in range(3)]
    alphas_radian = [alphas_degre[i] * np.pi/180 for i in range(len(alphas_degre))]
    print(alphas_degre, alphas_radian)
    gammas = np.zeros((len(alphas_radian), N+1))
    
    
    #########
    # Plots #
    #########
    
    ### Plot airfoil
    
    fig,ax = plt.subplots()
    
    ax.plot(airfoil_data[:,0], airfoil_data[:,1], 'k', label = 'Airfoil (data)')
    ax.plot(x/c, airfoil_fun/c, 'k.', label = 'Airfoil (fun)')
    ax.plot(x_c/c, ml/c,'k:', label = 'Mean line')
    ax.plot(x_c/c, t/c, 'k--', label = 'Thickness')
    
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
    
    for i in range(len(alphas_radian)):
        gammas[i] = circulation(alphas_radian[i], U_inf)               
        plt.plot(s/c, gammas[i]/U_inf, label=f"alpha = {alphas_degre[i]}°")

    plt.xlabel(r'$\frac{s}{c}$', fontsize=13)
    plt.ylabel(r'$\frac{\Gamma}{U_{\infty}}$', fontsize=13)
    plt.title('Circulation distribution along the profile')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    ### Plot C_p
    
    for i in range(len(alphas_radian)):
        plt.plot(x/c,-(1-(gammas[i]/U_inf)**2), label=f"alpha = {alphas_degre[i]}°")

    plt.xlabel(r'$\frac{x}{c}$', fontsize=13)
    plt.ylabel(r'$C_p$', fontsize=13)
    plt.title('Pressure coefficient as a function of x/c')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    for i in range(len(alphas_radian)):
        plt.plot(s/c,(1-(gammas[i]/U_inf)**2), label=f"alpha = {alphas_degre[i]}°")

    plt.xlabel(r'$\frac{s}{c}$', fontsize=13)
    plt.ylabel(r'$C_p$', fontsize=13)
    plt.title('Pressure coefficient as a function of s/c')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    ### Plot lift coefficient
    
    c_l_coefficients = np.zeros_like(alphas_radian)
    for i in range(len(alphas_radian)):
        gamma_tot = sum((gammas[i][j] + gammas[i][j+1]) * panels[j].b for j in range(N))
        c_l_coefficients[i] = -gamma_tot/(1/2 * U_inf * c)
        
    # We add a linear fit to see if it's linear or not

    fit_coeffs = np.polyfit(alphas_degre, c_l_coefficients, 1)
    fit_line = np.poly1d(fit_coeffs)
    fit_line_values = fit_line(alphas_degre)

    plt.plot(alphas_degre, c_l_coefficients, 'k.', label='Potential flow results')
    plt.plot(alphas_degre, fit_line_values, linestyle='--', color='red', label='Linear Fit')

    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel(r'$C_L$')
    plt.title('Lift coefficient as a function of the angle of attack')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    
    # ### Plot streamlines
    
    # alpha = np.zeros(N)
    # beta = np.zeros(N)

    # for i in range(N):
    #     alpha[i] = (gammas[0][i+1] - gammas[0][i])/(2*panels[i].b)
    #     beta[i] = (gammas[0][i] + gammas[0][i+1])/2

    # x_min, x_max = -0.5, 1.5
    # y_min, y_max = -0.5, 0.5
    # num_points = 100
    # x_mesh = np.linspace(x_min, x_max, num_points)
    # y_mesh = np.linspace(y_min, y_max, num_points)
    # X, Y = np.meshgrid(x_mesh, y_mesh)
    # # Psi = np.zeros_like(X)
    # u_mesh = np.zeros((num_points, num_points))
    # v_mesh = np.zeros((num_points, num_points))

    # for i in range(num_points):  # Iteration on the x points of the mesh
    #     for j in range(num_points):  # Iteration on the y points of the mesh
    #         for k in range(N):  # Effect of each panel on a point (i,j)
    #             panel_l = Panel(k, x_mesh[i]-1, y_mesh[i]-1, x_mesh[i]+1, y_mesh[i]+1)          # TODO : vérifier les coordonnées du panel
    #             x_temp, y_temp = calculate_coordinates(panel_l, panels[k])
    #             # Psi[i, j] += psi(x_temp, y_temp, alpha[k], beta[k], panels[k].b)
    #             u_mesh[i,j] += influence_coefficients(panel_l, panels[k])[0] * gammas[0][k] + influence_coefficients(panel_l, panels[k])[1] * gammas[0][k+1]
    #             # v_mesh[i,j] += v(x_temp, y_temp, panels[k].b)[0] * gammas[0][k] + v(x_temp, y_temp, panels[k].b)[1] * gammas[0][k+1]

    # plt.streamplot(X, Y, np.cos(u_mesh), np.sin(u_mesh), density=2)
    # plt.plot(x/c, airfoil_fun/c, 'k', label = 'Airfoil (fun)')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Streamlines')
    # plt.axis('equal')
    # plt.show()