#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import time
sys.path.append('../src')
import numpy as np
import numpy.linalg as LA
import multiprocessing as mp
import matplotlib.pyplot as plt
from progressbar import progressbar
from pylab import rcParams

from a03_FIELD0_DATA1 import velocity_field_Faraday1
from a09_PRTCLE_FOKAS import maxey_riley_fokas

"""
Created on Wed Sep  7 16:49:30 2022

@author: cfg4065 (Julio Urizarna Carasa)

This script provides the plot of a particle's trajectory on a 
real, experimental flow calculated with the method provided in the
paper 'Accurate Solution method for the Maxey-Riley equation, and
the effects of Basset history' by S. G. Prasath et al. (2019)

This example corresponds to a particle moving in a 2D experimental
flow driven by Faraday waves. More information about the experiment
can be found in:

    - Colombi, R., Rohde, N., Schlüter, M., & von Kameke, A. (2022).
      "Coexistence of inverse and direct energy cascades in faraday
      waves". Fluids, 7(5), 148.
    - Colombi, R., Schlüter, M., & Kameke, A. V. (2021). "Three
      dimensional flows beneath a thin layer of 2D turbulence induced
      by Faraday waves". Experiments in Fluids, 62(1), 1-13.

The parameters are defined as follows:
    - Stokes Number, S = 0.01 and
    - beta parameter, beta = 10.
    
The initial conditions are:
    - Initial position:
    - Initial velocity:

Please be aware that:
    - all plots are printed as PDFs and saved into the folder
      '02_VSUAL_OUTPUT',
    - the nonlinear solving process may not converge if two
      consecutive time nodes are too close to each other (for
      example if WAY TOO MUCH of a bigger amount of nodes is
      provided). This could be addressed by either
      (1) changing the nonlinear solver, (2) increasing the
      number of maximum iterations or (3) decreasing the
      tolerance. These parameters can be changed in the
      'a09_PRTCLE_FOKAS script', under the 'update' method.
    - The velocity field is defined in the 'a03_FIELD0_DATA1'
      file, whose class is imported.
"""
    

####################################
# Define Folder where to save data #
####################################

save_plot_to    = './02_VSUAL_OUTPUT/'



####################
# Define time grid #
####################

# Initial time
tini  = 0.0
# Final time
tend  = 5.0
# Time nodes (should be modulated as tend and tini change)
nt    = 501


# Create time vector
taxis  = np.linspace(tini, tend, nt)
dt     = taxis[1] - taxis[0]



############################################
# Define particle's and fluid's parameters #
############################################

# Densities (for the definition of beta = rho_p/rho_f):
# - Particle's density
rho_p   = 2.0

# - Fluid's density
rho_f   = 1.0


# Other parameters (for the definition of the Stokes number):
# St = rad_p**2 / (3 * nu_f * t_scale)
# - Particle's radius
rad_p   = np.sqrt(3)

# - Kinematic viscocity
nu_f    = 1.0

# - Time Scale of the flow
t_scale = 10.0



################################
# Import chosen velocity field #
################################

'''Faraday1 velocity field'''
vel     = velocity_field_Faraday1(field_boundary=False)


##########################
# Set bounds of the plot #
##########################

'''Bounds for Data-based field'''
x_left  = 0.0
x_right = 0.07039463189473738
y_down  = 0.0
y_up    = 0.0524872255355498


##############################################################
# Define spatial grid for the plotting of the Velocity Field #
##############################################################

nx = 50
ny = 51

xaxis = np.linspace(x_left, x_right, nx)
yaxis = np.linspace(y_down, y_up, ny)

X, Y = np.meshgrid(xaxis, yaxis)



########################################
# Define number of nodes per time step #
########################################

# Nodes in each time step as per Prasath et al. (2019) method #
# This means that between any two time nodes, taxis[i] and taxis[ii],
# we have 'nodes_dt' Chebyshev nodes. This is part of how the how
# the method is defined.
nodes_dt = 20  # DO NOT CHANGE (in principle)



###################################################
# Definition of the pseudo-spatial grid for Fokas #
###################################################

# Nodes in the frequency domain, k, as per Prasath et al. (2019) #
# These are the number of nodes in the pseudo-space. This
# is also part of the method. 
N_fokas     = 101  # DO NOT CHANGE.



########################################
# Define particle's initial conditions #
########################################

x0, y0 = 0.02, 0.02
u0, v0 = 0.0015, 0.0015



#####################################################
# Create particles instances and save them in a set #
#####################################################

particle = maxey_riley_fokas(1,np.array([x0, y0]),
                               np.array([u0, v0]),
                               vel, N_fokas, tini, dt,
                               nodes_dt,
                               particle_density    = rho_p,
                               fluid_density       = rho_f,
                               particle_radius     = rad_p,
                               kinematic_viscosity = nu_f,
                               time_scale          = t_scale)



#############################################################
# Calculate particles' trajectories and relative velocities #
#############################################################

for tt in progressbar(range(1, len(taxis))):
    particle.update()



##############################
# Plot particle's trajectory #
##############################

rcParams['figure.figsize'] = 6.2, 2.1
fs = 9

fig, axs = plt.subplots(1, 2, layout='tight')

# Plot background field
u, v = vel.get_velocity(X, Y, taxis[-1])
axs[0].quiver(X, Y, u, v)
# Plot particle's trajectory
axs[0].plot(particle.pos_vec[:,0], particle.pos_vec[:,1], '-', color='g', linewidth=1.5)
axs[0].set_xlabel('$y^{(1)}$', fontsize=fs, labelpad=0.25)
axs[0].set_ylabel('$y^{(2)}$', fontsize=fs, labelpad=0.25)
axs[0].set_xlim([0.017, 0.025])
axs[0].set_ylim([0.019, 0.031])


# Plot particle's trajectory
vel_vec = np.array([LA.norm(particle.q_vec[0])])
for tt in range(1, len(taxis)):
    vel_vec = np.append(vel_vec, LA.norm(particle.q_vec[tt * (nodes_dt - 1)]))
axs[1].plot(taxis, vel_vec, '-', color='b', linewidth=1.5)
axs[1].set_xlabel('t', fontsize=fs, labelpad=0.25)
axs[1].set_ylabel('Relative velocity', fontsize=fs, labelpad=0.25)
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].grid()
plt.savefig(save_plot_to + 'c01_FIGURE_DATA1.pdf', format='pdf', dpi=500)

plt.show()

print("\007")
