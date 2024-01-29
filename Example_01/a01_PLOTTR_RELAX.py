#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('../src')
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from progressbar import progressbar
from pylab import rcParams

from a03_FIELD0_RELAX import velocity_field_Relaxing
from a09_PRTCLE_FOKAS import maxey_riley_fokas

"""
Created on Wed Sep  7 16:49:30 2022

@author: cfg4065

This script reproduces the figures from Example 1 in the paper
'Accurate Solution method for the Maxey-Riley equation, and the
effects of Basset history' by S. G. Prasath et al. (2019)

We provide a plot of a particle's velocity and trajectory on a
static fluid.

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

save_plot_to    = './VISUAL_OUTPUT/'



####################
# Define time grid #
####################

# Initial time
tini  = 0.0
# Final time
tend  = 15.0
# Time nodes
nt    = 151


# Create time axis
taxis  = np.linspace(tini, tend, nt)
dt     = taxis[1] - taxis[0]



############################################
# Define particle's and fluid's parameters #
############################################

# Densities:
# - Particle's density
rho_p_v = np.array([0.01, 1, 5])
label_v = ["Beta = 0.01", "Beta = 1", "Beta = 5"]

# - Fluid's density
rho_f   = 1.0   

# Particle's radius
rad_p   = np.sqrt(3)

# Kinematic viscocity
nu_f    = 1.0

# Time Scale of the flow
t_scale = 1.0



################################
# Import chosen velocity field #
################################

'''No background flow   velocity field'''
vel     = velocity_field_Relaxing()



########################################
# Define number of nodes per time step #
########################################

# Nodes in each time step as per Prasath et al. (2019) method #
nodes_dt = 20



###################################################
# Definition of the pseudo-spatial grid for Fokas #
###################################################

# Nodes in the frequency domain, k, as per Prasath et al. (2019)
N_fokas     = 101



########################################
# Define particle's initial conditions #
########################################

x0, y0 = 1.0, 0.0
u0, v0 = 1.0, 0.0

particle_dict = dict()
for nn in range(0,3):

    particle_dict[nn] = maxey_riley_fokas(nn,
                                          np.array([x0, y0]),
                                          np.array([u0, v0]),
                                          vel, N_fokas, tini, dt,
                                          nodes_dt,
                                          particle_density    = rho_p_v[nn],
                                          fluid_density       = rho_f,
                                          particle_radius     = rad_p,
                                          kinematic_viscosity = nu_f,
                                          time_scale          = t_scale)



#############################################################
# Calculate particles' trajectories and relative velocities #
#############################################################

velocity_dict   = dict()
trajectory_dict = dict()
for nn in progressbar(particle_dict):
    
    velocity   = np.array([u0])
    trajectory = np.array([x0])
    for tt in range(1, len(taxis)):
        particle_dict[nn].update()
        
        velocity   = np.append(velocity, particle_dict[nn].q_vec[tt * (nodes_dt-1),0])
        trajectory = np.append(trajectory, particle_dict[nn].pos_vec[tt * (nodes_dt-1),0])
    
    velocity_dict[nn]   = velocity
    trajectory_dict[nn] = trajectory



##################
# Generate plots #
##################

rcParams['figure.figsize'] = 6.2, 2.1
fs = 9

fig, axs = plt.subplots(1, 2, layout='tight')

axs[0].plot(taxis, velocity_dict[0], color='r', label='0.01', linewidth=1.5)
axs[0].plot(taxis, velocity_dict[1], color='g', label='1', linewidth=1.5)
axs[0].plot(taxis, velocity_dict[2], color='b', label='5', linewidth=1.5)
axs[0].set_xlim(0, 15)
axs[0].set_ylim(1e-5, 10)
axs[0].set_yscale("log")
axs[0].set_xlabel('t', fontsize=fs, labelpad=0.25)
axs[0].set_ylabel('$q^{(1)}(0,t)$', fontsize=fs, labelpad=0.25)
axs[0].legend(loc='lower left', fontsize=fs, prop={'size':fs-2})
axs[0].grid()

axs[1].plot(taxis, trajectory_dict[0], color='r', label='0.01', linewidth=1.5)
axs[1].plot(taxis, trajectory_dict[1], color='g', label='1', linewidth=1.5)
axs[1].plot(taxis, trajectory_dict[2], color='b', label='5', linewidth=1.5)
axs[1].set_xlim(0, 15)
axs[1].set_xlabel('$t$', fontsize=fs, labelpad=0.25)
axs[1].set_ylabel('$y^{(1)}$', fontsize=fs, labelpad=0.25)
axs[1].legend(loc='upper left', fontsize=fs, prop={'size':fs-2})
axs[1].grid()

plt.savefig(save_plot_to + 'FIGURE_RELAXING.pdf', format='pdf', dpi=500)

plt.show()

print("\007")
