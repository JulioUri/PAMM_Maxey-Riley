#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('../src')
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from progressbar import progressbar
from pylab import rcParams

from a03_FIELD0_COUET import velocity_field_Couette
from a09_PRTCLE_FOKAS import maxey_riley_fokas

"""
Created on Wed Sep  7 16:08:30 2022

@author: cfg4065

This script reproduces the figures from Example 4 in the paper
'Accurate Solution method for the Maxey-Riley equation, and the
effects of Basset history' by S. G. Prasath et al. (2019)

We provide plots of a particles' velocity and trajectory on a
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

save_plot_to    = './02_VSUAL_OUTPUT/'



####################
# Define time grid #
####################

# Initial time
tini    = 0.0
# Final time
tend_v  = np.array([1.5, 20.0])
# Time nodes
nt_v    = np.array([351, 201])


# Create time axis, for the example S = 0.01
taxis1  = np.linspace(tini, tend_v[0], nt_v[0])
dt1     = taxis1[1] - taxis1[0]

# Create time axis, for the example S = 1
taxis2  = np.linspace(tini, tend_v[1], nt_v[1])
dt2     = taxis2[1] - taxis2[0]



############################################
# Define particle's and fluid's parameters #
############################################

# Densities:
# - Particle's density
rho_p   = 5.0

# - Fluid's density
rho_f   = 1.0   

# Particle's radius
rad_p   = np.sqrt(3)

# Kinematic viscocity
nu_f    = 1.0

# Time Scale of the flow
t_scale_v = np.array([100.0, 1.0])



################################
# Import chosen velocity field #
################################

'''Couette velocity field'''
vel     = velocity_field_Couette()



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



##############################################
# Decide whether to apply parallel computing #
##############################################

parallel_flag = True
number_cores  = int(mp.cpu_count())



########################################
# Define particle's initial conditions #
########################################

x0, y0    = 0.0, 0.0
u0, v0    = 1.0, 1.0



####################
# Create particles #
####################

particle1 = maxey_riley_fokas(1, np.array([x0, y0]),
                                 np.array([u0, v0]),
                                 vel, N_fokas, tini, dt1,
                                 nodes_dt,
                                 particle_density    = rho_p,
                                 fluid_density       = rho_f,
                                 particle_radius     = rad_p,
                                 kinematic_viscosity = nu_f,
                                 time_scale          = t_scale_v[0])
    
particle2 = maxey_riley_fokas(2, np.array([x0, y0]),
                                 np.array([u0, v0]),
                                 vel, N_fokas, tini, dt2,
                                 nodes_dt,
                                 particle_density    = rho_p,
                                 fluid_density       = rho_f,
                                 particle_radius     = rad_p,
                                 kinematic_viscosity = nu_f,
                                 time_scale          = t_scale_v[1])



##############################################################
# Calculate the trajectories by using update method in class #
##############################################################

pos1_vec = np.array([x0, y0])
pos2_vec = np.array([x0, y0])
q0_1_vec = np.array([u0, v0])
q0_2_vec = np.array([u0, v0])
for tt in progressbar(range(1, len(taxis1))):
    particle1.update()
    pos1_vec = np.vstack((pos1_vec, particle1.pos_vec[tt * (nodes_dt-1)]))
    q0_1_vec = np.vstack((q0_1_vec, particle1.q_vec[tt * (nodes_dt-1)]))


for tt in progressbar(range(1, len(taxis2))):
    particle2.update()
    pos2_vec = np.vstack((pos2_vec, particle2.pos_vec[tt * (nodes_dt-1)]))
    q0_2_vec = np.vstack((q0_2_vec, particle2.q_vec[tt * (nodes_dt-1)]))



##################
# Generate plots #
##################

rcParams['figure.figsize'] = 6.2, 4.2
fs = 9

fig, axs = plt.subplots(2, 2, layout='tight')
# Trajectory plot for S=0.01
axs[0,0].plot(pos1_vec[:,0], pos1_vec[:,1], color='g', linewidth=1.5)
axs[0,0].set_xlabel('$y^{(1)}(t)$', fontsize=fs, labelpad=0.25)
axs[0,0].set_ylabel('$y^{(2)}(t)$', fontsize=fs, labelpad=0.25)
axs[0,0].tick_params(axis='both', labelsize=fs)
axs[0,0].set_xlim(0.0, 0.1)
axs[0,0].set_ylim(0.0, 4.0e-2)
axs[0,0].grid()

# Velocity plot for S=0.01
axs[0,1].plot(taxis1, q0_1_vec[:,0], color='r', label='$q^{(1)}(0,t)$', linewidth=1.5)
axs[0,1].plot(taxis1, q0_1_vec[:,1], color='b', label='$q^{(2)}(0,t)$', linewidth=1.5)
axs[0,1].set_xlabel('t', fontsize=fs, labelpad=0.25)
axs[0,1].set_ylabel('Velocity', fontsize=fs, labelpad=0.25)
axs[0,1].tick_params(axis='both', labelsize=fs)
axs[0,1].set_xlim(0, 0.2)
axs[0,1].set_ylim(-0.5, 1.0)
axs[0,1].legend(loc='upper right', fontsize=fs)
axs[0,1].grid()

# Trajectory plot for S=1
axs[1,0].plot(pos2_vec[:,0], pos2_vec[:,1], color='g', linewidth=1.5)
axs[1,0].set_xlabel('$y^{(1)}(t)$', fontsize=fs, labelpad=0.25)
axs[1,0].set_ylabel('$y^{(2)}(t)$', fontsize=fs, labelpad=0.25)
axs[1,0].tick_params(axis='both', labelsize=fs) 
axs[1,0].set_xlim(0.0, 20)
axs[1,0].set_ylim(0.0, 4.0)
axs[1,0].grid()

# Velocity plot for S=1
axs[1,1].plot(taxis2, q0_2_vec[:,0], color='r', label="$q^{(1)}(0,t)$", linewidth=1.5)
axs[1,1].plot(taxis2, q0_2_vec[:,1], color='b', label="$q^{(2)}(0,t)$", linewidth=1.5)
axs[1,1].set_xlabel('t', fontsize=fs, labelpad=0.25)
axs[1,1].set_ylabel('Velocity', fontsize=fs, labelpad=0.25)
axs[1,1].tick_params(axis='both', labelsize=fs)
axs[1,1].set_xlim(0, 20)
axs[1,1].set_ylim(-1.05, 1.0)
axs[1,1].legend(loc='upper right', fontsize=fs)
axs[1,1].grid()

plt.savefig(save_plot_to + 'c01_FIGURE_COUETT.pdf', format='pdf', dpi=500)

plt.show()

print("\007")