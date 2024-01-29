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


from a03_FIELD0_VORTX import velocity_field_Vortex
from a09_PRTCLE_FOKAS import maxey_riley_fokas

"""
Created on Wed Sep  7 16:49:30 2022

@author: cfg4065 (Julio Urizarna Carasa)

This script reproduces the figures from Example 5 in the paper
'Accurate Solution method for the Maxey-Riley equation, and the
effects of Basset history' by S. G. Prasath et al. (2019)

This example corresponds to a particle moving in a 2D velocity
field of a *single point vortex*. The velocity field is given
by:
    
    u = [-y^(2), y^(1)] * ( 1 / |y| ),
    
where y = [y^(1), y^(2)] is the position vector, y^(1) and
y^(2) its corresponding horizontal and vertical components,
respectively. |.| corresponds to the Euclidean norm.

The parameters are defined as described in the paper,
S = 0.01 and beta = 10. The initial velocities of the
particles are [0.05, 0.05]. The initial positions are
not given, but the distance between them (0.15) is given
instead.

We therefore decided to place all particles in the horizontal
axis, meaning:
    
    - first particle's initial position: [0.15, 0]
    - second particle's initial position: [0.30, 0]
    - third particle's initial position: [0.45, 0]
    ...

and so on.

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
    - The velocity field is defined in the 'a03_FIELD0_VORTX'
      file, whose class is imported.
"""

print("\nThis script could take up to 30 min... Grab a coffee and your favourite scientific paper.\n\n")

##################################
# Define function to parallelise #
##################################

# This function uses the 'update' method of a class instance
# when this is provided, together with the time axis.
# This will make more sense later on.
def compute_particle(particle, taxis):
    for tt in range(1, len(taxis)):
        particle.update()
        
    return (particle.tag, particle.pos_vec, particle.q_vec)

def save_to_file(sol_v):
    np.savetxt('pos_vec_' + str(sol_v[0]) + '.txt', sol_v[1])
    np.savetxt('vel_vec_' + str(sol_v[0]) + '.txt', sol_v[2])
    

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
tend  = 80.0
# Time nodes (should be modulated as tend and tini change)
nt    = 11001


# Create time vector
taxis  = np.linspace(tini, tend, nt)
dt     = taxis[1] - taxis[0]



############################################
# Define particle's and fluid's parameters #
############################################

# Densities (for the definition of beta = rho_p/rho_f):
# - Particle's density
rho_p   = 10.0

# - Fluid's density
rho_f   = 1.0


# Other parameters (for the definition of the Stokes number):
# St = rad_p**2 / (3 * nu_f * t_scale)
# - Particle's radius
rad_p   = np.sqrt(3)

# - Kinematic viscocity
nu_f    = 1.0

# - Time Scale of the flow
t_scale = 100.0



################################
# Import chosen velocity field #
################################

'''Vortex velocity field'''
vel     = velocity_field_Vortex()



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



##############################################
# Decide whether to apply parallel computing #
##############################################

parallel_flag = True
if parallel_flag == True:
    number_cores  = int(mp.cpu_count())



########################################
# Define particle's initial conditions #
########################################

# Vector of horizontal coordinate. As said, particles are placed
# on the horizontal axis.
x0_v   = np.array([])
for ii in range(0,5):
    x0   = 0.15 + 0.3 * ii
    x0_v = np.append(x0_v, x0)

y0     = 0.0
u0, v0 = 0.05, 0.05



#####################################################
# Create particles instances and save them in a set #
#####################################################

particle_dict = dict()
for nn in range(0, len(x0_v)):

    particle_dict[nn] = maxey_riley_fokas(nn+1,
                                          np.array([x0_v[nn], y0]),
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

if parallel_flag == True:
    if __name__ == '__main__':
        p = mp.Pool(number_cores)
        
        t0 = time.time()
        for nn in range(0, len(x0_v)):
            result  = p.apply_async(compute_particle,
                                    args=(particle_dict[nn],
                                          taxis),
                                    callback=save_to_file)
        
        p.close()
        p.join()
        tf = time.time()
        print('Calculation time: ' + str(tf - t0) + ' seconds.')
        
        for nn in range(0, len(x0_v)):
            particle_dict[nn].pos_vec = np.loadtxt('pos_vec_' + \
                                                   str(particle_dict[nn].tag) + \
                                                   '.txt')
            particle_dict[nn].q_vec = np.loadtxt('vel_vec_' + \
                                                  str(particle_dict[nn].tag) + \
                                                  '.txt')
            os.remove('pos_vec_' + str(particle_dict[nn].tag) + '.txt')
            os.remove('vel_vec_' + str(particle_dict[nn].tag) + '.txt')
            

else:
    for nn in progressbar(range(0, len(x0_v))):
        compute_particle(particle_dict[nn], taxis)



######################################################################
# Calculate Radial and Angular velocities as well as Radial distance #
######################################################################

rad_dis_dict = dict()
rad_vel_dict = dict()
ang_vel_dict = dict()
for nn in range(0, len(x0_v)):
    
    pos_v = np.array([particle_dict[nn].pos_vec[0]])
    vel_v = np.array([particle_dict[nn].q_vec[0]])
    for tt in range(1, len(taxis)):
        pos_v = np.vstack((pos_v,
                           particle_dict[nn].pos_vec[tt * (nodes_dt - 1)]))
        vel_v = np.vstack((vel_v,
                           particle_dict[nn].q_vec[tt * (nodes_dt - 1)]))
    
    # Radial distance
    rad_dis_v        = LA.norm(pos_v, axis=1)
    rad_dis_dict[nn] = rad_dis_v
    
    # Radial velocity
    rad_vel_v        = np.sum(np.multiply(pos_v, vel_v), axis=1) / rad_dis_v
    rad_vel_dict[nn] = rad_vel_v
    
    # Angular velocity
    ang_vel_v        = abs(np.cross(pos_v, vel_v, axisa=1, axisb=1) / rad_dis_v**2.0)
    ang_vel_dict[nn] = ang_vel_v



################
# Plot results #
################

#plt.rc('font', size=15)
rcParams['figure.figsize'] = 6.2, 2.1
fs = 9


fig, axs = plt.subplots(1, 2, layout='tight')
# Plot n° 1: Radial distance
axs[0].plot(taxis, rad_dis_dict[0], '-', color='b', linewidth=1.5, label="0.15")
axs[0].plot(taxis, rad_dis_dict[1], '-', color='y', linewidth=1.5, label='0.45')
axs[0].plot(taxis, rad_dis_dict[2], '-', color='g', linewidth=1.5, label='0.75')
axs[0].plot(taxis, rad_dis_dict[3], '-', color='r', linewidth=1.5, label='1.05')
axs[0].plot(taxis, rad_dis_dict[4], '-', color='m', linewidth=1.5, label='1.35')
line1 = np.linspace(1, tend, 101)
axs[0].plot(line1, 0.5*line1**0.25, '-', color='k')
axs[0].text(7, 0.45, "$t^{1/4}$", fontsize=fs)
axs[0].set_xlabel('t', fontsize=fs, labelpad=0.25)
axs[0].set_ylabel('Radial distance',  fontsize=fs, labelpad=0.25)
axs[0].tick_params(axis='both', labelsize=fs)
axs[0].set_xlim(1e-1, 1e2)
axs[0].set_ylim(1e-1, 10**0.5)
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].legend(loc='lower right', fontsize=fs, prop={'size':fs-2})
#axs[0].xticks(fontsize=fs)
axs[0].grid()

# Plot n° 2: Radial velocity
axs[1].plot(taxis, rad_vel_dict[0], '-', color='b', linewidth=1.5, label="0.15")
axs[1].plot(taxis, rad_vel_dict[1], '-', color='y', linewidth=1.5, label='0.45')
axs[1].plot(taxis, rad_vel_dict[2], '-', color='g', linewidth=1.5, label='0.75')
axs[1].plot(taxis, rad_vel_dict[3], '-', color='r', linewidth=1.5, label='1.05')
axs[1].plot(taxis, rad_vel_dict[4], '-', color='m', linewidth=1.5, label='1.35')
line2 = np.linspace(1, tend, 101)
axs[1].plot(line2, 0.5*line2**(-0.75), '-', color='k')
axs[1].text(10, 0.15, "$t^{-3/4}$")
axs[1].set_xlabel('t', fontsize=fs, labelpad=0.25)
axs[1].set_ylabel('Radial Velocity',  fontsize=fs, labelpad=0.25)
axs[1].tick_params(axis='both', labelsize=fs)
axs[1].set_xlim(1e-1, 6e2)
axs[1].set_ylim(1e-3, 1e1)
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].legend(loc='upper right', fontsize=fs, prop={'size':fs-2})
#axs[1].xticks(fontsize=fs)
axs[1].grid()

plt.savefig(save_plot_to + 'FIGURE_VORTEX.pdf',
            format='pdf', dpi=1200, bbox_inches='tight')

plt.show()

print("\007")