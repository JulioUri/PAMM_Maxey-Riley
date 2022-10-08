import numpy as np

'''
This script defines all the parameters needed for solving the MRE.

The name of the parameters is kept as per Prasath et al. (2019).
Please check the paper for further reference.
'''

class mr_parameter(object):

  def set_beta(self):
    # Beta, parameter that relates the densities of the paritcle and the fluid
    self.beta = self.rho_p /self.rho_f
  
  def set_S(self):
    # Stokes number
    self.S = (1.0/3.0)*self.a**2/(self.nu * self.T)
  
  def set_R(self):
    # Parameter R, that uses beta
    self.R = (1.0 + 2.0*self.beta)/3.0
    
  def set_alpha(self):
    # Parameter alpha, used in the reformulation. This is the coefficient in
    # front of the Stokes drag.
    self.alpha = 1 / (self.R * self.S) 

  def set_gamma(self):
    # Parameter gamma, used in the reformulation. This is the coefficient in
    # front of the Basset History Term.
    self.gamma = (1 / self.R) * np.sqrt(3 / self.S)
  
  def __init__(self, particle_density, fluid_density, particle_radius, kinematic_viscosity, time_scale):
    # Class initialization.
    self.rho_p = particle_density
    self.rho_f = fluid_density
    self.a     = particle_radius
    self.nu    = kinematic_viscosity
    self.T     = time_scale

    self.set_beta()
    self.set_S()
    self.set_R()
    self.set_alpha()
    self.set_gamma()