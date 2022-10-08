#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:12:39 2020

@author: cfg4065

Definition of the velocity field corresponding to the SINGLE POINT VORTEX
with GAMMA constant equal to 2*pi

This is used for the calculation of the particles' trajectories and
velocities of Example 5 in Prasath et al (2019), showed in figure 11.
"""

import numpy as np
from a03_FIELD0_00000 import velocity_field

class velocity_field_Vortex(velocity_field):

  def __init__(self):
    # Define GAMMA constant
    self.Gamma = 2 * np.pi
    # Analytical flow, therefore no limits
    self.limits = False
    
  def get_velocity(self, x, y, t):
    # Define velocity of field with formula 5.15 in Prasath et al. (2019) paper
    norm_sq = x**2 + y**2
    u = (self.Gamma/(2*np.pi)) * (-1/norm_sq) * y
    v = (self.Gamma/(2*np.pi)) * (1/norm_sq) * x
    return u, v
  
  def get_gradient(self, x, y, t):
    # Define spatial derivatives of field.
    norm_sq = x**2 + y**2
    const = (self.Gamma / (2*np.pi))
    ux = const * 2.0 * x * y / (norm_sq**2)
    uy = const * (y**2 - x**2) / (norm_sq**2)
    vx = const * (y**2 - x**2) / (norm_sq**2)
    vy = - const * 2.0 * x * y / (norm_sq**2)
    return ux, uy, vx, vy

  def get_dudt(self, x, y, t):
    # Define time derivatives of field.
    ut = 0.0
    vt = 0.0
    return ut, vt