#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:12:39 2020

@author: cfg4065

Definition of the velocity field corresponding to the COUETTE FLOW with
lambda constant equal to 1.

This is used for the calculation of the particles' trajectories and
velocities of Example 4 in Prasath et al (2019).
"""

import numpy as np
from a03_FIELD0_00000 import velocity_field

class velocity_field_Couette(velocity_field):

  def __init__(self):
    # Define constant
    self.lamda  = 1.0
    # Analytical flow, therefore no limits
    self.limits = False
    
  def get_velocity(self, x, y, t):
    # Define velocity of the field
    u = self.lamda * y
    v = 0.0
    return u, v
  
  def get_gradient(self, x, y, t):
    # Define spatial derivatives of the field
    ux = 0.0
    uy = self.lamda
    vx = 0.0
    vy = 0.0
    return ux, uy, vx, vy

  def get_dudt(self, x, y, t):
    # Define time derivatives of the field
    ut = 0.0
    vt = 0.0
    return ut, vt