#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:12:39 2020

@author: cfg4065

Definition of a static velocity field.
This is used for the calculation of the relaxing particle's velocity and
trajectory.
"""

#import numpy as np
from a03_FIELD0_00000 import velocity_field

class velocity_field_Relaxing(velocity_field):

  def __init__(self):
    # We know the parameters of the field in the whole R^2, therefore
    # no limits
    self.limits = False
    
  def get_velocity(self, x, y, t):
    # The fluid does not move, therefore its velocity is zero
    u = 0.0
    v = 0.0
    return u, v
  
  def get_gradient(self, x, y, t):
    # The velocity of the fluid particles does not vary in space, therefore
    # all are zero.
    ux = 0.0
    uy = 0.0
    vx = 0.0
    vy = 0.0
    return ux, uy, vx, vy

  def get_dudt(self, x, y, t):
    # No change in velocity with respect to time, therefore all equal to zero
    ut = 0.0
    vt = 0.0
    return ut, vt