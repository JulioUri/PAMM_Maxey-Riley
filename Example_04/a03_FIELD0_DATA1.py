import numpy as np
import scipy.io
from scipy.interpolate import RectBivariateSpline
from a03_FIELD0_00000 import velocity_field

'''
This is the class for the velocity field defined in the file: 2mm_Faraday_50Hz_40ms_1_6g.mat

The main feature of this class is that the velocity field is only known at discrete
values of time and space, therefore intermediate values have to be interpolated.

At the time for which we have values of the velocity, the velocity is obtained with a splines.

At the time for which we do not have values, the velocity is interpolated between the
velocity values of the closest splines.

Experimental data provided by A. Von Kameke.
'''

class velocity_field_Faraday1(velocity_field):

  def __init__(self, field_boundary=True):
      
    '''
    Obtain the Grid points as well as the velocity values at those points.
    '''
    self.filename = './00_2mm_Faraday_50Hz_40ms_1_6g.mat'
    mat = scipy.io.loadmat(self.filename)
    self.X = mat['X']
    self.Y = mat['Y']
    self.u = mat['vx']
    self.v = mat['vy']
    
    self.limits = False
    
    [ny_mesh, nx_mesh] = np.shape(self.X)
    [nt_data, ny_data, nx_data] = np.shape(self.u)
    
    '''
    delta_t defines the gap between the time nodes for which we have velocity data.
    For times in between, a linear interpolation is carried out.
    
    In this case, 40ms were used.
    '''
    self.delta_t = 40
    
    '''
    Check the amount of data coincide
    '''
    assert ny_mesh==ny_data, "Number of entries in y direction in mesh and data do not match"
    assert nx_mesh==nx_data, "Number of entries in x direction in mesh and data do not match"
    
    self.xaxis = self.X[0,:]
    self.yaxis = self.Y[:,0]
    self.x_left  = np.min(self.xaxis)
    self.x_right = np.max(self.xaxis)
    self.y_down  = np.min(self.yaxis)
    self.y_up    = np.max(self.yaxis)
    
    '''
    Generate the splines that interpolate the velocity between the grid points
    '''
    self.spline_u = []
    self.spline_v = []
    for nn in range(0,nt_data):
      uu = np.copy(self.u[nn,:,:])
      self.spline_u.append(RectBivariateSpline(self.xaxis, self.yaxis, uu.transpose(), bbox = [self.x_left, self.x_right, self.y_down, self.y_up], \
                               kx = 3, ky = 3, s = 0.0))
      vv = np.copy(self.v[nn,:,:])
      self.spline_v.append(RectBivariateSpline(self.xaxis, self.yaxis, vv.transpose(), bbox = [self.x_left, self.x_right, self.y_down, self.y_up], \
                               kx = 3, ky = 3, s = 0.0))
      
    '''
    Obtain the residuals, if they are different from zero, the data has been smoothed and
    the value of the spline at the grid points do not coincide with the values of the
    velocity field
    '''
    self.residuals = np.zeros((2,nt_data))
    for nn in range(0,nt_data):
      self.residuals[0,nn] = self.spline_u[nn].get_residual()
      self.residuals[1,nn] = self.spline_v[nn].get_residual()
    
    self.field_boundary = field_boundary
    
  '''
  We will provide the values  of velocity, gradient and the partial derivative at any time.
  Since the splines are only optained for discrete values of t, a linear interpolation is necessary.
  '''
  def get_velocity(self, x, y, t):
    
    assert type(x) == type(y), "Variables 'x' and 'y' are of a different type."
    
    if self.field_boundary == True:
        if (x > self.x_right or x < self.x_left or y > self.y_up or y < self.y_down):
            u = 0.0
            v = 0.0
        else:
            t_ms = t * 1e3
            assert t_ms <= 42240 and t_ms >= 0, "t_ms must be within the time domain, t_ms in [0,42240] (in miliseconds)"
            nt = int(np.floor(t_ms/self.delta_t))
            t_remain = t_ms % self.delta_t
            #print(nt,t_remain)
    
            if t_remain == 0.0:
                u = self.spline_u[nt].ev(xi = x, yi = y, dx = 0, dy = 0)
                v = self.spline_v[nt].ev(xi = x, yi = y, dx = 0, dy = 0)
            else:
                u_next = self.spline_u[nt+1].ev(xi = x, yi = y, dx = 0, dy = 0)
                u_prev = self.spline_u[nt].ev(xi = x, yi = y, dx = 0, dy = 0)
        
                u = u_prev + (u_next - u_prev) * (t_remain / self.delta_t)
    
                v_next = self.spline_v[nt+1].ev(xi = x, yi = y, dx = 0, dy = 0)
                v_prev = self.spline_v[nt].ev(xi = x, yi = y, dx = 0, dy = 0)
        
                v = v_prev + (v_next - v_prev) * (t_remain / self.delta_t)
    else:
        t_ms = t * 1e3
        assert t_ms <= 42240 and t_ms >= 0, "t_ms must be within the time domain, t_ms in [0,42240] (in miliseconds)"
        nt = int(np.floor(t_ms/self.delta_t))
        t_remain = t_ms % self.delta_t
        #print(nt,t_remain)
    
        if t_remain == 0.0:
            u = self.spline_u[nt].ev(xi = x, yi = y, dx = 0, dy = 0)
            v = self.spline_v[nt].ev(xi = x, yi = y, dx = 0, dy = 0)
        else:
            u_next = self.spline_u[nt+1].ev(xi = x, yi = y, dx = 0, dy = 0)
            u_prev = self.spline_u[nt].ev(xi = x, yi = y, dx = 0, dy = 0)
        
            u = u_prev + (u_next - u_prev) * (t_remain / self.delta_t)
    
            v_next = self.spline_v[nt+1].ev(xi = x, yi = y, dx = 0, dy = 0)
            v_prev = self.spline_v[nt].ev(xi = x, yi = y, dx = 0, dy = 0)
        
            v = v_prev + (v_next - v_prev) * (t_remain / self.delta_t)
        
    return u, v
  
  def get_gradient(self, x, y, t):
    if self.field_boundary == True:
        if (x > self.x_right or x < self.x_left or y > self.y_up or y < self.y_down):
            ux = 0.0
            uy = 0.0
            vx = 0.0
            vy = 0.0
        else:
            t_ms = t * 1e3
            assert t_ms <= 42240 and t_ms >= 0, "t_ms must be within the time domain, t_ms in [0,42240] (in miliseconds)"
            nt = int(np.floor(t_ms/self.delta_t))
            t_remain = t_ms % self.delta_t
        
            if t_remain == 0.0:
                ux = self.spline_u[nt].ev(xi = x, yi = y, dx = 1, dy = 0)
                uy = self.spline_u[nt].ev(xi = x, yi = y, dx = 0, dy = 1)
                vx = self.spline_v[nt].ev(xi = x, yi = y, dx = 1, dy = 0)
                vy = self.spline_v[nt].ev(xi = x, yi = y, dx = 0, dy = 1)
            else:
                ux_next = self.spline_u[nt+1].ev(xi = x, yi = y, dx = 1, dy = 0)
                ux_prev = self.spline_u[nt].ev(xi = x, yi = y, dx = 1, dy = 0)
        
                ux = ux_prev + (ux_next - ux_prev) * (t_remain / self.delta_t)
        
                uy_next = self.spline_u[nt+1].ev(xi = x, yi = y, dx = 0, dy = 1)
                uy_prev = self.spline_u[nt].ev(xi = x, yi = y, dx = 0, dy = 1)
        
                uy = uy_prev + (uy_next - uy_prev) * (t_remain / self.delta_t)
        
                vx_next = self.spline_v[nt+1].ev(xi = x, yi = y, dx = 1, dy = 0)
                vx_prev = self.spline_v[nt].ev(xi = x, yi = y, dx = 1, dy = 0)
        
                vx = vx_prev + (vx_next - vx_prev) * (t_remain / self.delta_t)
        
                vy_next = self.spline_v[nt+1].ev(xi = x, yi = y, dx = 0, dy = 1)
                vy_prev = self.spline_v[nt].ev(xi = x, yi = y, dx = 0, dy = 1)
        
                vy = vy_prev + (vy_next - vy_prev) * (t_remain / self.delta_t)  
    else:
        t_ms = t * 1e3
        assert t_ms <= 42240 and t_ms >= 0, "t_ms must be within the time domain, t_ms in [0,42240] (in miliseconds)"
        nt = int(np.floor(t_ms/self.delta_t))
        t_remain = t_ms % self.delta_t
        
        if t_remain == 0.0:
            ux = self.spline_u[nt].ev(xi = x, yi = y, dx = 1, dy = 0)
            uy = self.spline_u[nt].ev(xi = x, yi = y, dx = 0, dy = 1)
            vx = self.spline_v[nt].ev(xi = x, yi = y, dx = 1, dy = 0)
            vy = self.spline_v[nt].ev(xi = x, yi = y, dx = 0, dy = 1)
        else:
            ux_next = self.spline_u[nt+1].ev(xi = x, yi = y, dx = 1, dy = 0)
            ux_prev = self.spline_u[nt].ev(xi = x, yi = y, dx = 1, dy = 0)
        
            ux = ux_prev + (ux_next - ux_prev) * (t_remain / self.delta_t)
        
            uy_next = self.spline_u[nt+1].ev(xi = x, yi = y, dx = 0, dy = 1)
            uy_prev = self.spline_u[nt].ev(xi = x, yi = y, dx = 0, dy = 1)
        
            uy = uy_prev + (uy_next - uy_prev) * (t_remain / self.delta_t)
        
            vx_next = self.spline_v[nt+1].ev(xi = x, yi = y, dx = 1, dy = 0)
            vx_prev = self.spline_v[nt].ev(xi = x, yi = y, dx = 1, dy = 0)
        
            vx = vx_prev + (vx_next - vx_prev) * (t_remain / self.delta_t)
        
            vy_next = self.spline_v[nt+1].ev(xi = x, yi = y, dx = 0, dy = 1)
            vy_prev = self.spline_v[nt].ev(xi = x, yi = y, dx = 0, dy = 1)
        
            vy = vy_prev + (vy_next - vy_prev) * (t_remain / self.delta_t)
    
    return ux, uy, vx, vy

  def get_dudt(self, x, y, t):
      
    if self.field_boundary == True:
        if (x > self.x_right or x < self.x_left or y > self.y_up or y < self.y_down):
            ut = 0.0
            vt = 0.0
        else:
            t_ms = t * 1e3
            assert t_ms <= 42240 and t_ms >= 0, "t_ms must be within the time domain, t_ms in [0,42240] (in miliseconds)"
            nt = int(np.ceil(t_ms/self.delta_t))
    
            if t_ms == 0.0:
                ut = 0.0 # At initial time, du/dt = 0
                vt = 0.0
            else:
                u_next = self.spline_u[nt].ev(xi = x, yi = y, dx = 0, dy = 0)
                u_prev = self.spline_u[nt-1].ev(xi = x, yi = y, dx = 0, dy = 0)
                ut = ( u_next - u_prev ) / self.delta_t
        
                v_next = self.spline_v[nt].ev(xi = x, yi = y, dx = 0, dy = 0)
                v_prev = self.spline_v[nt-1].ev(xi = x, yi = y, dx = 0, dy = 0)
                vt = ( v_next - v_prev ) / self.delta_t
    else:
        t_ms = t * 1e3
        assert t_ms <= 42240 and t_ms >= 0, "t_ms must be within the time domain, t_ms in [0,42240] (in miliseconds)"
        nt = int(np.ceil(t_ms/self.delta_t))
    
        if t_ms == 0.0:
            ut = 0.0 # At initial time, du/dt = 0
            vt = 0.0
        else:
            u_next = self.spline_u[nt].ev(xi = x, yi = y, dx = 0, dy = 0)
            u_prev = self.spline_u[nt-1].ev(xi = x, yi = y, dx = 0, dy = 0)
            ut = ( u_next - u_prev ) / self.delta_t
        
            v_next = self.spline_v[nt].ev(xi = x, yi = y, dx = 0, dy = 0)
            v_prev = self.spline_v[nt-1].ev(xi = x, yi = y, dx = 0, dy = 0)
            vt = ( v_next - v_prev ) / self.delta_t
    return ut, vt
