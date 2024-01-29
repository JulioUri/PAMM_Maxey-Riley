from a00_PMTERS_CONST import mr_parameter
from scipy.integrate import quad
from scipy.optimize import newton, root, minimize, fsolve
import numpy.polynomial.chebyshev as cheb
import matplotlib.pyplot as plt
import numpy as np
import time
from os.path import exists
from progressbar import progressbar
from scipy.sparse.linalg import spsolve

'''
The class defined below calculates the trajectory and velocity of particles
whose dynamics is governed by the FULL MAXEY-RILEY by using the Fokas method
given in the paper Prasat et al. (2019). 


Function "__init__" defines all the variables associated to the particle.

Function "update" initiates the update in the particle's velocity and trajectory
for a given time step.
'''


'''
###############################################################################
#########################FULL MAXEY RILEY (Prasath)############################
###############################################################################
'''
class maxey_riley_fokas(object):

    def __init__(self, tag, x, v, velocity_field, Nk, t0, dt, time_nodes,
               particle_density=1, fluid_density=1, particle_radius=1,
               kinematic_viscosity=1, time_scale=1):
      
        self.tag      = tag             # Particle name/tag
        
        self.t0       = t0              # Time at which particle starts moving
        self.time     = t0              # Initial time of the calculation/update interval
        self.dt       = dt              # Time step, should be cte
        
        self.x        = x[0]            # Particle's initial position (x-component)
        self.y        = x[1]            # Particle's initial position (y-component)
        
        self.v        = v               # Particle's initial velocity
        
        self.vel      = velocity_field  # Velocity field class with methods to calculate velocity and derivatives
        
        self.Nk       = Nk              # Nodes in pseudospace
        
        # Class with all the parameter definition: alpha, gamma, R, St...
        self.p        = mr_parameter(particle_density, fluid_density,
                                    particle_radius, kinematic_viscosity,
                                    time_scale)
        
        '''Maybe self.time_nodes can be eliminated?'''
        self.time_nodes = time_nodes    # Number of time nodes
        
        self.pos_vec = np.copy(x)       # Create vector where position solution is saved
        
        # Check particle initial conditions are within field limits, applicable to experimental fields
        if self.vel.limits == True:
            if (x[0] > self.vel.x_right or x[0] < self.vel.x_left or x[1] > self.vel.y_up or x[1] < self.vel.y_down):
                raise Exception("Particle's initial position is outside the spatial domain") 
        
        # Create solution vectors
        self.solvec_def()
        # Create spatial grid
        self.knodes_def()
        # Create time grid
        self.sigmav_def()
        self.tnodes_def()
        # Precompute (or upload) matrices used for the integration of the Cheb poly
        self.ChebMat()



    def solvec_def(self):
        # Create initial condition q(x, t=t_0) for both x- and y-components
        u0, v0        = self.vel.get_velocity(self.x, self.y, self.t0)
        self.qx_tj    = np.zeros([1, self.Nk])[0]
        self.qx_tj[0] = self.v[0] - u0
        self.qy_tj    = np.zeros([1, self.Nk])[0]
        self.qy_tj[0] = self.v[1] - v0
        
        # Create array for solution q(0, t) for both x- and y-components
        self.q0       = np.array([self.qx_tj[0], self.qy_tj[0]])
        self.qx_x0    = np.zeros([1, self.time_nodes])[0]
        self.qx_x0[0] = self.q0[0]
        self.qy_x0    = np.zeros([1, self.time_nodes])[0]
        self.qy_x0[0] = self.q0[1]

        self.q_vec    = np.copy(self.q0)



    # Define Spatial grid in the pseudospace 
    def knodes_def(self):        
        # Create Chebyschev nodes and map them to [0, inf)
        index_v       = np.arange(0, self.Nk)
        self.k_hat_v  = - np.cos(index_v * np.pi / self.Nk)
        self.k_v      = (1.0 + self.k_hat_v) / (1.0 - self.k_hat_v)
        
    
        
    # Define chebyshev nodes in time subdomain [t_i, ti + dt], i.e. s_1, s_2, ...
    def sigmav_def(self):
        sigma_v = np.array([])
        for jj in range(0, self.time_nodes):
            sigma_v = np.append(sigma_v,
                                -np.cos(jj * np.pi / (self.time_nodes-1.0)))
        self.sigma_v = sigma_v
             
        
        
    def tnodes_def(self):
        self.time_vec = np.array([])
        self.time_vec = self.time + 0.5*self.dt * (1.0 + self.sigma_v)

    
    
    # Calculation of L(m) function, used to obtain matrix M and then F.
    def L(self, m_v):
        result_v = np.array([])
        for m in m_v:
            # We use the exp(log()) of the functions that make L(m), because
            # this avoids overflow errors.
            fun_exp  = lambda k: -m * k**2.0
            fun_frac = lambda k: np.log( self.p.gamma * k**2.0 / ((k * self.p.gamma)**2.0 + (self.p.alpha - k**2.0)**2.0) )
            
            fun      = lambda k: np.exp( fun_exp(k) + fun_frac(k) )
            
            fun_v    = np.array([])
            for kk in range(0, len(self.k_v)):
                if self.k_v[kk] == 0.0:
                    fun_v = np.append(fun_v, 0.0)
                else:
                    if fun_exp(self.k_v[kk]) + fun_frac(self.k_v[kk]) < -500:
                        fun_v    = np.append(fun_v, 0.0)
                    else:
                        fun_v = np.append(fun_v, fun(self.k_v[kk]) * (2.0 / (1.0 - self.k_hat_v[kk])**2.0))
                        
            coeff     = cheb.chebfit(self.k_hat_v, fun_v, len(self.k_v) - 1)
            coeff_int = cheb.chebint(coeff, m=1, lbnd=-1.)
        
            result    = cheb.chebval(1.0, coeff_int)
            result_v  = np.append(result_v, result)
            
        return result_v



    def h(self, kl, sigma):
        rational_fnc = kl * self.p.gamma / \
                       ((self.p.alpha - kl**2.)**2. + (kl * self.p.gamma)**2.)
        exponent_fnc = -kl**2. * (1. - sigma) * self.dt * 0.5
        exponent_fnc[exponent_fnc < -70] = -70
        return rational_fnc * np.exp(exponent_fnc)
    
    
    
    def G(self, k_tilde, t_tilde):
        k = (1. + k_tilde) / (1. - k_tilde)
        exponent = -k**2. * t_tilde
        exponent[exponent < -70.] = -70.
        return k * np.exp(exponent) * 2. / (1. - k_tilde)**2.



    # Calculation of Matrix M from paper
    def ChebMat(self):
        
        length_t   = len(self.time_vec)
        
        name_file1 = 'a00_MAT-F_VALUES.npy'
        name_file2 = 'a00_MAT-H_VALUES.npy'
        name_file3 = 'a00_MAT-Y_VALUES.npy'
        name_file4 = 'a00_MAT-I_VALUES.npy'
        name_file5 = 'a00_TSTEP_VALUE.npy'
        
        # We only consider two scenarios: (1) all files exist and (2) either
        # of the files does not exist or either matrix does not have the
        # desired dimensions. We do this to avoid complications in the code
        # that do not provide any special feature.
        if exists(name_file1) == True and exists(name_file2) == True and \
            exists(name_file3) == True and exists(name_file4) == True and \
             exists(name_file5) == True:
            with open(name_file1, 'rb') as file1:
                MatF = np.load(file1)
            with open(name_file2, 'rb') as file2:
                MatH = np.load(file2)
            with open(name_file3, 'rb') as file3:
                MatY = np.load(file3)
            with open(name_file4, 'rb') as file4:
                MatI = np.load(file4)
            with open(name_file5, 'rb') as file5:
                dt = np.load(file5)
        
        if exists(name_file1) == False or exists(name_file2) == False or \
              exists(name_file3) == False or exists(name_file4) == False or \
                     MatF.shape != (length_t, length_t) or \
                       MatH.shape != (length_t, self.Nk) or \
                         MatY.shape != (length_t, length_t) or \
                           MatI.shape != (self.Nk, length_t) or \
                             dt - self.dt != 0.0:
            
            print("Creating matrices M_F, M_H, M_Y and M_I for Prasath et al.'s method.")                     
            
            MatF    = np.zeros([length_t, length_t])
            MatH    = np.zeros([length_t, self.Nk])
            MatY    = np.zeros([length_t, length_t])
            MatI    = np.zeros([self.Nk,  length_t])
        
            for tt in range(0, length_t):
                if tt != 0:
                    coeffF2      = cheb.chebfit(self.sigma_v,
                                                self.L(self.time_vec - self.time_vec[0]),
                                                self.time_nodes-1)
                    for nn in range(0, length_t):
                        coeffF1      = np.zeros([1, nn+1])[0]
                        coeffF1[nn]  = 1.0
                        coeffF       = cheb.chebmul(coeffF1, coeffF2)
                        coeffF_int   = cheb.chebint(coeffF, m=1, lbnd=-1.)
                        MatF[tt][nn] = cheb.chebval(self.sigma_v[tt], coeffF_int)
                        
                        
                        coeffY       = np.zeros([1, nn+1])[0]
                        coeffY[nn]   = 1.0
                        coeffY_int   = cheb.chebint(coeffY, m=1, lbnd=-1.)
                        MatY[tt][nn] = cheb.chebval(self.sigma_v[tt], coeffY_int)

                for kk in range(0, self.Nk):
                    # Create vector of coefficients to define Chebyshev Polynomial
                    coeffI1      = np.zeros([1, tt+1])[0]
                    coeffI1[tt]  = 1.0
                    
                    coeffI2      = cheb.chebfit(self.sigma_v,
                                                self.h(self.k_v[kk], self.sigma_v),
                                                self.time_nodes-1)
                    
                    coeffI       = cheb.chebmul(coeffI1, coeffI2)
                    coeffI_int   = cheb.chebint(coeffI, m=1, lbnd=-1.)
                    MatI[kk][tt] = cheb.chebval(1.0, coeffI_int)
                    
                    
                    if tt != 0:
                        coeffH1      = np.zeros([1, kk+1])[0]
                        coeffH1[kk]  = 1.0
                        coeffH2      = cheb.chebfit(self.k_hat_v,
                                                    self.G(self.k_hat_v,
                                                           self.time_vec[tt] - self.time_vec[0]),
                                                    self.Nk - 1)
                        coeffH       = cheb.chebmul(coeffH1, coeffH2)
                        coeffH_int   = cheb.chebint(coeffH, m=1, lbnd=-1.)
                        MatH[tt][kk] = cheb.chebval(1.0, coeffH_int)
                    
            with open(name_file1, 'wb') as file1:
                np.save(file1, MatF)
                
            with open(name_file2, 'wb') as file2:
                np.save(file2, MatH)
                
            with open(name_file3, 'wb') as file3:
                np.save(file3, MatY)
            
            with open(name_file4, 'wb') as file4:
                np.save(file4, MatI)
                
            with open(name_file5, 'wb') as file5:
                np.save(file5, self.dt)
   
        self.MatF   = MatF
        self.MatH   = MatH
        self.MatY   = MatY
        self.MatI   = MatI


    # Define f(q(0,t))    # This will differ for different boundary problems
    def calculate_f(self, qv, x, t):
        
        q, p           = qv[0], qv[1]
      
        coeff          = (1.0 / self.p.R) - 1.0
      
        u, v           = self.vel.get_velocity(x[0], x[1], t)
        ux, uy, vx, vy = self.vel.get_gradient(x[0], x[1], t)
      
        ut, vt         = self.vel.get_dudt(x[0], x[1], t)
      
        f              = coeff * ut + (coeff * u - q) * ux + \
                                    (coeff * v - p) * uy
        g              = coeff * vt + (coeff * u - q) * vx + \
                                    (coeff * v - p) * vy
      
        return f, g



    # Calculation of F(t_j)
    def F_def(self, q_guess, x_guess):
        # Change solution guess according to function f
        f_vec      = np.array([])
        g_vec      = np.array([])
        for tt in range(0, len(self.time_vec)):
            q_vec  = np.array([q_guess[tt], q_guess[tt + self.time_nodes]])
            x_vec  = np.array([x_guess[tt], x_guess[tt + self.time_nodes]])
            f, g   = self.calculate_f(q_vec, x_vec, self.time_vec[tt])
            
            f_vec  = np.append(f_vec, f)
            g_vec  = np.append(g_vec, g)
        
        self.f_vec = f_vec
        self.g_vec = g_vec
        
        # Obtain Chebyshev coefficients of approximation polynomial
        coeff_x    = cheb.chebfit(self.sigma_v, f_vec, self.time_nodes-1)
        coeff_y    = cheb.chebfit(self.sigma_v, g_vec, self.time_nodes-1)
        
        # Obtain F as the dot product of the coeff times the matrix entries
        result_x   = np.array([])
        result_y   = np.array([])
        for jj in range(0,len(self.time_vec)):
            result_x  = np.append(result_x, self.dt/2. * np.dot(coeff_x, self.MatF[jj]))
            result_y  = np.append(result_y, self.dt/2. * np.dot(coeff_y, self.MatF[jj]))
        
        result_v = np.append(result_x, result_y)
        
        return result_v



    def ImCalH_fun(self):
        
        resultx_v   = np.array([])
        resulty_v   = np.array([])
        for k in self.k_v:
            fun_den    = (self.p.alpha - k**2.0)**2.0 + (k * self.p.gamma)**2.0
            fun_x      = - k * self.p.gamma * self.qx_x0[0] / fun_den
            fun_y      = - k * self.p.gamma * self.qy_x0[0] / fun_den
            
            resultx_v  = np.append(resultx_v, fun_x)
            resulty_v  = np.append(resulty_v, fun_y)
        
        result_v    = np.array([resultx_v, resulty_v])
        
        self.CalH_v_imag = result_v

        return result_v



    def ImCalH_update(self):
        
        f_coeff   = cheb.chebfit(self.sigma_v, self.f_vec, self.time_nodes-1)
        g_coeff   = cheb.chebfit(self.sigma_v, self.g_vec, self.time_nodes-1)
        I1x_v     = np.array([])
        I1y_v     = np.array([])
        for kk in range(0, self.Nk):
            I1x_v     = np.append(I1x_v, np.dot(f_coeff, self.MatI[kk]))
            I1y_v     = np.append(I1y_v, np.dot(g_coeff, self.MatI[kk]))
        
        exponent       = -self.k_v**2.0 * self.dt
        exponent[exponent < -70.] = -70.0
        resultx_v      = np.exp(exponent) * self.CalH_v_imag[0] - \
                         self.dt/2. * I1x_v
        resulty_v      = np.exp(exponent) * self.CalH_v_imag[1] - \
                         self.dt/2. * I1y_v
        
        result_v    = np.array([resultx_v, resulty_v])
        
        self.CalH_v_imag = result_v
        
        return result_v

    
    
    def H_def(self):
        
        if self.time_vec[0] - self.t0 == 0.0:
            self.ImCalH_fun()
        else:
            self.ImCalH_update()
        
        Hxcoeff = cheb.chebfit(self.k_hat_v, self.CalH_v_imag[0], self.Nk - 1)
        Hycoeff = cheb.chebfit(self.k_hat_v, self.CalH_v_imag[1], self.Nk - 1)
        
        resultx_v     = np.array([])
        resulty_v     = np.array([])
        for tt in range(1, self.time_nodes):
            resultx_v = np.append(resultx_v, np.dot(Hxcoeff, self.MatH[tt]))
            resulty_v = np.append(resulty_v, np.dot(Hycoeff, self.MatH[tt]))
        
        cte = -np.pi / 2.0
        if self.time_vec[0] - self.t0 == 0.0:
            resultx_v  = np.append(cte * self.qx_x0[0], resultx_v)
            resulty_v  = np.append(cte * self.qy_x0[0], resulty_v)
        else:
            resultx_v  = np.append(cte * self.qx_x0[-1], resultx_v)
            resulty_v  = np.append(cte * self.qy_x0[-1], resulty_v)
        
        
        result_v   = np.append(resultx_v, resulty_v)
        
        self.H_v   = result_v
        
        return resultx_v, resulty_v
    
    
    
    def eta_def(self, q_v, p_v, x_v, y_v):
        U_v  = np.array([])
        V_v  = np.array([])
        for nn in range(0, len(self.time_vec)):
            u, v   = self.vel.get_velocity(x_v[nn], y_v[nn], self.time_vec[nn])
            U_v    = np.append(U_v, u)
            V_v    = np.append(V_v, v)
        
        x_coeff    = cheb.chebfit(self.sigma_v, (U_v + q_v), self.time_nodes-1)
        y_coeff    = cheb.chebfit(self.sigma_v, (V_v + p_v), self.time_nodes-1)
        
        resultx_v  = np.array([])
        resulty_v  = np.array([])
        for jj in range(0, len(self.time_vec)):
            resultx_v = np.append(resultx_v, self.dt/2. * np.dot(x_coeff, self.MatY[jj]))
            resulty_v = np.append(resulty_v, self.dt/2. * np.dot(y_coeff, self.MatY[jj]))
        
        result_v   = np.append(resultx_v, resulty_v)
        
        return result_v



    # Define function J to obtain rules on
    def J_def(self, guess):
        
        len_v     = len(self.time_vec)
        
        q_v       = guess[:len_v]
        p_v       = guess[len_v:len_v*2]
        x_v       = guess[len_v*2:len_v*3]
        y_v       = guess[len_v*3:]
        
        relvel_v  = np.append(q_v, p_v)
        pos_v     = np.append(x_v, y_v)
        
        veln1     = self.H_v  - self.F_def(relvel_v, pos_v)
        posn1     = self.y0_v + self.eta_def(q_v, p_v, x_v, y_v)
        
        vel_zero  = relvel_v + (2./np.pi) * veln1
        pos_zero  = pos_v - posn1
        
        return np.append(vel_zero, pos_zero)
    
    
    
    def ForwardEuler(self, q_guess, p_guess):
        if self.time - self.t0 == 0:
            x_guess      = np.array([self.x])
            y_guess      = np.array([self.y])
        else:
            x_guess      = np.array([self.x[-1]])
            y_guess      = np.array([self.y[-1]])
            
        for tt in range(0, len(self.time_vec)-1):
            dt        = self.time_vec[tt+1] - self.time_vec[tt]
            
            q, p      = q_guess[tt], p_guess[tt]
            
            u, v      = self.vel.get_velocity(x_guess[-1], y_guess[-1],
                                              self.time_vec[tt])
            x_guess   = np.append(x_guess, x_guess[-1] + dt * (q + u))
            y_guess   = np.append(y_guess, y_guess[-1] + dt * (p + v))
            
        return x_guess, y_guess
      
    
    # Function that runs nonlinear solver and calculates solution at boundary.
    def update(self):
        
        # Set y(t_0) vector for the nonlinear solver.
        if self.time - self.t0 == 0.0:
            x = np.ones([1, self.time_nodes])[0] * self.x
            y = np.ones([1, self.time_nodes])[0] * self.y
        else:
            x = np.ones([1, self.time_nodes])[0] * self.x[-1]
            y = np.ones([1, self.time_nodes])[0] * self.y[-1]
            
        self.y0_v     = np.append(x, y)
        
      
        # relative velocity's guess for the nonlinear solver.
        q_guess, p_guess = self.H_def()
        q_guess *= -(2./np.pi)
        p_guess *= -(2./np.pi)
        
        # position vector's guess for the nonlinear solver, calculated with
        # forward's Euler.
        x_guess, y_guess = self.ForwardEuler(q_guess, p_guess)
        
        guess      = np.concatenate((q_guess, p_guess,
                                     x_guess, y_guess))
        
        ######################### NON-LINEAR SOLVER #########################
        #print("Initial guess: \n" + str(guess))
        iter_limit = 5000
        tolerance  = 1e-13
        
        result     = newton(self.J_def,
                            guess,
                            tol=tolerance,
                            maxiter=iter_limit,
                            full_output=True)
        
        # Check method converged before reaching maxiter.
        if np.any(np.invert(result[1])) == False: # Method converged!
            solution = result[0]
        else:
            raise Exception("Solver did not converge")
        
        self.qx_x0  = solution[:len(self.time_vec)]
        self.qy_x0  = solution[len(self.time_vec):2*len(self.time_vec)]
        self.x      = solution[2*len(self.time_vec):3*len(self.time_vec)]
        self.y      = solution[3*len(self.time_vec):]
        
        pos_vec       = np.array([self.x[1:], self.y[1:]])
        q_vec         = np.array([self.qx_x0[1:], self.qy_x0[1:]])
        self.pos_vec  = np.vstack([self.pos_vec, np.transpose(pos_vec)])
        self.q_vec    = np.vstack([self.q_vec, np.transpose(q_vec)])
        
        self.time_old_v = self.time_vec
        self.time    += self.dt
        self.tnodes_def()
  
        return solution