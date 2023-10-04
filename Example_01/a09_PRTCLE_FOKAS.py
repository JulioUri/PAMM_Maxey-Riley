from a00_PMTERS_CONST import mr_parameter
from scipy.optimize import newton
import numpy.polynomial.chebyshev as cheb
import numpy as np
from os.path import exists
from progressbar import progressbar

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
    
    def __init__(self, tag, x, v, velocity_field, Nx, t0, dt, time_nodes,
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
        
        self.Nx       = Nx              # Number of spatial nodes
        
        # Class with all the parameter definition: alpha, gamma, R, St...
        self.p        = mr_parameter(particle_density, fluid_density,
                                    particle_radius, kinematic_viscosity,
                                    time_scale)
        
        self.time_nodes = time_nodes    # Number of time nodes
        
        self.pos_vec = np.copy(x)       # Create vector where position solution is saved
        
        # Check particle initial conditions are within field limits, applicable to experimental fields
        if self.vel.limits == True:
            if (x[0] > self.vel.x_right or x[0] < self.vel.x_left or x[1] > self.vel.y_up or x[1] < self.vel.y_down):
                raise Exception("Particle's initial position is outside the spatial domain") 
        
        # Create spatial grid
        self.xnodes_def()
        # Create time grid
        self.tnodes_def()
        # Precompute (or upload) matrices used for the integration of the Cheb poly
        self.ChebMat()


    # Define Spatial grid in the pseudospace 
    def xnodes_def(self):
        # Create initial condition q(x,t=t_0) for both x- and y-components
        u0, v0        = self.vel.get_velocity(self.x, self.y, self.t0)
        self.qx_tj    = np.zeros([1, self.Nx])[0]
        self.qx_tj[0] = self.v[0] - u0
        self.qy_tj    = np.zeros([1, self.Nx])[0]
        self.qy_tj[0] = self.v[1] - v0
        
        # Create array for solution q(0, t) for both x- and y-components
        self.q0       = np.array([self.qx_tj[0], self.qy_tj[0]])
        self.qx_x0    = np.zeros([1, self.time_nodes])[0]
        self.qx_x0[0] = self.q0[0]
        self.qy_x0    = np.zeros([1, self.time_nodes])[0]
        self.qy_x0[0] = self.q0[1]

        self.q_vec    = np.copy(self.q0)
        
        # Create Chebyschev nodes and map them to [0, inf)
        index_v       = np.arange(0, self.Nx)
        self.k_hat_v  = - np.cos(index_v * np.pi / self.Nx)
        self.k_v      = (1.0 + self.k_hat_v) / (1.0 - self.k_hat_v)        
        
          
    # Define chebyshev nodes in time subdomain [t_i, ti + dt], i.e. s_1, s_2, ...
    def tnodes_def(self):
        
        self.time_vec = np.array([])
        for jj in range(0, self.time_nodes):
            self.time_vec = np.append(self.time_vec,
                                      self.time + 0.5*self.dt * \
                        (1.0 - np.cos(jj * np.pi / (self.time_nodes-1.0))) )

    
    # Calculation of L(m) function, used to obtain matrix M and then F.
    def Lm(self, m):
        
        # We use the exp(log()) of the functions that make L(m), because
        # this avoids overflow errors.
        fun_exp  = lambda k: -m * k**2.0
        fun_frac = lambda k: np.log( self.p.gamma * k**2.0 / ((k * self.p.gamma)**2.0 + (k**2.0 - self.p.alpha)**2.0))
        
        fun      = lambda k: np.exp( fun_exp(k) + fun_frac(k) )
        
        fun_v    = np.array([])
        for kk in range(0, len(self.k_v)):
            if self.k_v[kk] == 0.0:
                fun_v = np.append(fun_v, 0.0)
            else:
                fun_v = np.append(fun_v, fun(self.k_v[kk]) * (2.0 / (1.0 - self.k_hat_v[kk])**2.0))
            
        coeff     = cheb.chebfit(self.k_hat_v, fun_v, len(self.k_v) - 1)
        coeff_int = cheb.chebint(coeff, m=1, lbnd=-1.)
        
        result    = cheb.chebval(1.0, coeff_int)
        return result


    # Calculation of Matrix M from paper
    def ChebMat(self):
        
        length_t   = len(self.time_vec)       
        
        self.t_vec = (2. * (self.time_vec - self.t0) - self.dt) / self.dt
        
        name_file1 = 'a00_MAT01_VALUES.npy'
        name_file2 = 'a00_MAT02_VALUES.npy'
        
        # We only consider two scenarios: (1) both files exist and (2) either
        # of the files does not exist or either matrix does not have the
        # desired dimensions. We do this to avoid complications in the code
        # that do not provide any special feature.
        if exists(name_file1) == True and exists(name_file2) == True:
            with open(name_file1, 'rb') as file1:
                mat1 = np.load(file1)
            with open(name_file2, 'rb') as file2:
                mat2 = np.load(file2)
        
        if exists(name_file1) == False or \
                exists(name_file2) == False or \
                     mat1.shape[0] != length_t or \
                          mat2.shape[0] != length_t:

            mat1    = np.zeros([length_t, length_t])
            mat2    = np.zeros([length_t, length_t])
        
            for ii in progressbar(range(1, length_t)):
                
                Lm_vec   = np.array([])
                for elem in range(0, ii+1):
                    m = self.time_vec[ii] - self.time_vec[elem]
                    if m == 0.0:
                        Lm_vec = np.append(Lm_vec, np.pi/2.)
                    else:
                        Lm_vec = np.append(Lm_vec, self.Lm(m))
                
                Lm_coeff = cheb.chebfit(self.t_vec[:(ii+1)], Lm_vec, len(Lm_vec) - 1)
                
                for nn in range(0, length_t):
                    # Create vector of coefficients to define Chebyshev Polynomial
                    coeff        = np.zeros([1, length_t])[0]
                
                    # Fill in the element of the vector corresponding to the matrix entry
                    coeff[nn]    = 1.0
                
                    # integrate
                    mul_coeff    = cheb.chebmul(coeff, Lm_coeff)
                    
                    coeff_int1   = cheb.chebint(mul_coeff, m=1, lbnd=-1.)
                    aux1         = cheb.chebval(self.t_vec[ii], coeff_int1)
                    
                    mat1[ii][nn] = aux1
                    
                    coeff_int2   = cheb.chebint(coeff, m=1, lbnd=-1.)
                    aux2         = cheb.chebval(self.t_vec[ii], coeff_int2)
                                  
                    mat2[ii][nn] = aux2
        
            with open(name_file1, 'wb') as file1:
                np.save(file1, mat1)
                
            with open(name_file2, 'wb') as file2:
                np.save(file2, mat2)
            
        self.M_nn   = mat1
        self.T_nn   = mat2


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
    def F(self, q_guess, x_guess):
        
        # Change solution guess according to function f
        f_vec      = np.array([])
        g_vec      = np.array([])
        
        for tt in range(0,len(self.time_vec)):
            q_vec  = np.array([q_guess[tt], q_guess[tt + int(len(q_guess)/2.0)]])
            x_vec  = np.array([x_guess[tt], x_guess[tt + int(len(x_guess)/2.0)]])
            f, g   = self.calculate_f(q_vec, x_vec, self.time_vec[tt])
            
            f_vec  = np.append(f_vec, f)
            g_vec  = np.append(g_vec, g)
        
        self.f_vec = f_vec
        self.g_vec = g_vec
        
        # Approximate by Chebyshev coeff
        coeff_x    = cheb.chebfit(self.t_vec, f_vec * self.dt/2., len(self.time_vec)-1)
        coeff_y    = cheb.chebfit(self.t_vec, g_vec * self.dt/2., len(self.time_vec)-1)
        
        # Obtain F as the dot product of the coeff times the matrix entries
        result_x   = np.array([])
        result_y   = np.array([])
        for jj in range(0,len(self.time_vec)):
            result_x  = np.append(result_x, np.dot(coeff_x, self.M_nn[jj]))
            result_y  = np.append(result_y, np.dot(coeff_y, self.M_nn[jj]))
        
        result_v = np.append(result_x, result_y)
        
        return result_v


    def G_fun(self):
        
        resultx_v   = np.array([])
        resulty_v   = np.array([])
        for k in self.k_v:
            fun_den    = (self.p.alpha - k**2.0)**2.0 + (k * self.p.gamma)**2.0
            fun_x      = - k * self.p.gamma * self.qx_x0[0] / fun_den
            fun_y      = - k * self.p.gamma * self.qy_x0[0] / fun_den
            
            resultx_v  = np.append(resultx_v, fun_x)
            resulty_v  = np.append(resulty_v, fun_y)
        
        result_v    = np.array([resultx_v, resulty_v])
        
        self.G_v_imag  = result_v

        return result_v


    def G_update(self):
        
        length_t = self.time_nodes
        
        I1x_v     = np.array([])
        I1y_v     = np.array([])
        
        for k in self.k_v:
            
            # Approximate by Chebyshev coeff
            exponent = -(self.time_old_v[-1] - self.time_old_v) * k**2.0
            exponent[exponent < -70.] = -70.0
            
            coeff_x    = cheb.chebfit(self.time_old_v,
                                      np.exp(exponent) * self.f_vec, length_t-1)
            coeff_y    = cheb.chebfit(self.time_old_v,
                                      np.exp(exponent) * self.g_vec, length_t-1)
            coeffx_int = cheb.chebint(coeff_x, m=1, lbnd=self.time_old_v[0])
            coeffy_int = cheb.chebint(coeff_y, m=1, lbnd=self.time_old_v[0])
            f_tld      = cheb.chebval(self.time_old_v[-1], coeffx_int)
            g_tld      = cheb.chebval(self.time_old_v[-1], coeffy_int)
            
            f_tld_frac = k * self.p.gamma * f_tld /\
                        ((self.p.alpha - k**2.0)**2.0 + (k * self.p.gamma)**2.0)
            g_tld_frac = k * self.p.gamma * g_tld /\
                        ((self.p.alpha - k**2.0)**2.0 + (k * self.p.gamma)**2.0)
            
            I1x_v     = np.append(I1x_v, f_tld_frac)
            I1y_v     = np.append(I1y_v, g_tld_frac)
        
        exponentx      = -self.k_v**2.0 * self.dt
        exponentx[exponentx < -70.] = -70.0
        resultx_v      = np.exp(exponentx) * self.G_v_imag[0] - I1x_v
        
        exponenty      = -self.k_v**2.0 * self.dt
        exponenty[exponenty < -70.] = -70.0
        resulty_v      = np.exp(exponenty) * self.G_v_imag[1] - I1y_v
        
        result_v    = np.array([resultx_v, resulty_v])
        
        self.G_v_imag = result_v
        
        return result_v
    
    def H(self):
        
        if self.time_vec[0] - self.t0 == 0.0:
            self.G_fun()
        else:
            self.G_update()
        
        resultx_v     = np.array([])
        resulty_v     = np.array([])
        
        for jj in range(1, len(self.time_vec)):        
            
            exponent   = - (self.time_vec[jj] - self.time_vec[0]) * self.k_v**2.0
            exponent[exponent < -70.] = -70.0
            
            chng_var   = 2.0 / (1 - self.k_hat_v)**2.0
            
            Hx_intgrnd = self.k_v * np.exp(exponent) * self.G_v_imag[0] * chng_var
            Hy_intgrnd = self.k_v * np.exp(exponent) * self.G_v_imag[1] * chng_var
            
            coeff_x    = cheb.chebfit(self.k_hat_v, Hx_intgrnd, len(self.k_hat_v)-1)
            coeff_y    = cheb.chebfit(self.k_hat_v, Hy_intgrnd, len(self.k_hat_v)-1)
        
            coeffx_int = cheb.chebint(coeff_x, m=1, lbnd=-1.)
            coeffy_int = cheb.chebint(coeff_y, m=1, lbnd=-1.)
        
            resultx_v  = np.append(resultx_v, cheb.chebval(1.0, coeffx_int))
            resulty_v  = np.append(resulty_v, cheb.chebval(1.0, coeffy_int))
        
        cte  = -np.pi/2.
        if self.time_vec[0] - self.t0 == 0.0:
            resultx_v  = np.append(cte * self.qx_x0[0], resultx_v)
            resulty_v  = np.append(cte * self.qy_x0[0], resulty_v)
        else:
            resultx_v  = np.append(cte * self.qx_x0[-1], resultx_v)
            resulty_v  = np.append(cte * self.qy_x0[-1], resulty_v)
        
        result_v   = np.append(resultx_v, resulty_v)
        
        self.H_v   = result_v
        
        return resultx_v, resulty_v
    
    def eta(self, q_v, p_v, x_v, y_v):
        U_v, V_v   = self.vel.get_velocity(x_v, y_v, self.time_vec)
        
        x_coeff    = cheb.chebfit(self.t_vec, (U_v + q_v) * self.dt/2., len(self.t_vec)-1)
        y_coeff    = cheb.chebfit(self.t_vec, (V_v + p_v) * self.dt/2., len(self.t_vec)-1)
        
        resultx_v  = np.array([])
        resulty_v  = np.array([])
        for jj in range(0, len(self.time_vec)):
            resultx_v = np.append(resultx_v, np.dot(x_coeff, self.T_nn[jj]))
            resulty_v = np.append(resulty_v, np.dot(y_coeff, self.T_nn[jj]))
        
        result_v   = np.append(resultx_v, resulty_v)
        
        return result_v
    
    # Define function J to obtain rules on
    def J(self, guess):
        
        len_v     = len(self.time_vec)
        
        q_v       = guess[:len_v]
        p_v       = guess[len_v:len_v*2]
        x_v       = guess[len_v*2:len_v*3]
        y_v       = guess[len_v*3:]
        
        relvel_v  = np.append(q_v, p_v)
        pos_v     = np.append(x_v, y_v)
        
        veln1     = -self.F(relvel_v, pos_v) - self.H_v
        posn1     = self.x0_v + self.eta(q_v, p_v, x_v, y_v)
        
        vel_zero  = (np.pi/2.) * relvel_v - veln1
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
            
        self.x0_v     = np.append(x, y)
        
      
        # relative velocity's guess for the nonlinear solver.
        q_guess, p_guess = self.H()
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
        
        result     = newton(self.J,
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