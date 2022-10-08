from abc import ABC, abstractmethod

'''
This is the abstract method that defines the velocity field.
'''

class velocity_field(ABC):
    
    def __init__(self):
        # Define if your velocity field is defined over a bounded domain.
        # Analytical velocity fields do not usually have limits.
        self.limits = False
        # Define limits
        self.x_left  = None
        self.x_right = None
        self.y_down  = None
        self.y_up    = None
        pass
    
    @abstractmethod
    def get_velocity(self, x, y, t):
        # Define velocity of the flow at a specific position and time
        u = 0 * x
        v = 0 * y
        return u, v
    
    @abstractmethod
    def get_gradient(self, x, y, t):
        # Define spatial derivatives of the velocity field at a specific
        # position and time
        ux, uy = 0 * x, 0 * y
        vx, vy = 0 * x, 0 * y
        return ux, uy, vx, vy

    @abstractmethod
    def get_dudt(self, x, y, t):
        # Define time derivative of the velocity field at a specific
        # position and time
        ut = 0 * x
        vt = 0 * y
        return ut, vt