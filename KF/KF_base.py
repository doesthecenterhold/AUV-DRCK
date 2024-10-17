import numpy as np


class KalmanFilter:

    def __init__(self, state_dim, control_dim, Hz):
        
        self.cur_mean = np.zeros(state_dim)
        self.cur_cov = np.eye(state_dim)

        self.Qt = np.eye(state_dim)

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.Hz = Hz
        self.dt = 1/self.Hz
    
    def init(self):
        pass

    def prediction(self):
        pass

    def correction(self):
        pass