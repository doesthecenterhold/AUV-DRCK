import numpy as np
from KF_base import KalmanFilter

# resource https://nitinjsanket.github.io/tutorials/attitudeest/kf

class OrientationKalmanFilter(KalmanFilter):
    """This filter estimates only the orientation on the basis
    of an accelerometer and a gyroscope. The state is 6DOF: roll, pitch, yaw, 3x gyro bias"""
    def __init__(self, params):
        super().__init__(params)

        self.At_default = np.eye(self.state_dim)
        self.At_default[0,3] = -self.dt
        self.At_default[1,4] = -self.dt
        self.At_default[2,5] = -self.dt

        self.Bt_default = np.zeros((self.state_dim, self.control_dim))
        self.Bt_default[0,0] = self.dt
        self.Bt_default[1,1] = self.dt
        self.Bt_default[2,2] = self.dt

        self.C = np.zeros((self.state_dim, self.state_dim))
        self.C[0,0] = 1
        self.C[1,1] = 1

    def get_cur_R(self):
        phi, theta, psi, b1, b2, b3 = self.cur_mean
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        ctheta = np.cos(theta)
        stehta = np.sin(theta)

        R = np.array([[ctheta, 0, -cphi*stehta],
                      [0, 1, sphi],
                      [stehta, 0, cphi*ctheta]])
        
        return R

    def init(self, mean, cov, Qt, dt = None):

        self.cur_mean = mean
        self.cur_cov = cov
        self.Qt = Qt


    def prediction(self, gyrot, dt = None):

        # In this filter, wt are the gyroscope measurements (angular velocities)

        if not dt is None:
            At = np.eye(self.state_dim)
            At[0,3] = -self.dt
            At[1,4] = -self.dt
            At[2,5] = -self.dt

            Bt = np.zeros((self.state_dim, self.control_dim))
            Bt[0,0] = self.dt
            Bt[1,1] = self.dt
            Bt[2,2] = self.dt
        else:
            At = self.At_default
            Bt = self.Bt_default


        R = self.get_cur_R() # It is calculated from the current estimate of the mean
        ut = R.T @ gyrot

        #  Finally, calculate the new state
        self.cur_mean = At @ self.cur_mean + Bt @ ut
        self.cur_cov = At @ self.cur_cov @ At.T + self.Qt

        return self.cur_mean, self.cur_cov
    
    def correction(self, acct):
        
        ax, ay, az = acct
        phi = np.atan2(ay, np.sqrt(ax**2 + az**2))
        theta = np.atan2(ax, np.sqrt(ay**2 + az**2))

        zt = np.zeros(self.state_dim)
        zt[0] = phi
        zt[1] = theta

        R = self.get_cur_R() # calculated from the current estimate of the mean
        
        KalmanGain = self.cur_cov @ self.C.T @ np.linalg.inv(self.C @ self.cur_cov @ self.C.T + R)

        self.cur_mean = self.cur_mean + KalmanGain @ (zt - self.C @ self.cur_mean)
        self.cur_cov = self.cur_cov - KalmanGain @ self.C @ self.cur_cov

        return self.cur_mean, self.cur_cov