import numpy as np
import pyquaternion as pq


class MadgwickFilter:
    """This class implements the Madgwick filter for orientation tracking. It performs
    sensor fusion of accelerometer, gyroscope and magnetometer data."""

    def __init__(self, Hz = 100) -> None:
        
        self.Hz = Hz
        self.dt = 1/self.Hz

        self.earth_in_sensor = pq.Quaternion(1,0,0,0) # earth relative to sensor
        self.beta = 1 # algorithm gain parameter
        self.zeta = 0

        self.g_bias = np.zeros(3)

    # def list2quat(self, quat_list):
    #     return pq.Quaternion(quat_list)


    def update(self, gyro, acc, magnet, dt=None):
        """all inputs are three element numpy arrays"""

        if dt is None:
            dt = self.dt

        # Weird measurement detection and normalization
        if np.linalg.norm(acc) == 0:
            return None
        else:
            acc = acc/np.linalg.norm(acc)

        if np.linalg.norm(magnet) == 0:
            return None
        else:
            magnet = magnet/np.linalg.norm(magnet)

        magnet_quat = pq.Quaternion([0, magnet[0], magnet[1], magnet[2]])
        q = self.earth_in_sensor

        # Get the direction of the magnetic field in the  Earth frame
        # Here we assume that the Earth's magnetic field has componenets in one horizontal axis and the vertical axis.
        h = q * magnet_quat * q.conjugate
        b = [0, np.linalg.norm(h[1:3]), 0, h[3]]

        # Now we calculate one step of gradient descent
        
        F = np.array([2 * (q[1] * q[3] - q[0] * q[2]) - acc[0],
                      2 * (q[0] * q[1] + q[2] * q[3]) - acc[1],
                      2 * (0.5 - q[1]**2 - q[2]**2) - acc[2],
                      2 * b[1] * (0.5 - q[2]**2 - q[3]**2) + 2 * b[3] * (q[1] * q[3] - q[0] * q[2]) - magnet[0],
                      2 * b[1] * (q[1] * q[2] - q[0] * q[3]) + 2 * b[3] * (q[0] * q[1] + q[2] * q[3]) - magnet[1],
                      2 * b[1] * (q[0] * q[2] + q[1] * q[3]) + 2 * b[3] * (0.5 - q[1]**2 - q[2]**2) - magnet[2]])
        
        J = np.array([[-2 * q[2], 2 * q[3], -2 * q[0], 2 * q[1]],
                      [2 * q[1], 2 * q[0], 2 * q[3], 2 * q[2]],
                      [0, -4 * q[1], -4 * q[2], 0],
                      [-2 * b[3] * q[2], 2 * b[3] * q[3], -4 * b[1] * q[2] - 2 * b[3] * q[0], -4 * b[1] * q[3] + 2 * b[3] * q[1]],
                      [-2 * b[1] * q[3] + 2 * b[3] * q[1], 2 * b[1] * q[2] + 2 * b[3] * q[0], 2 * b[1] * q[1] + 2 * b[3] * q[3], -2 * b[1] * q[0] + 2 * b[3] * q[2]],
                      [2 * b[1] * q[2], 2 * b[1] * q[3] - 4 * b[3] * q[1], 2 * b[1] * q[0] - 4 * b[3] * q[2], 2 * b[1] * q[1]]])
        
        step = J.T @ F
        step = step / np.lingalg.norm(step)

        # Computer gyro drift bias
        gyro_errors = 2 * q.conjugate * step
        self.g_bias += self.zeta * gyro_errors[1:] * dt
        
        gyro = gyro - self.g_bias
        gyro_quat = pq.Quaternion([0, gyro[0], gyro[1], gyro[2]])

        # Calculate the rate of change
        qDot = 0.5 * q * gyro_quat - self.beta * step.T

        # Finally, integrate
        q = q + qDot*dt
        q = q / np.linalg.norm(q)

        # Finally update the position of earth in the sensor frame
        self.earth_in_sensor = q

        return q.conjugate