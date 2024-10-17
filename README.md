# AUV-DRCK
Repository for dead-reckoning algorithms, using sensors commonly available on AUVs


This reposity implements various algorithms for performing dead-reconing pose estimation of an AUV, containing different combinations of sensors like Accelerometer, Gyroscope, Magnetometer, Pressure/Depth, and a Doppler Velocity Log. More sensor will possible be added in the future.

Currently:
    - KalmanFilter for acc + gyro fusion for orientation estimation.
    - MadgwickFilter for acc + gyro + magnet for orientation estimation.