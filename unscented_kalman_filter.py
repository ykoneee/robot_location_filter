from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints
import numpy as np
def get_ukf():
    def f_cv(x, dt,v_angle):
        v=v_angle[0]
        rad=v_angle[1]
        x[2]+=rad*dt
        x[2]%=2*np.pi    
        dist = (v*dt)
        x[0]+= np.cos(x[2]) * dist
        x[1]+= np.sin(x[2]) * dist          
        return x
    def h_cv(x):
        return x[:2]
    sigmas = MerweScaledSigmaPoints(3, alpha=.1, beta=2., kappa=0.)
    ukf = UnscentedKalmanFilter(dim_x=3, dim_z=2, fx=f_cv,
                                hx=h_cv, dt=0.05, points=sigmas)
    ukf.x = np.array([0., 0.,0.])
    ukf.R *= 0.05
    ukf.Q = Q_discrete_white_noise(3, dt=0.05, var=0.1)
    
    return ukf