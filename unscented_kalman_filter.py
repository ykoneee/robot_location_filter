from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints
import numpy as np
def get_ukf(R_std,Q_std,dt):
    def f_(x, dt,v_angle):
        v=v_angle[0]
        rad=v_angle[1]
        x[2]+=rad*dt
        x[2]%=2*np.pi    
        dist = (v*dt)
        x[0]+= np.cos(x[2]) * dist
        x[1]+= np.sin(x[2]) * dist          
        return x
    def h_(x):
        return x[:2]
    sigmas = MerweScaledSigmaPoints(3, alpha=.1, beta=2., kappa=0.)
    ukf = UnscentedKalmanFilter(dim_x=3, dim_z=2, fx=f_,
                                hx=h_, dt=dt, points=sigmas)
    ukf.x = np.array([0., 0.,0.])
    ukf.R *= R_std
    ukf.Q = Q_discrete_white_noise(3, dt=dt, var=Q_std)
    
    return ukf