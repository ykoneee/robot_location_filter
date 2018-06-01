from filterpy.monte_carlo  import systematic_resample
import numpy as np
from numpy.random import randn,uniform
import scipy.stats



class pf():
    def __init__(self,init_point,std):
        self.N=1000
        self.particles=self.normal_particles(init_point,std,self.N)
        #self.particles=self.uniform_particles([-2,2],[-2,2],[1,2],self.N)
        self.weights = np.zeros(self.N)
        self.resample_time=0
    def uniform_particles(self,x_range, y_range, v_range, N):
        particles = np.empty((N, 3))
        particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
        particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
        particles[:, 2] = uniform(v_range[0], v_range[1], size=N)
        particles[:, 2] %= 2 * np.pi
        return particles
    def normal_particles(self,mean, std, N):
        particles = np.empty((N, 3))
        particles[:, 0] = mean[0] + (randn(N) * std[0])
        particles[:, 1] = mean[1] + (randn(N) * std[1])
        particles[:, 2] = mean[2] + (randn(N) * std[2])
        particles[:, 2] %= 2 * np.pi
        return particles    
    def predict(self, u, std, dt=1.,angle_increase_mode=True):
        
        if angle_increase_mode:
            self.particles[:, 2] += u[1]*dt + (randn(self.N) * std[1])
        else:
            self.particles[:, 2] = u[1] + (randn(self.N) * std[1])
        self.particles[:, 2] %= 2 * np.pi          

        dist = (u[0] * dt) + (randn(self.N) * std[0])
        self.particles[:, 0] += np.cos(self.particles[:, 2]) * dist
        self.particles[:, 1] += np.sin(self.particles[:, 2]) * dist  
    def update(self,z,std):
        z=np.array(z)
        distance=np.linalg.norm(self.particles[:,0:2]-z, axis=1)
        self.weights=scipy.stats.norm(distance,std).pdf(0)
        self.weights+=1.e-300
        self.weights/=np.sum(self.weights)
    def estimate(self):
        pos = self.particles[:, 0:2]
        mean = np.average(pos, weights=self.weights, axis=0)
        var  = np.average((pos - mean)**2, weights=self.weights, axis=0)
        return mean, var    
    def neff(self):
        return 1. / np.sum(np.square(self.weights))  
    def resample_from_index(self,indexes):
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights.fill (1.0 / len(self.weights))    
    def resample(self):
        if self.neff() < self.N/2:
            #print('resample!')
            self.resample_time+=1
            indexes = systematic_resample(self.weights)
            self.resample_from_index(indexes)     