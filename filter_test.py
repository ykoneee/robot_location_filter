import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from filterpy.monte_carlo  import systematic_resample
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints
import scipy.stats
import time
from numpy.random import randn,uniform
class fake_robot():
    def __init__(self,sensor_std=0.1,control_std=0.001):
        self.a=0
        self.v=1.5
        self.angle=0
        self.x=0
        self.y=0
        self.state=0
        self.state_holdcount=0
        self.dt=0.05
        self.sensor_std=sensor_std
        self.control_std=control_std
        self.rotate_acc=0
    def sample(self):
        self.state_holdcount+=1
        if self.state_holdcount>80:
            self.state_holdcount=0
            if uniform()<0.7:
                tempnum1=np.random.randint(0,3)
                while self.state==tempnum1:
                    tempnum1=np.random.randint(0,3)
                self.state=tempnum1
            if self.state==0:
                self.v=1+uniform()
                self.rotate_acc=0
            elif self.state==1:
                self.v=0.6+uniform(0,1)
                if uniform()<0.5:
                    self.rotate_acc=np.deg2rad(uniform(4,12))
                else:
                    self.rotate_acc=np.deg2rad(-uniform(4,12))
                
        self.angle+=self.rotate_acc*self.dt+randn()*.01*self.control_std
        dist=self.v*self.dt+abs(randn())*self.control_std
        self.x+=np.cos(self.angle)*dist
        self.y+=np.sin(self.angle)*dist
        return self.v,self.rotate_acc,\
               self.x,self.y,self.angle,\
               self.x+randn()*self.sensor_std,self.y+randn()*self.sensor_std,self.angle+randn()*0.01*self.sensor_std
class sliding_window_filter():
    def __init__(self,window_size):
        self.datalist=[]
        self.window_size=window_size
    def update(self,x):
        self.datalist.append(x)
        if len(self.datalist)>self.window_size:
            self.datalist.pop(0)
    def getvalue(self):
        return np.mean(self.datalist,axis=0)
class pf():
    def __init__(self,init_point,std):
        self.N=500
        self.particles=self.normal_particles(init_point,std,self.N)
        #self.particles=self.uniform_particles([-2,2],[-2,2],[1,2],self.N)
        self.weights = np.zeros(self.N)
        self.resample_time=0
    def uniform_particles(self,x_range, y_range, v_range, N):
        particles = np.empty((N, 3))
        particles[:, 0] = np.random.uniform(x_range[0], x_range[1], size=N)
        particles[:, 1] = np.random.uniform(y_range[0], y_range[1], size=N)
        particles[:, 2] = np.random.uniform(v_range[0], v_range[1], size=N)
        particles[:, 2] %= 2 * np.pi
        return particles
    def normal_particles(self,mean, std, N):
        particles = np.empty((N, 3))
        particles[:, 0] = mean[0] + (np.random.randn(N) * std[0])
        particles[:, 1] = mean[1] + (np.random.randn(N) * std[1])
        particles[:, 2] = mean[2] + (np.random.randn(N) * std[2])
        particles[:, 2] %= 2 * np.pi
        return particles    
    def predict(self, u, std, dt=1.,angle_increase_mode=True):
        
        if angle_increase_mode:
            self.particles[:, 2] += u[1]*dt + (np.random.randn(self.N) * std[1])
        else:
            self.particles[:, 2] = u[1] + (np.random.randn(self.N) * std[1])
        self.particles[:, 2] %= 2 * np.pi          

        dist = (u[0] * dt) + (np.random.randn(self.N) * std[0])
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
class ekf():
    def __init__(self):
        pass
robot=fake_robot(sensor_std=.1,control_std=.04)
origin_data=np.array([robot.sample() for _ in range(3000)])
move_v_angle=origin_data[:,:2]
move_no_noise=origin_data[:,2:5]
move_with_noise=origin_data[:,5:]
move_with_filter=None
def refrashdata():
    global robot,origin_data,move_v_angle,move_no_noise,move_with_noise,move_with_filter
    robot=fake_robot(std=.1)
    origin_data=np.array([robot.sample() for _ in range(3000)])
    move_v_angle=origin_data[:,:2]
    move_no_noise=origin_data[:,2:5]
    move_with_noise=origin_data[:,5:]
    move_with_filter=None
def calculate_filter_error(name,subname):
    a=move_no_noise[:,:2]
    b=move_with_filter
    if b.shape[1]!=2:
        b=b[:,:2]
    err=np.linalg.norm(a-b,axis=1)
    plt.subplot(subname)
    plt.xlim(0,0.6)
    sns.distplot(err,bins=100)
    print(f'{name} err:avg:{err.mean():.3f} std:{err.std():.3f} max:{err.max():.3f} min:{err.min():.3f}')
def swtest(subplotname='151'):
    global move_with_filter
    plt.subplot(subplotname)
    move_with_filter=[]
    f=sliding_window_filter(window_size=10)
    for point in move_with_noise:
        f.update(point)
        move_with_filter.append(f.getvalue())
    move_with_filter=np.array(move_with_filter)
    #plt.plot(x,move_with_noise,'k',x,move_with_filter,'c',x,move_no_noise,'r')
    plt.scatter(move_with_filter[:,0],move_with_filter[:,1],label='with_filter',s=20,c='g')
    plt.scatter(move_no_noise[:,0],move_no_noise[:,1],label='origin',s=10,c='r')
    plt.scatter(move_with_noise[:,0],move_with_noise[:,1],label='with_noise',s=10,alpha=0.5)
    plt.grid(True)
    ax=plt.gca()
    ax.set_aspect(1)
    #ax.spines['right'].set_color('none')
    #ax.spines['top'].set_color('none')
    #ax.xaxis.set_ticks_position('bottom')
    #ax.spines['bottom'].set_position(('data',0))
    #ax.yaxis.set_ticks_position('left')
    #ax.spines['left'].set_position(('data',0))
    #plt.show()
    calculate_filter_error('swf','122')
def pftest(a='121'):
    global move_with_filter
    plt.subplot(a)
    init_point=move_with_noise[0]
    f=pf(init_point,[.5,.5,1000])
    move_with_filter=[]
    move_with_filter.append(np.array(init_point[:2]))
    t1=time.time()
    for i in range(1,len(move_with_noise)):
        f.predict(move_v_angle[i],[0.015,0.03], dt=0.05)
        f.update(move_with_noise[i][:2],0.08)
        f.resample()
        mean0,std0=f.estimate()
        #plt.scatter(f.particles[:,0],f.particles[:,1],s=2,alpha=0.2,label=f'particles:{i}')
        move_with_filter.append(mean0)
        #plt.scatter(old[:,0],old[:,1],s=3,label='old')
        #plt.scatter(new[:,0],new[:,1],s=3,label='new')
        #plt.scatter(mean0[0],mean0[1],s=3,c='k')
        if i%500==0:
            print(i,(time.time()-t1)/500)        
            t1=time.time()
    move_with_filter=np.array(move_with_filter)
    #plt.scatter(move_with_filter[:,0],move_with_filter[:,1],label='with_filter',s=15,c='g')
    
    plt.scatter(move_no_noise[:,0],move_no_noise[:,1],label='origin',s=9,c='r')
    plt.scatter(move_with_noise[:,0],move_with_noise[:,1],label='with_noise',s=10,alpha=0.1)
    plt.plot(move_with_filter[:,0],move_with_filter[:,1],label='with_filter')
    #plt.legend()
    plt.grid(True)
    ax=plt.gca()
    ax.set_aspect(1)
    #fig=plt.gcf()
    #fig.set_size_inches(8,8,forward=True)
    #ax.spines['right'].set_color('none')
    #ax.spines['top'].set_color('none')
    #ax.xaxis.set_ticks_position('bottom')
    #ax.spines['bottom'].set_position(('data',0))
    #ax.yaxis.set_ticks_position('left')
    #ax.spines['left'].set_position(('data',0))
    #plt.show()    
    calculate_filter_error('pf',a+1)
def ukftest(a=221):
    global move_with_filter
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
    sigmas = MerweScaledSigmaPoints(3, alpha=.01, beta=2., kappa=0.)
    ukf = UnscentedKalmanFilter(dim_x=3, dim_z=2, fx=f_cv,
              hx=h_cv, dt=0.05, points=sigmas)
    ukf.x = np.array([0., 0.,0.])
    ukf.R *= 0.05
    ukf.Q = Q_discrete_white_noise(3, dt=0.05, var=0.1)
    uxs = []  
    t1=time.time()
    for i in range(len(move_with_noise)):
        ukf.predict(v_angle=move_v_angle[i])
        ukf.update(move_with_noise[i][:2])
        uxs.append(ukf.x.copy())
        if i%500==0:
            print(i,(time.time()-t1)/500)
            t1=time.time()        
    uxs = np.array(uxs)
    move_with_filter=np.array([uxs[:, 0], uxs[:, 1]]).T
    plt.subplot(a)
    plt.scatter(move_no_noise[:,0],move_no_noise[:,1],label='origin',s=9,c='r')
    plt.scatter(move_with_noise[:,0],move_with_noise[:,1],label='with_noise',s=10,alpha=0.1)
    plt.plot(move_with_filter[:,0],move_with_filter[:,1],label='with_filter')
    plt.grid(True)
    ax=plt.gca()
    ax.set_aspect(1)    
    calculate_filter_error('ukf',a+1)
def ukf_aug_search(R=5,Q_var=0.1,alpha=0.05):
    global move_with_filter
    def f_cv(x, dt,v_angle):
        v=v_angle[0]
        rad=v_angle[1]
        dx=np.cos(rad)*v*dt
        dy=np.sin(rad)*v*dt
        x[0]+=dx
        x[1]+=dy
        return x
    def h_cv(x):
        return x    
    sigmas = MerweScaledSigmaPoints(2, alpha=alpha, beta=2., kappa=1.)
    ukf = UnscentedKalmanFilter(dim_x=2, dim_z=2, fx=f_cv,
                                hx=h_cv, dt=0.05, points=sigmas)
    ukf.x = np.array([0., 0.])
    ukf.R *= R
    ukf.Q = Q_discrete_white_noise(2, dt=0.05, var=Q_var)
    uxs = []  
    t1=time.time()
    for i in range(len(move_with_noise)):
        ukf.predict(v_angle=move_v_angle[i])
        ukf.update(move_with_noise[i][:2])
        uxs.append(ukf.x.copy())
        #if i%500==0:
            #print(i,(time.time()-t1)/500)
            #t1=time.time()        
    move_with_filter = np.array(uxs)
    a=move_no_noise[:,:2]
    b=move_with_filter
    if b.shape[1]!=2:
        b=b[:,:2]
    err=np.linalg.norm(a-b,axis=1)
    print(f'avg:{err.mean():.3f} std:{err.std():.3f} max:{err.max():.3f} min:{err.min():.3f}')    
    return err.mean()
if __name__=='__main__':
    #plt.figure(1)
    ukftest(121)
    #pftest(223)
    #swtest('121')
    plt.show()
    #l=[]
    #for R in np.linspace(0.01,10):
        #aug=0
        #for i in range(10):
            #refrashdata()
            #aug+=ukf_aug_search(R=R)
        #aug/=10
        #print([R,aug])
        #l.append([R,aug])
        
    #print(l)