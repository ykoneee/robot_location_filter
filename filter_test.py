import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time
from robot_utils import fake_robot,calculate_filter_error
from particle_filter import pf
from unscented_kalman_filter import get_ukf

robot=fake_robot(sensor_std=.14,control_std=.004)
origin_data=np.array([robot.sample() for _ in range(5000)])
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

def pftest(a='121'):
    global move_with_filter
    plt.subplot(a)
    init_point=move_with_noise[0]
    f=pf(init_point,[.5,.5,1000])
    move_with_filter=[]
    move_with_filter.append(np.array(init_point[:2]))
    t1=time.time()
    for i in range(1,len(move_with_noise)):
        f.predict(move_v_angle[i],[0.015,0.01], dt=0.05)
        f.update(move_with_noise[i][:2],0.09)
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
    plt.scatter(move_with_noise[:,0],move_with_noise[:,1],label='with_noise',s=10,alpha=0.3)
    plt.plot(move_with_filter[:,0],move_with_filter[:,1],label='with_filter')
    #plt.legend()
    plt.grid(True)
    ax=plt.gca()
    ax.set_aspect(1)
    calculate_filter_error('pf',a+1,move_no_noise,move_with_filter)
def ukftest(a=221):
    global move_with_filter
    ukf=get_ukf(R_std=0.05,Q_std=0.1)
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
    plt.scatter(move_with_noise[:,0],move_with_noise[:,1],label='with_noise',s=10,alpha=0.3)
    plt.plot(move_with_filter[:,0],move_with_filter[:,1],label='with_filter')
    plt.grid(True)
    ax=plt.gca()
    ax.set_aspect(1)    
    calculate_filter_error('ukf',a+1,move_no_noise,move_with_filter)
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
    ukftest(221)
    pftest(223)
    plt.show()
    
    #========================================
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