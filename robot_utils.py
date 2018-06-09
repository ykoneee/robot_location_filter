from numpy.random import randn,uniform,randint
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns
class fake_robot():
    def __init__(self,sensor_std=0.1,control_std=0.001,dt=0.05):
        self.a=0
        self.v=1.5
        self.angle=0
        self.x=0
        self.y=0
        self.state=0
        self.state_holdcount=60
        self.dt=dt
        self.sensor_std=sensor_std
        self.control_std=control_std
        self.rotate_acc=0
    def sample(self):
        self.state_holdcount+=1
        if self.state_holdcount>90:
            self.state_holdcount=0
            if uniform()<0.8:
                tempnum1=randint(0,3)
                while self.state==tempnum1:
                    tempnum1=randint(0,3)
                self.state=tempnum1
            if self.state==0:
                self.v=0.8+uniform()
                self.rotate_acc=0
            elif self.state==1:
                self.v=0.7+uniform(0,1)
                if uniform()<0.5:
                    self.rotate_acc=np.deg2rad(uniform(4,12))
                else:
                    self.rotate_acc=np.deg2rad(-uniform(4,12))
            elif self.state==2:
                self.v=0
                self.rotate_acc=0
        
        real_control_std=0 if self.state==2 else self.control_std
        self.angle+=self.rotate_acc*self.dt+randn()*.06*real_control_std
        dist=self.v*self.dt+abs(randn())*real_control_std
        self.x+=np.cos(self.angle)*dist
        self.y+=np.sin(self.angle)*dist
        return self.v,self.rotate_acc,\
               self.x,self.y,self.angle,\
               self.x+randn()*self.sensor_std,self.y+randn()*self.sensor_std,self.angle+randn()*0.01*self.sensor_std
    
def calculate_filter_error(filter_name,subplot_name,data_ori,data_filter):
    a=data_ori[:,:2]
    b=data_filter
    if b.shape[1]!=2:
        b=b[:,:2]
    err=np.linalg.norm(a-b,axis=1)
    plt.subplot(subplot_name)
    plt.xlim(0,0.5)
    sns.distplot(err,bins=300)
    plt.vlines(err.mean(), 0, 100,'r')
    print(f'{filter_name} err:avg:{err.mean():.3f} std:{err.std():.3f} max:{err.max():.3f} min:{err.min():.3f}')