import numpy as np
import matplotlib.pyplot as plt
from filterpy.monte_carlo  import systematic_resample
import scipy.stats
class fake_robot():
    def __init__(self,std=0.1):
        self.a=0
        self.v=1.5
        self.angle=0
        self.x=0
        self.y=0
        self.state=0
        self.state_holdcount=100
        self.sampleHZ=20
        self.noise_std=std
        self.rotate_acc=20
    def sample(self):
        
        #self.v=self.v+self.a*1/self.sampleHZ
        self.state_holdcount+=1
        if self.state_holdcount>30:
            self.state_holdcount=0
            if np.random.rand()<0.85:
                tempnum1=np.random.randint(0,3)
                while self.state==tempnum1:
                    tempnum1=np.random.randint(0,3)
                self.state=tempnum1
            if self.state==0:
                self.v=1+np.random.rand()
                self.rotate_acc=0
                #self.angle+=np.random.rand()*270-135
            elif self.state==1:
                self.v=0.5+np.random.uniform(0,1)
                if np.random.rand()<0.5:
                    self.rotate_acc=np.random.uniform(20,35)
                else:
                    self.rotate_acc=-np.random.uniform(20,35)
                #self.angle+=np.random.rand()*270-135
        self.angle+=self.rotate_acc*(1/self.sampleHZ)+np.random.randn()
        self.x+=self.v*(1/self.sampleHZ)*np.cos(np.deg2rad(self.angle))+np.random.randn()*0.004
        self.y+=self.v*(1/self.sampleHZ)*np.sin(np.deg2rad(self.angle))+np.random.randn()*0.004       
        #return self.x,self.y,np.random.uniform(0,2*np.pi),self.x+np.random.uniform(-0.5,0.5),self.y+np.random.uniform(-0.5,0.5),np.random.uniform(0,2*np.pi)
        return self.v,np.deg2rad(self.angle),self.x,self.y,np.random.uniform(0,2*np.pi),self.x+np.random.randn()*self.noise_std,self.y+np.random.randn()*self.noise_std,np.random.uniform(0,2*np.pi)
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
        self.N=1000
        #self.particles=self.normal_particles(init_point,std,self.N)
        self.particles=self.uniform_particles([-4,4],[-4,4],[1,2],self.N)
        self.weights = np.zeros(self.N)
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
    def predict(self, u, std, dt=1.,angle_increase_mode=False):
        
        if angle_increase_mode:
            self.particles[:, 2] += u[1] + (np.random.randn(self.N) * std[1])
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
            indexes = systematic_resample(self.weights)
            self.resample_from_index(indexes)        
robot=fake_robot(std=.11)
origin_data=np.array([robot.sample() for _ in range(2000)])
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
    plt.hist(err,bins=150)
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
    f=pf(init_point,[.5,.5,10000])
    move_with_filter=[]
    move_with_filter.append(np.array(init_point[:2]))
    for i in range(1,len(move_with_noise)):
        if i%500==0:print(i)
        
        f.predict(move_v_angle[i],[0.045,0.2], dt=0.05)
        f.update(move_with_noise[i][:2],0.1)
        f.resample()
        mean0,std0=f.estimate()
        #plt.scatter(f.particles[:,0],f.particles[:,1],s=2,alpha=0.2,label=f'particles:{i}')
        move_with_filter.append(mean0)
        #plt.scatter(old[:,0],old[:,1],s=3,label='old')
        #plt.scatter(new[:,0],new[:,1],s=3,label='new')
        #plt.scatter(mean0[0],mean0[1],s=3,c='k')
        
    move_with_filter=np.array(move_with_filter)
    plt.scatter(move_with_filter[:,0],move_with_filter[:,1],label='with_filter',s=40,c='g')
    plt.scatter(move_no_noise[:,0],move_no_noise[:,1],label='origin',s=30,c='r')
    plt.scatter(move_with_noise[:,0],move_with_noise[:,1],label='with_noise',s=10,alpha=0.5)
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
    calculate_filter_error('pf','122')

def pftest2():
    n=15
    plt.figure(1)
    init_point=move_no_noise[0]
    f=pf(init_point,[0.01,0.01,10000])
    move_with_filter=[]
    move_with_filter.append(np.array(move_with_noise[0,:2]))
    for i in range(1,n):
        print(i)
        f.predict([1.5,0],[0.05,10000], dt=0.05)
        f.update(move_with_noise[i][:2],0.1)
        f.resample()
        mean0,std0=f.estimate()
        move_with_filter.append(mean0)
        #plt.scatter(old[:,0],old[:,1],s=3,label='old')
        plt.scatter(f.particles[:,0],f.particles[:,1],s=2,alpha=0.2,label=f'particles:{i}')
        #plt.scatter(mean0[0],mean0[1],s=30,c='k')
        plt.scatter(move_no_noise[i,0],move_no_noise[i,1],s=30,c='r')
    move_with_filter=np.array(move_with_filter)
    plt.plot(move_with_filter[:,0],move_with_filter[:,1])
    plt.plot(move_with_noise[:n,0],move_with_noise[:n,1])
    #plt.legend()
    plt.grid(True)
    ax=plt.gca()
    ax.set_aspect(1)
    #fig=plt.gcf()
    #fig.set_size_inches(8,8,forward=True)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    #plt.show()       
if __name__=='__main__':
    #pftest2()
    pftest('121')
    #swtest('121')
    #plt.hist(1.5+np.random.randn(10000)*0.15,bins=50)
    plt.show()