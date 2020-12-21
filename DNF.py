from scipy import signal, random, math

import numpy as np

class DNF :

    def __init__(self,size=(45,45),sigmaexc=0.05,noise=False):
        self.dt = 0.1
        self.to = 0.5
        self.h = -4
        self.gain = 5
        self.sigmaexc=sigmaexc
        self.cofexc = 15
        self.gi = 250

        self.Map = np.zeros(size)
        self.Input_Map =self.gaussian_activity((0.3, 0.3), (0.7, 0.7), 0.1)  # ex 0.2 0.5
        self.g = self.gaussian_distribution((0.5, 0.5), self.Map.shape, 0.1)


        if(noise):
            self.Input_Map=self.generate_Noise(self.Input_Map,0.2,minval=0,maxval=0.3)

        self.K = self.generate_selection_Kernel((45,45), self.cofexc, self.sigmaexc, self.gi)
        self.Map_conv = signal.convolve2d(self.Map, self.K, mode='same')
        self.Map_sig = self.sigmoid(self.Map)


        #print(np.unravel_index(np.argmax(self.K, axis=None), self.K.sself.hape))


    def euclidean_dist(self,x,y):
        s=0
        for i in range(len(x)):
            s+=(x[i]-y[i])**2
        return math.sqrt(s)

    def gaussian_distribution(self,position, size, sigma):
        G=np.zeros(size)
        for i in range(size[0]):
            for j in range(size[1]):
                a,b=i/size[0],j/size[1]
                d=self.euclidean_dist(position,(a,b))
                G[i][j]=math.exp(-1/2*((d/sigma)**2))
        return G


    def gaussian_activity(self,a, b, sigma):
        Size=self.Map.shape[0]
        g1=self.gaussian_distribution(a,(Size,Size),sigma)
        g2=self.gaussian_distribution(b,(Size,Size),sigma)
        return g1+g2


    def gaussian_distribution_constant(self,position,size, sigma):
        g=self.gaussian_distribution(position,size,sigma)
        g/=np.sum(g)
        return g

    def generate_selection_Kernel(self,size, coef_excitation, sigma_excitation, global_hibition):
        g=self.gaussian_distribution_constant((0.5,0.5),((size[0]*2),(size[1]*2)),sigma_excitation)
        g*=coef_excitation
        g-=(global_hibition/((size[0]*2)*(size[1]*2)))
        return g

    def sigmoid(self,x):
      s=1/(1+np.exp(-x))
      #return np.ones(x.shape)-s
      s=((s*(-1))+np.sum(s))
      return s/s.size
      a=(x*(-1))+np.sum(x)
      return 1/(1+np.exp(-a))

    def set_input(self,input_map):
      self.Input_Map=signal.fftconvolve(input_map,self.g, mode='same')


    def normliz(self,Map):

        max = np.max(np.abs(Map))
        if max!=0:
            Map/=max

        for i in range(Map.shape[0]):
            for j in range(Map.shape[1]):
                if (Map[i,j]<0):
                    Map[i,j]=0
                if (Map[i,j]>1):
                    Map[i,j]=1

        return Map

    def generate_Noise(self,Input,rate,minval=0,maxval=1):
        noise=np.zeros(Input.shape)
        for i in range(Input.shape[0]):
            for j in range(Input.shape[1]):
                noise[i,j]=0
                uniform_noise = random.uniform(0,1)
                if uniform_noise <= rate:
                    noise[i,j] = random.uniform(minval,maxval)
        return Input+noise


    def update_neuron(self,p):
         u=self.Map[p[0],p[1]]
         us=self.Map_sig[p[0],p[1]]
         conv=self.Map_conv[p[0],p[1]]
         return u+self.dt/self.to*(-u+self.h+conv+self.Input_Map[p[0],p[1]]*self.gain)


    def RK_F(self,x, u,dt):
         us=self.Map_sig[x[0],x[1]]
         conv=self.Map_conv[x[0],x[1]]
         return u+dt/self.to*(-u+self.h+us*conv+self.Input_Map[x[0],x[1]]*self.gain)

    def rK_update(self,x):
            rK1 = self.RK_F(x, self.Map[x[0]][x[1]], self.dt)
            rK2 = self.RK_F(x, self.Map[x[0]][x[1]]+rK1* self.dt/2 , self.dt/2)
            rK3 = self.RK_F(x, self.Map[x[0]][x[1]]+rK2* self.dt/2 , self.dt/2)
            rK4 = self.RK_F(x, self.Map[x[0]][x[1]]+rK3* self.dt   , self.dt)
            return (self.dt/6)*(rK1+2*rK2+2*rK3+rK4)


    def syncronous_run(self, update=0):
                Map_dt = np.zeros(self.Map.shape)
                self.Map_sig = self.sigmoid(self.Input_Map)
                self.Map_conv = signal.fftconvolve(self.Map, self.K, mode='same')

                for i in range(self.Map.shape[0]):
                    for j in range(self.Map.shape[1]):
                        if update == 0:
                            Map_dt[i,j] = self.update_neuron((i,j))
                        if update == 1:
                            Map_dt[i,j] = self.rK_update((i,j))
                            # if update==2:
                            # self.Map_self.dt[i,j]=LIF_update((i, j),seuil)

                self.Map = self.normliz(Map_dt)
                return self.Map

    def Asynchronous_run(self, update=0):
                Map_time = np.random.rand(self.Map.shape[0], self.Map.shape[1])/10
                indx = np.unravel_index(np.argsort(Map_time, axis=None), Map_time.shape)
                indx = np.transpose(indx)
                dt_prec = self.dt

                self.Map_sig = self.sigmoid(self.Input_Map)
                self.Map_conv = signal.fftconvolve(self.Map, self.K, mode='same')
                for x in indx:
                        i, j = x[0], x[1]
                        # print(i,j)
                        self.dt = Map_time[i,j]
                        if update == 0:
                            self.Map[i, j] = self.update_neuron((i, j))
                        if update == 1:
                            self.Map[i, j] = self.rK_update((i, j))
                            # if update==2:
                            #   self.Map[i,j]=LIF_update((i,j),seuil)

                self.Map = self.normliz(self.Map)
                self.dt=dt_prec
                return self.Map

    #Leaself.Ky_Integrate_Fire
    # Wij distance normalisée
    #seuil=0.9
    #R=10 # fuite
    #self.h=0.1

'''    

    def In_spiskes(self,x):
        S=0
        for i in range(self.Map.shape[0]):
            for j in range(self.Map.shape[1]):
                w=self.euclidean_dist(x,(i,j))/(self.size[0]*self.size[1])
                s=self.Map[i,j]==1
                S+=w*s
        return S

    def LIF_update(x,s):
        u = self.Map[x[0],x[1]]
        if u>s :
            return 0
        val=u+self.h*((-u/R) + self.Input_Map[x[0],x[1]])+In_spiself.Kes(x)
        if(val>s) :
            return 1
        return val
'''










