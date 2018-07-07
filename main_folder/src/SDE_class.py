
#class to implementation SDE

import numpy as np#Pakkege must be put outside class
import pdb
from matplotlib import pyplot as plt



'''
mymat:trancefform matrix
mymat_noise:noise term trancefform atrix
myvar:noise matrix
myterm:terminal
mydiv:
mymean:noise Guiss mean
myscale:noise Guiss var
myinit:初期値 second dimension
step_size:
'''

'''
X(t)=X(t-1)+mymat*X(t-1)+myvar*normal
'''

class SDE_Markov:
    def __init__(self, **keyargs):


        self.key = keyargs
        #it can ignore order to use dictionary style
        #value to use class
        self.default_initialize()
        self.terminal= self.key['term'] #terminal
        self.deltaT = self.key['stepsize']
        self.division= int(np.ceil(self.terminal/self.deltaT))#divisiton number
        self.dimen = len(self.key['init'])#dimension
        self.init= self.key['init'].reshape(self.dimen, 1)#初期値(ベクトル)

        self.sanity_check()

    def default_initialize(self):#put in defolt value
        if self.key['default']==True :
            self.key['init']=np.array([1.])
        else:
            pass

    def outcome_output(self,**keyargs):
        path_box=np.zeros((self.dimen,self.division+1)) #box for recoding path value
        qv_box=np.zeros((self.dimen,self.division+1)) #box for recoding quadalic valuation
        times=np.arange(0,self.terminal+self.deltaT,self.deltaT) #array of time
        times = np.asarray(times).reshape(1,len(times))
        return(path_box,qv_box,times)



    def one_step(self,now_position=np.array([1,1])):
        now_position = new_position
        return(new_position)

    def many_step(self, **keyargs):
        path_box,qv_box,times=self.outcome_output
        now_position=self.init
        path_box[:,0]=now_position
        for k in range(division):
            new_position = self.one_step(now_position)
            path_box[:,k+1]=new_position
            now_position = new_position
        return(path_box)

    def simulation(self, numsamples = 10):
        sampledat = []
        for k in range(numsamples):
            times, trajectory_box = self.many_step()
            trajectory = trajectory_box[0]
            sampledat.append(trajectory)
        return(times, sampledat)

    def saveFig(self,figpath, numsamples= 10):
        times, trajectory_box = self.simulation(numsamples)
        for k in range(len(trajectory_box)):
            k_th_trajectory = trajectory_box[k]
            myfig = plt.plot(times, k_th_trajectory, color='r')
        plt.savefig(figpath)


    def sanity_check(self):#alarm
        pass

        '''
        if len(self.transform_matrix_step.shape)<2 or\
         len(self.noise_var_matrix.shape)<2:
            print("you must input a matrix for the transformation and noise!!")
        else:
            mymatD, _ = self.transform_matrix_step.shape
            mymatD2, _ =  self.noise_var_matrix.shape
            if mymatD !=  self.dimen:
                print("Transformation dimension mismatch!! ")
            if mymatD2 !=  mymatD:
                print("noise and transformation dimension mismatch!! ")
        '''
#pdb.set_trace()　coad looking for error
#np.array([[],[],[]]) matrix outline
#np.dot(,) matrix times
