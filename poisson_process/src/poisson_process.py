#poisson process class

import sys
sys.path.append("../src")
sys.path.append("../data")
sys.path.append("../fig")

sys.path.append("../../main_folder/src/")#road a file
import SDE_class as sde #road  SDE_class file

import importlib #funciton for doing reload

import numpy as np
from matplotlib import pyplot as plt


import math
from scipy import integrate
import argparse
import pdb
import os


class poisson_process(sde.SDE_Markov):
    def __init__(self,**keyargs):
        super(poisson_process, self).__init__(**keyargs)#becase of using for function,don't use self

        self.key=keyargs
        self.observation= self.key['observation']#noize matrix
        self.function_select=self.key['function_select']
        self.repeat_time=self.key['repeat_time']
        self.lam=self.key['lambda']
        self.repeat_time= self.key['repeat_time']

    def model_select(self,**keyargs):
        if self.function_select=='standard':#standard brownian mortion
            path_box,qv_box,times=self.standard_poisson()
        elif self.function_select=='inhomogeneous':#standard brownian mortion
            path_box,qv_box,times=self.inhomogeneous_poisson()
        else:
            print('manuke')
        if self.observation=='path':
            observation=path_box[0]
        if self.observation=='qv':
            observation=qv_box[0]
        return(times,observation)

    """
    this function repeat simulation in time simulate_time
    """

    def simulation(self, simulate_time = 10):
        print("Initiating the simulation sequence...")
        sampledat = []#box for recoding all data
        for k in range(simulate_time):
            if np.mod(k, 100) == 0:
                print('%s paths complete'%k)
            times,observation=self.model_select()
            sampledat.append(observation)
        sampledat = np.array(sampledat)
        print("Simulations successfully completed!")
        return(times, sampledat)


    def saveResult(self,figpath,times,path_box):
        times=times[0] #x axis of glaph
        plt.figure(figsize = (20,20))
        for k in range(len(path_box)):
            k_th_path = path_box[k,:]
            myfig = plt.plot(times, k_th_path, color='r')
        plt.savefig(figpath)

    def one_step(self,now_position):
        random_variable_T=np.random.poisson(self.lam*self.deltaT,self.dimen).reshape(self.dimen,1)
        #make to noize
        now_position_T=now_position.reshape(self.dimen,1)
        #updatevalue =  np.dot(self.transform_matrix_step, now_position_T)*self.deltaT\
        #+np.dot(self.noise_var_matrix, random_variable_T)*np.sqrt(self.deltaT)
        new_position=now_position_T+ random_variable_T
        now_position = new_position
        return(new_position)

    def standard_poisson(self,**keyargs):
        path_box,qv_box,times=self.outcome_output()
        now_position=self.init
        path_box[:,0]=now_position
        #now_time=0
        for k in range(self.division):
            #now_time=now_time+self.deltaT
            random_variable=np.random.poisson(self.lam*self.deltaT,self.dimen).reshape(self.dimen,1)
            new_position = now_position+random_variable-self.lam*self.deltaT
            path_box[:,k+1]=new_position
            qv_box[:,k+1]=qv_box[:,k]+(new_position-now_position)**2
            now_position = new_position
        return(path_box,qv_box,times)



    def integrate_function(self,x):
        return (np.exp(x))

    def inhomogeneous_poisson(self,**keyargs):
        path_box,qv_box,times=self.outcome_output()
        now_position=self.init
        path_box[:,0]=now_position
        qv=0
        qv_box[:,0]=qv
        for k in range(self.division):
            random_variable_T=np.random.poisson(self.integrate_function(k*self.deltaT)*self.deltaT,self.dimen).reshape(self.dimen,1)
            now_position_T=now_position.reshape(self.dimen,1)
            new_position=now_position_T+ random_variable_T-self.integrate_function(k*self.deltaT)*self.deltaT
            path_box[:,k+1]=new_position
            qv=qv+(new_position-now_position)**2
            qv_box[:,k+1]=qv
            now_position = new_position
        return(path_box,qv_box,times)

    def inhomo_poi_martin(self,time_s,**keyargs):
        path_box,qv_box,times=self.outcome_output()
        now_position=self.init
        path_box[:,0]=now_position
        sampledat=np.zeros([self.repeat_time,self.division+1])
        time=0
        for k in range(time_s):
            random_variable=np.random.poisson(integrate.romberg(self.integrate_function,k*self.deltaT,(k+1)*self.deltaT),1)#start of for rope number is 0
            new_position=now_position+ random_variable-integrate.romberg(self.integrate_function,k*self.deltaT,(k+1)*self.deltaT)
            path_box[:,k+1]=new_position
            now_position = new_position
        s_position=now_position
        print(s_position)
        for j in range(self.repeat_time):
            now_position=s_position
            time=int(time_s*self.deltaT)
            if np.mod(j, 100) == 0:
                print('%s paths complete'%j)
            for k in range(int(self.division-time_s)):
                random_variable=np.random.poisson(integrate.romberg(self.integrate_function,time_s+k*self.deltaT,time_s+(k+1)*self.deltaT),1)
                new_position=now_position+random_variable-integrate.romberg(self.integrate_function,time_s+k*self.deltaT,time_s+(k+1)*self.deltaT)
                path_box[:,time_s+k+1]=new_position
                now_position = new_position
            sampledat[j,:]=path_box[0]
        return(sampledat,s_position,times)


    def saveplot(self,figpath,times,sampledat,s_position): #figpath:save place,times:x axis,sampledat:y axis,s_position:conditional exp time
        times=times[0] #x axis of glaph
        plt.figure(figsize = (20,20))
        for k in range(self.repeat_time):
            k_th_path = sampledat[k]
            myfig = plt.plot(times, k_th_path)
        numpath,numstep = sampledat.shape
        lastval= sampledat[:,numstep-1]#terminal value
        meanval =  np.mean(lastval)#mean of terminal value
        varval  = np.var(lastval)#var of terminal value
        Cexp_minus_time_s=meanval-s_position
        print(Cexp_minus_time_s)
        label = np.arange(0,self.terminal+0.5,0.5)
        plt.xticks(label, label,fontsize=30)
        plt.title("$N_t-\int_0^t \lambda(s)ds:\lambda(x)=x$",fontsize=70)
        plt.xlabel("time",fontsize=50)
        plt.ylabel("value",fontsize=50)
        plt.text(0, meanval, r'$E[N_t-\int_0^t \lambda(s)ds|\mathcal{F}_s]-X_s$=%s'%(Cexp_minus_time_s),fontsize=30)
        plt.savefig(figpath)
