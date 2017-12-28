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

        self.lam=self.key['lambda']

    def model_select(self,**keyargs):
        if self.function_select=='standard':#standard brownian mortion
            path_box,times=self.standard_poisson()
        elif self.function_select=='inhomogeneous':#standard brownian mortion
            path_box,times=self.inhomogeneous_poisson()
        else:
            print('manuke')
        if self.observation=='path':
            observation=path_box[0]
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
            times,path=self.model_select()
            sampledat.append(path)
        sampledat = np.array(sampledat)
        print("Simulations successfully completed!")
        return(times, sampledat)


    def saveResult(self, figpath, times, path_box):
        times=times[0]
        plt.figure(figsize = (20,20))
        for k in range(len(path_box)):
            k_th_path = path_box[k]
            myfig = plt.plot(times, k_th_path,ls='steps', color='m')
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
        for k in range(self.division):
            new_position = self.one_step(now_position)
            path_box[:,k+1]=new_position
            now_position = new_position
        return(path_box,times)

    def standard_poisson(self,**keyargs):
        path_box,qv_box,times=self.outcome_output()
        now_position=self.init
        path_box[:,0]=now_position
        for k in range(self.division):
            new_position = self.one_step(now_position)
            path_box[:,k+1]=new_position
            now_position = new_position
        return(path_box,times)

    def integrate_function(self,x):
        return (x)

    def inhomogeneous_poisson(self,**keyargs):
        path_box,qv_box,times=self.outcome_output()
        now_position=self.init
        path_box[:,0]=now_position
        for k in range(self.division):
            random_variable_T=np.random.poisson(self.integrate_function(k*self.deltaT)*self.deltaT,self.dimen).reshape(self.dimen,1)
            now_position_T=now_position.reshape(self.dimen,1)
            new_position=now_position_T+ random_variable_T
            path_box[:,k+1]=new_position
            now_position = new_position
        return(path_box,times)
