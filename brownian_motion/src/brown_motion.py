#brown_motion trancefform

import sys
sys.path.append("../src")
sys.path.append("../data")
sys.path.append("../fig")

sys.path.append("../../main_folder/src/")#road a file
import SDE_class as sde
import importlib #funciton for doing reload
import numpy as np
from matplotlib import pyplot as plt

import math
from scipy.stats import norm
import argparse
import pdb
import os


class brown_motion(sde.SDE_Markov):
    def __init__(self,**keyargs):
        super(brown_motion, self).__init__(**keyargs)#becase of using for function,don't use self

        self.key=keyargs
        self.normal_mean= self.key['n_mean']#noize nomal mean
        self.normal_scale= self.key['n_scale']#noize nomal variance
        self.transform_matrix_step= self.key['matrix']#transition matrix
        self.noise_var_matrix= self.key['var_matrix']#noize matrix

        self.observation= self.key['observation']#noize matrix
        self.function_select=self.key['function_select']

    """
    this function is format of recoding simulation outcome
    """
    def outcome_output(self,**keyargs):
        path_box=np.zeros((self.dimen,self.division+1)) #box for recoding path value
        qv_box=np.zeros((self.dimen,self.division+1)) #box for recoding quadalic valuation
        times=np.arange(0,self.terminal+self.deltaT,self.deltaT) #array of time
        times = np.asarray(times).reshape(1,len(times))
        return(path_box,qv_box,times)

    """
    this funciton runs to select model.
    """

    def model_select(self,**keyargs):
        if self.function_select=='standard':#standard brownian mortion
            times, path_box,qv_box=self.standard_brownian_motion()
        elif self.function_select=='square':#square brownian mortion
            times, path_box,qv_box=self.square_brownian_motion()
        elif self.function_select=='Doob_mayer':#simulation of Doob mayer
            times, path_box,qv_box=self.brown_motion_Doob_mayer()
        else:
            print('manuke')
            raise NotImplementedError
        if self.observation=='qv':
            observation=qv_box[0]
        if self.observation=='path':
            observation=path_box[0]
        return(times,observation)



    def one_step(self,now_position):
        random_variable_T=np.random.normal(self.normal_mean,self.normal_scale,self.dimen).reshape(self.dimen,1)
        #make to noize
        now_position_T=now_position.reshape(self.dimen,1)
        updatevalue =  np.dot(self.transform_matrix_step, now_position_T)*self.deltaT\
        +np.dot(self.noise_var_matrix, random_variable_T)*np.sqrt(self.deltaT)
        new_position=now_position_T+ updatevalue
        now_position = new_position
        #translate next step
        return(new_position)

    def many_step(self,**keyargs):
        path_box,qv_box,times=self.outcome_output()
        now_position=self.init
        path_box[:,0]=now_position
        for k in range(self.division):
            new_position = self.one_step(now_position)
            path_box[:,k+1]=new_position
            now_position = new_position
        return(path_box,qv_box,times)
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


    def saveResult(self, figpath, times, path_box):
        times=times[0]
        plt.figure(figsize = (20,20))
        for k in range(len(path_box)):
            k_th_path = path_box[k]
            myfig = plt.plot(times, k_th_path, color='black')
        plt.savefig(figpath)

    """
    this funciton runs standard brownian motion
    """

    def standard_brownian_motion(self,**keyargs):
        path_box,qv_box,times=self.outcome_output() #pick up format to record
        now_position = self.init
        path_box[:,0]=now_position
        qv_box[:,0]=0
        for k in range(self.division):
            new_position = self.one_step(now_position)
            qv_box[:,k+1]=qv_box[:,k]+(now_position-new_position)**2
            path_box[:,k+1]=new_position
            now_position = new_position
        return(times,path_box,qv_box)

    """
    this funciton runs square brownian motion
    """

    def square_brownian_motion(self,**keyargs):
        path_box,qv_box,times=self.outcome_output()#pick up format to record
        time,path_BM,qv_box=self.standard_brownian_motion()#make standard brownian motion
        now_position = path_BM[:,0]
        now_position_sq = now_position**2
        path_box[:,0]=now_position_sq
        qv_box[:,0]=0
        for k in range(self.division):#make square brownian motion
            new_position = path_BM[:,k]
            new_position_sq = new_position**2
            qv_box[:,k+1]=qv_box[:,k]+(new_position_sq-now_position_sq)**2
            path_box[:,k+1]=new_position_sq
            now_position = new_position
            now_position_sq = new_position_sq
        return(times,path_box,qv_box)

    """
    this funciton simulation Doob mayer
    simulation which square brownian motion minus quadalic valuation is martingale.
    """

    def brown_motion_Doob_mayer(self,**keyargs):
        path_box,qv_box,times=self.outcome_output()
        now_position = self.init #init value
        now_position_square = self.init**2 #square brownian motion init value
        now_position_doob = self.init**2
        now_position_time = 0.
        path_box[:,0]=now_position_doob
        qv_box[:,0]=0
        for k in range(self.division):
            new_position = self.one_step(now_position)# make standard brownian mortion
            new_position_time = now_position_time + self.deltaT
            new_position_square = new_position**2 # make square brownian mortion
            new_position_doob=new_position_square - now_position_time #square minus quadalic varuation
            qv_box[:,k+1]=qv_box[:,k]+(new_position_doob-now_position_doob)**2
            path_box[:,k+1]=new_position_doob
            now_position = new_position
            now_position_square = new_position_square
            now_position_time = new_position_time
            now_position_doob = new_position_doob
        return(times,path_box,qv_box)
