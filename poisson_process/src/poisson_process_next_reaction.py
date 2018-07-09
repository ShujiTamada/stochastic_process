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


class poisson_process_next_reaction(sde.SDE_Markov):
    def __init__(self,**keyargs):
        super(poisson_process_next_reaction, self).__init__(**keyargs)#becase of using for function,don't use self

        self.key=keyargs
        self.observation= self.key['observation']#noize matrix
        self.function_select=self.key['function_select']
        self.repeat_time=self.key['repeat_time']
        self.lam=self.key['lambda']
        self.repeat_time= self.key['repeat_time']
        self.slope= self.key['slope']

    def poisson_process(self,init,**keyargs):
        path_box,qv_box,times=self.outcome_output()
        poi=init
        accident=0
        path_box[:,0]=poi
        k=0
        count=0
        while accident<self.terminal:
            accident_time=np.random.exponential(1)#accident number per time　
            next_accident_time=accident+accident_time
            while accident/self.deltaT<=k<next_accident_time/self.deltaT and k+1<=self.terminal/self.deltaT:
                k=k+1
                path_box[:,k]=poi #事故回数を配列に入力
            k=k #これがないと
            poi=poi+1
            accident=next_accident_time
        return(path_box)


    '''
    method intensity

    ps: list of parameters
    mode : function choice. mode0 is a simple linear scaling

    '''

    def intensity(self,ps, mode = 0,**keyargs):
        if mode == 0:
            lamb = ps*self.slope
        else:
            lamb = np.inf
            raise notImplementedError
        return(lamb)

    def poisson_process_next_reaction(self,init,**keyargs):
        path_box,qv_box,times=self.outcome_output()
        poi=init#事故回数
        real_accident=0#事故発生時間
        path_box[:,0]=poi#時間毎の事故発生回数を入力する配列
        k=0
        #intensitymode = args.mode
        #intensityparameters = args.parameters
        while real_accident<self.terminal:#終点まで回す。
            #myintensity = self.intensity(intensityparameters, poi, mode = intensitymode )
            accident_time=np.random.exponential(1)#accident number per time　次の事故が怒る内部間隔
            real_scale=accident_time/self.intensity(poi,0)#次の事故が起こる現実での尺度での間隔
            next_real_accident_time=real_accident+real_scale#次の事故が起こる現実での尺度での時間
            while real_accident/self.deltaT<=k<next_real_accident_time/self.deltaT and k+1<=self.terminal/self.deltaT:
                #事故回数を記入
                k=k+1
                path_box[:,k]=poi
            k=k#k を更新しておかないと次のwhileで支障をきたす
            poi=poi+1#事故回数を一回増加
            real_accident=next_real_accident_time
        return(times,path_box)


    def integral_intensity(self,path_box):
        integral_volue=0#積分の値
        for k in range(int(1/self.deltaT)):
            integral_volue=integral_volue+self.intensity(path_box[0,k+1],0)*self.deltaT #離散で積分の値を合わせる。
        return(integral_volue)

    def oputional_sampling(self,init,simulate_time = 10):
        print("Initiating the simulation sequence...")
        integral_box=np.zeros(self.repeat_time)
        Y_t=np.zeros(self.repeat_time)
        for k in range(simulate_time):
            if np.mod(k, 100) == 0:
                print('%s paths complete'%k)
            times,observation=self.poisson_process_next_reaction(init)
            integral=self.integral_intensity(observation)
            integral_box[k]=integral
            Y_t[k]=observation[0,int(self.terminal/self.deltaT)]
        print("Simulations successfully completed!")
        average_integral=init+np.average(integral_box)
        average_Y_t=np.average(Y_t)
        return(average_integral, average_Y_t)


    def simulation(self,init,simulate_time = 10):
        print("Initiating the simulation sequence...")
        sampledat = []#box for recoding all data
        for k in range(simulate_time):
            if np.mod(k, 100) == 0:
                print('%s paths complete'%k)
            times,observation=self.poisson_process_next_reaction(init)
            sampledat.append(observation)
        sampledat = np.array(sampledat)
        print("Simulations successfully completed!")
        return(times, sampledat)


    def saveResult(self,figpath,times,path_box):
        times=times[0] #x axis of glaph
        plt.figure(figsize = (20,20))
        for k in range(len(path_box)):
            k_th_path = path_box[k,:]
            k_th_path=k_th_path[0]
            myfig = plt.plot(times, k_th_path, color='r')
        plt.savefig(figpath)
