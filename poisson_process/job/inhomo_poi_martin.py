# poisson process inhomogeneous
import sys
sys.path.append("../src")#road a file
sys.path.append("../data")#road a file
sys.path.append("../fig")#road a file

sys.path.append("../../main_folder/src/")#road a file
import SDE_class as sde
from poisson_process import poisson_process as poi
import importlib #function to reroad a file
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.stats import norms
import argparse
import pdb
import os

def main():
    div=args.terminal/args.step#jump_number
    init=np.array([0.])#init_value

    sdekey={}
    sdekey['default'] = False
    sdekey['init'] = init
    sdekey['stepsize'] = args.step
    sdekey['term'] = args.terminal
    sdekey['repeat_time'] = args.repeat_time

    sdekey['lambda'] = 1
    sdekey['observation'] = args.observation
    sdekey['function_select'] = args.function



    time_s=0.4

    figplace = '../fig'#move to fig file
    jobplace = '../data'
    figname= str('inhomo_poi_martin.png')
    arrayname= str('inhomo_poi_martin')

    arraypath = os.path.join(jobplace,arrayname)
    figpath = os.path.join(figplace,figname)


    mymodel = sde.SDE_Markov(**sdekey)
    mypoi=poi(**sdekey)

    sampledat,s_position,times=mypoi.inhomo_poi_martin(int(time_s/args.step))

    np.save(arraypath,sampledat)
    #print(sampledat)
    mypoi.saveplot(figpath,times,sampledat,s_position)

if __name__ == '__main__':
    '''
    how to use argparse
    1. write "argparse.ArgumentParser(description='runnning parameters')"
    2. add parameter
       for example  "parser.add_argument('--repeat_time', '-n', type=int, default =10,  help='number of trajectories')"
         parser.add_argument('parameter name, command(dicide myself),defalt value, meaning of parameter')
    3. write args= parser.parse_args()
    4. write
    '''
    parser = argparse.ArgumentParser(description='runnning parameters')
    parser.add_argument('--repeat_time', '-n', type=int, default =1000,  help='number of trajectories')
    parser.add_argument('--terminal', '-t', type=int, default =1,  help='terminal time')
    parser.add_argument('--step', '-s', type=float, default =0.005,  help='step size')
    parser.add_argument('--mean', '-me', type=int, default =0,  help='noize nomal mean')
    parser.add_argument('--variance', '-v', type=int, default =1,  help='noize nomal variance')

    parser.add_argument('--function', '-f', type=str, default ='standard',  help='function of the random walk')
    parser.add_argument('--observation', '-m', type=str, default ='qv',  help='mode of the random walk')

    args= parser.parse_args()
    #pdb.set_trace()
    main()
