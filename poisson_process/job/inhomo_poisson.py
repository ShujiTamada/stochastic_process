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
from scipy.stats import norm
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
    sdekey['lambda'] = 1

    sdekey['observation'] = args.observation
    sdekey['function_select'] = args.function

    figplace = '../fig'#move to fig file

    figname= str('inhomogeneous_poisson.png')
    arrayname= str('inhomogeneous_poisson')

    arraypath = os.path.join(figplace,arrayname)
    figpath = os.path.join(figplace,figname)


    mymodel = sde.SDE_Markov(**sdekey)
    mypoi=poi(**sdekey)




    times,path_box=mypoi.simulation(args.repeat_time)
    np.save(arraypath,path_box)

    #mypoi.saveResult(figpath, times, path_box)

    numpath,numstep = path_box.shape
    lastval= path_box[:,numstep-1]#terminalinal value
    meanval =  np.mean(lastval)#mean of terminalinal value
    varval  = np.var(lastval)#var of terminalinal value
    print('over %spaths, mean at time%s is %s. var is %s'%(numpath,args.terminal, meanval, varval))









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
    parser.add_argument('--repeat_time', '-n', type=int, default =10000,  help='number of trajectories')
    parser.add_argument('--terminal', '-t', type=int, default =1,  help='terminal time')
    parser.add_argument('--step', '-s', type=float, default =0.001,  help='step size')
    parser.add_argument('--function', '-f', type=str, default ='inhomogeneous',  help='function of the random walk')
    parser.add_argument('--observation', '-m', type=str, default ='path',  help='mode of the random walk')

    args= parser.parse_args()
    #pdb.set_trace()
    main()
