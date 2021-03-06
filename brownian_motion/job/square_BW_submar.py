#練習
import sys
sys.path.append("../src")#road a file
sys.path.append("../data")#road a file
sys.path.append("../fig")#road a file

sys.path.append("../../main_folder/src/")#road a file
import SDE_class as sde
from brown_motion import brown_motion as bmt
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

    sdekey['matrix'] = np.array([[0.]])
    sdekey['var_matrix'] = np.array([[1.]])
    sdekey['n_mean'] = args.mean# random variable mean of noise
    sdekey['n_scale'] = args.variance# random variable variance of noise
    sdekey['observation'] = args.observation
    sdekey['function_select'] = args.function
    sdekey['repeat_time'] = args.repeat_time

    figplace = '../figs'#move to fig file
    jobplace = '../data'
    figname= str('square_submar.png')
    arrayname= str('square_submar.npy')

    arraypath = os.path.join(jobplace,arrayname)
    figpath = os.path.join(figplace,figname)


    mymodel = sde.SDE_Markov(**sdekey)
    mybmt=bmt(**sdekey)

    time_s=0.5
    sampledat,s_position,times=mybmt.square_BW_submar(int(time_s/args.step))
    np.save(arraypath,sampledat)
    mybmt.saveplot(figpath,times,sampledat,s_position)

    #print('over %spaths, mean at time%s is %s. var is %s'%(numpath,args.terminal, meanval, varval))

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
    parser.add_argument('--step', '-s', type=float, default =0.001,  help='step size')
    parser.add_argument('--mean', '-me', type=int, default =0,  help='noize nomal mean')
    parser.add_argument('--variance', '-v', type=int, default =1,  help='noize nomal variance')

    parser.add_argument('--function', '-f', type=str, default ='square',  help='function of the random walk')
    parser.add_argument('--observation', '-m', type=str, default ='path',  help='mode of the random walk')

    args= parser.parse_args()
    #pdb.set_trace()
    main()
