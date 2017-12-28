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

    #SDE_Markov variation_number
    div=args.terminal/args.step#jump_number
    init=np.array([0.])#init_value

    sdekey={}
    sdekey['default'] = False
    sdekey['init'] = init
    sdekey['stepsize'] = args.step
    sdekey['term'] = args.terminal

    sdekey['matrix'] = np.array([[0.]])
    sdekey['var_matrix'] = np.array([[1.]])
    sdekey['n_mean'] = args.mean
    sdekey['n_scale'] = args.variance

    sdekey['observation'] = args.observation
    sdekey['function_select'] = args.function

    figplace = '../figs'#move to fig file
    figname= str('Doob_mayar%s.png'%sdekey['observation'])
    arrayname= str('Doob_mayar_%s.npy'%sdekey['observation'])
    arraypath = os.path.join(figplace,arrayname)
    figpath = os.path.join(figplace,figname)


    mymodel = sde.SDE_Markov(**sdekey)
    mybmt=bmt(**sdekey)

    times,trajectory_box=mybmt.simulation(args.repeat_time)

    np.save(arraypath,trajectory_box)
    mybmt.saveResult(figpath, times, trajectory_box)


    numpath,numstep = trajectory_box.shape
    lastval= trajectory_box[:,numstep-1]#terminal value
    meanval =  np.mean(lastval)#mean of terminal value
    varval  = np.var(lastval)#var of terminal value

    print('over %spaths, mean at time%s is %s. var is %s'%(numpath,args.terminal, meanval, varval))

    #simulate.saveFig(brownian_motion_sq,repeat_time)
    #np.save(brownian_motion_sq,trajectory)








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
    parser.add_argument('--repeat_time', '-r', type=int, default =100,  help='number of trajectories')
    parser.add_argument('--terminal', '-t', type=int, default =50,  help='terminal time')
    parser.add_argument('--step', '-s', type=float, default =0.01,  help='step size')
    parser.add_argument('--mean', '-me', type=int, default =0,  help='noize nomal mean')
    parser.add_argument('--variance', '-v', type=int, default =1,  help='noize nomal variance')

    parser.add_argument('--function', '-f', type=str, default ='Doob_mayer',  help='function of the random walk')
    parser.add_argument('--observation', '-m', type=str, default ='path',  help='mode of the random walk')

    args= parser.parse_args()
    #pdb.set_trace()
    main()
