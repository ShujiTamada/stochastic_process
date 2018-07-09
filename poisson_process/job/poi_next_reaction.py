# poisson process inhomogeneous
import sys
sys.path.append("../src")#road a file
sys.path.append("../data")#road a file
sys.path.append("../fig")#road a file

sys.path.append("../../main_folder/src/")#road a file
import SDE_class as sde
from poisson_process_next_reaction import poisson_process_next_reaction as poi_next
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
    init=np.array([1.])#init_value

    sdekey={}
    sdekey['default'] = False
    sdekey['init'] = init
    sdekey['stepsize'] = args.step
    sdekey['term'] = args.terminal
    sdekey['repeat_time'] = args.repeat_time
    sdekey['slope'] = args.slope
    '''
    傾きが大きいと事故が頻繁に起こるのでデルタを細かくしないと誤差が大きい。
    '''

    sdekey['lambda'] = 1
    sdekey['observation'] = args.observation
    sdekey['function_select'] = args.function



    figplace = '../fig'#move to fig file
    jobplace = '../data'
    figname= str('inhomo_poi_next_reaction.png')
    arrayname= str('inhomo_poi_next_reaction')

    arraypath = os.path.join(jobplace,arrayname)
    figpath = os.path.join(figplace,figname)


    mymodel = sde.SDE_Markov(**sdekey)
    mypoi=poi_next(**sdekey)

    #times,path_box=mypoi.poisson_process_next_reaction(init)
    #integral_volue=mypoi.integral_intensity(path_box)
    #print(path_box)
    #print(integral_volue)

    average_integral, average_Y_t=mypoi.oputional_sampling(init,args.repeat_time)
    theory_value=math.e**args.slope
    print(average_integral, average_Y_t,'theory value is %s'%theory_value)
    #times,path_box=mypoi.poisson_process_next_reaction(init)








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
    parser.add_argument('--slope', '-sl', type=int, default =5,  help='intensity slope')

    parser.add_argument('--function', '-f', type=str, default ='standard',  help='function of the random walk')
    parser.add_argument('--observation', '-m', type=str, default ='qv',  help='mode of the random walk')

    args= parser.parse_args()
    #pdb.set_trace()
    main()
