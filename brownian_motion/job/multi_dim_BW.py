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
import util as util
import math
from scipy.stats import norm
import argparse
import pdb
import os

def main():

    #SDE_Markov variation_number
    term=1.#terminal_time
    step=0.25#step_size
    div=term/step#jump_number
    init=np.zeros((2,1))#init_value

    mean=np.zeros((2,2))
    variance=np.identity((2,2))

    repeat_time=1000

    function='standrd'#select observation function
    observation='path'#select observation value


    sdekey={}
    sdekey['default'] = False
    sdekey['init'] = init
    sdekey['stepsize'] = step
    sdekey['term'] = term

    sdekey['matrix'] = np.zeros((2,2))
    sdekey['var_matrix'] = np.identity(2)

    sdekey['n_mean'] = mean
    sdekey['n_scale'] = variance

    sdekey['observation'] = observation
    sdekey['function_select'] = function

    figplace = '../figs'#move to fig file
    figname= str('multi_dim.png')
    arrayname= str('multi_dim.npy')
    arraypath = os.path.join(figplace,arrayname)
    figpath = os.path.join(figplace,figname)


    mymodel = sde.SDE_Markov(**sdekey)
    mybmt=bmt(**sdekey)


    times,trajectory_box=mybmt.simulation(repeat_time)

    np.save(arraypath,trajectory_box)
    #mybmt.saveResult(figpath, times, trajectory_box)
    numpath,numstep = trajectory_box.shape
    lastval= trajectory_box[:,numstep-1]#terminal value
    meanval =  np.mean(lastval)#mean of terminal value
    varval  = np.var(lastval)#var of terminal value



    print('over %spaths, mean at time%s is %s. var is %s'%(numpath,term, meanval, varval))







if __name__ == '__main__':
    main()
