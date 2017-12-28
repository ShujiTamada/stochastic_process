import numpy as np
import SDE_class as sde
from matplotlib import pyplot as plt
import sys
import os
import pdb

figpath = "../figs"
datpath = "../data"
unko = 0


def simulate(filename,term,step,init):
    mymodel = sde.SDE_Markov(mymat=np.array([[0.]]), myvar =np.array([[1.]]),\
    myinit=init,myscale=1.,myterm=term,step_size=step)

    #Trajectory must also return time for the real implementation
    trajectory = mymodel.many_step(now_position=init)

    #YOU MUST CHANGE THIS!!!
    times = np.arange(0,term+step,step)

    plt.plot(times, trajectory[0])
    filepath= os.path.join(figpath, filename)
    datFilepath= os.path.join(datpath, filename)
    plt.savefig(filepath)
    np.save(datFilepath,trajectory)
