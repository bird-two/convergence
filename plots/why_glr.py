'''
Plot FL and SL with different learning rate
SIZE: title=15, legend=14, label=14, ticks=12
'''
from msilib.schema import Error
import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

#import sys
#sys.path.append('D:\\study\\birds\\Convegence Analysis of SL and FL\\EXP\\')

#from utils.outputs import read_fromcsv

from outputs import read_fromcsv

class Data():
    '''select the files as the curves'''
    def __init__(self):
        self.mnist_cd = [4.1295e-6, 0.0001, 0.0003, 0.0041, 0.0292, 1.9039, 3.8719, 24.5378, 50.1447]
        self.mnist_sd = [1.0138e-5, 0.0002, 0.0009, 0.0204, 0.0771, 3.5278, 9.6727, 63.7398, 98.9363]
        self.fashionmnist_cd = []
        self.fashionmnist_sd = []
        self.cifar10_cd = []
        self.cifar10_sd = []

        self.llr = [1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    
    def get_y(self, raw):
        #b = np.array(raw)
        y = []
        for i in range(len(self.llr)):
            y.append(raw[i] / (self.llr[i])**2)
        return y

        #self.llr = [1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
        
        #self.path = '../save/effect_glr/{}/'.format(self.args.d)

# python cmp_glr.py -d mnist --dist 5 -b 10 --clients 10 -t 3 -x "round" --fl 0

def plotcurve():
    data = Data()
    c = 1 
    s = 1
    #title = fncurve.get_title()
    #xlabel = fncurve.get_xlabel(args.x)
    #ylabel = fncurve.get_ylabel(args.t)

    #fl_files, sl_files = fncurve.select_files()
    #fl_legend, sl_legend = fncurve.get_legend()

    plt.figure()
    #epochs = range(start_epoch, end_epoch+1)
    lines = []
    if c == 1:
        x_axis = data.llr
        y = data.get_y(data.mnist_cd)
        plt.plot(x_axis, y, linestyle='dashed', color=None, lw=2, marker='d')
        lines.append('{}'.format('mnist client'))
    
    if s == 1:
        x_axis = data.llr
        y = data.get_y(data.mnist_sd)
        plt.plot(x_axis, y, color=None, lw=2, marker='d')
        lines.append('{}'.format('mnist server'))
    
    #plt.title(title, fontsize=15)        
    #plt.ylabel(ylabel, fontsize=14)
    #plt.xlabel(xlabel, fontsize=14)
    #plt.ylim(ymin=0, ymax=3.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ncol = 2 if c == 1 and s == 1 else 1
    plt.legend(lines, loc=None, ncol= ncol, prop={'size': 14}) # loc = 1 or 4
    plt.grid()
    #plt.savefig('{}/{} {}.png'.format(pathplus[0], title, ylabel))
    #plt.savefig('{}/{} {}.pdf'.format(path, title, ylabel), bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plotcurve() 

