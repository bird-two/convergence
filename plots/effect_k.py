'''
Plot FL and SL in different non-IID distribution
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

unbalanced = 1

from outputs import read_fromcsv

class FileNameCurve():
    '''select the files as the curves'''
    def __init__(self, args):
        self.args = args
        self.path = '../save/effect_k/{}/'.format(self.args.d)
        self.pdfpath = '../save/effect_k/'

    def get_title(self):
        #num_clients = self.args.clients
        if self.args.d == 'mnist':
            title = 'MNIST'
        elif self.args.d == 'fashionmnist':
            title = 'Fashion-MNIST'
        elif self.args.d == 'cifar10':
            title = 'CIFAR-10'
        elif self.args.d == 'cifar100':
            title = 'CIFAR100'
        return title

    def get_xlabel(self, x):
        xlabel_map = {'round': 'Rounds', 'iteration': 'Iterations'}
        return xlabel_map[x]
    
    def get_ylabel(self, t=0):
        ylabel_name = ['Train Loss', 'Test Loss', 'Train Top1 Accuracy', 'Test Top1 Accuracy', 'Train Top5 Accuracy', 'Test Top5 Accuracy']
        return ylabel_name[t]
    
    def get_legend(self):
        return self.fl_legend, self.sl_legend

    def compute_iterations(self, E):
        if self.args.d == 'mnist' or self.args.d == 'fashionmnist':
            total_n_sample = 60000
        elif self.args.d == 'cifar10' or self.args.d == 'cifar100':
            total_n_sample = 50000
        iterations = E * total_n_sample // self.args.b 
        return iterations
    
    def get_iterations(self):
        return self.sl_iterations

    def get_raw_files(self):
        raw_files = os.listdir(self.path)
        return raw_files

    def select_files(self):
        raw_files = self.get_raw_files()
        #dist_dict = { 0:'(iid-b)', 1:'(pat2-b 1.0)', 2:'(pat2-b 2.0)', 3:'(pat2-b 3.0)', 4:'(pat2-b 4.0)', 5:'(pat2-b 5.0)', 6:'(pat2-b 6.0)', 8:'(pat2-b 8.0)'}
        dist_dict = { 0:'(iid-b)', 10.0:'(dir-u 10.0)', 5.0:'(dir-u 5.0)', 1.0:'(dir-u 1.0)', 0.5:'(dir-u 0.5)' }
        if self.args.d == 'cifar10' or self.args.d == 'cifar100':
            fl_model_setup = 'vgg11'
            sl_model_setup = 'vgg11(4)'
            fl_alg = 'flv2'
        else:
            fl_model_setup = 'lenet5'
            sl_model_setup = 'lenet5(2)'
            fl_alg = 'fl'
        sl_alg = 'sl'

        fl_files = []
        sl_files = []
        self.fl_legend = []
        self.sl_legend = []
        self.sl_iterations = []
        for e in self.args.E:
            for d in self.args.dist:
                dist = dist_dict[d]
                fl_temp = 'base {}({} {} 1.0) {} {}{} sgd({} 0.9 0.0001) hp({})'.format(fl_alg, self.args.clients, e, fl_model_setup, self.args.d, dist, args.lr, args.b)
                sl_temp = 'base {}({} {} 1.0) {} {}{} sgd({} 0.9 0.0001) hp({})'.format(sl_alg, self.args.clients, e, sl_model_setup, self.args.d, dist, args.lr, args.b)
                dlengend = 'iid' if d == 0 else d
                if '{}.csv'.format(fl_temp) in raw_files:
                    fl_files.append(fl_temp) 
                    self.fl_legend.append('FL($\\alpha$={}) E={}'.format(dlengend, e))
                if '{}.csv'.format(sl_temp) in raw_files:
                    sl_files.append(sl_temp)
                    self.sl_legend.append('SL($\\alpha$={}) E={}'.format(dlengend, e))
                    self.sl_iterations.append(self.compute_iterations(e))
                
        return fl_files, sl_files
    

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default='mnist', help='dataset')
#parser.add_argument('--dist', type=int, default=0, help='distribution')
if unbalanced == 1:
    parser.add_argument('--dist', type=float, nargs='*', default=[0, 10.0, 5.0, 1.0, 0.5], help='different non-IID distribution')  
else:
    parser.add_argument('--dist', type=int, nargs='*', default=[0, 1, 2, 3, 4, 5, 6, 8], help='different non-IID distribution')
parser.add_argument('--clients', type=int, default=10, help='number of clients')
parser.add_argument('-t', type=int, default=0, help='test accuracy or train loss')
parser.add_argument('-b', type=int, default=10, help='mini-batch')
parser.add_argument('-x', type=str, default=0, choices=['round', 'iteration', 'amount'], help='X-axis')
parser.add_argument('--fl', type=int, default=1, help='Use FL or not')
parser.add_argument('--sl', type=int, default=1, help='Use SL or not')
parser.add_argument('-s', type=int, default=1, help='start epoch')
parser.add_argument('-e', type=int, default=400, help='end epoch')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('-E', type=int, nargs='*', default=[1, 2, 5, 10, 20, 50], help='local epochs')
args = parser.parse_args()
print(args)

# python effect_k.py -d cifar10 --dist 1.0 -b 100 --clients 10 -t 0 -x "round" --fl 0 --lr 0.005 -E 1 2 4 8 10 20 -e 100
# python effect_k.py -d fashionmnist --dist 1.0 -b 1000 --clients 10 -t 0 -x "round" --fl 0 --lr 0.01 -E 1 2 4 8 10 20
# python effect_k.py -d mnist --dist 1.0 -b 1000 --clients 10 -t 0 -x "round" --fl 0 --lr 0.01 -E 1 2 4 8 10 20 -e 30
# 1e-05 5e-05 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1
def plotcurve(args):
    fncurve = FileNameCurve(args=args)
    path = fncurve.path
    title = fncurve.get_title()
    xlabel = fncurve.get_xlabel(args.x)
    ylabel = fncurve.get_ylabel(args.t)

    fl_files, sl_files = fncurve.select_files()
    fl_legend, sl_legend = fncurve.get_legend()

    plt.figure(figsize=(6, 4)) #figsize=(8, 6) # figsize=(6.4, 4.8)
    start_epoch = args.s # 1
    #epochs = range(start_epoch, end_epoch+1)
    lines = []
    if args.fl == 1:
        for i in range(len(fl_files)):
            #print(fl_files[i])
            df = read_fromcsv(fl_files[i], path)
            end_epoch = min(args.e, len(df))
            x_axis = range(1, end_epoch+1)
            plt.plot(x_axis, df.iloc[start_epoch-1:end_epoch, args.t].values, linestyle='dashed', color=None, lw=2)
            lines.append('{}'.format(fl_legend[i]))
    
    steps = fncurve.get_iterations()
    if args.sl == 1:
        for i in range(len(sl_files)):
            df = read_fromcsv(sl_files[i], path)
            end_epoch = min(args.e, len(df))
            if args.x == 'iteration':
                x_axis = range(1*steps[i], end_epoch*steps[i]+1, steps[i])
            elif args.x == 'round':
                x_axis = range(1, end_epoch+1)
            plt.plot(x_axis, df.iloc[start_epoch-1:end_epoch, args.t].values, color=None, lw=2)
            lines.append('{}'.format(sl_legend[i]))
    
    #plt.title(title, fontsize=15)        
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    #plt.ylim(ymin=0, ymax=1)
    if args.x == 'iteration':
        plt.xlim(xmin=0, xmax=50*steps[0]) # 200000
    #plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ncol = 2 if args.fl == 1 and args.sl == 1 else 1
    plt.legend(lines, loc=None, ncol= ncol, prop={'size': 14}) # loc = 1 or 4
    plt.grid()
    #plt.savefig('{}/{} {}.png'.format(pathplus[0], title, ylabel))
    plt.savefig('{}/{} {}.pdf'.format(fncurve.pdfpath, title, ylabel), bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plotcurve(args) 

