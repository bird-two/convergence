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

from outputs import read_fromcsv

unbalanced = 0

class FileNameCurve():
    '''select the files as the curves'''
    def __init__(self, args):
        self.args = args
        self.pdfpath = '../save/cmp_lr/'
        if unbalanced == 1:
            self.path = '../save/cmp_lr/{}/'.format(self.args.d)
        else:
            self.path = '../save/cmp_lr/{}/'.format(self.args.d)      

    def get_title(self):
        num_clients = self.args.clients
        if self.args.d == 'mnist':
            title = 'MNIST'.format(num_clients)
        elif self.args.d == 'fashionmnist':
            title = 'Fashion-MNIST'.format(num_clients)
        elif self.args.d == 'cifar10':
            title = 'CIFAR-10'.format(num_clients)
        elif self.args.d == 'cifar100':
            title = 'CIFAR-100'.format(num_clients)
        return title

    def get_xlabel(self, x):
        xlabel_map = {'round': 'Rounds', 'iteration': 'Iterations'}
        return xlabel_map[x]
    
    def get_ylabel(self, t=0):
        ylabel_name = ['Train Loss', 'Test Loss', 'Train Top1 Accuracy', 'Test Top1 Accuracy', 'Train Top5 Accuracy', 'Test Top5 Accuracy']
        return ylabel_name[t]
    
    def get_legend(self):
        return self.fl_legend, self.sl_legend

    def get_raw_files(self):
        raw_files = os.listdir(self.path)
        return raw_files

    def select_files(self):
        raw_files = self.get_raw_files()
        if unbalanced == 1:
            dist_dict = { 0:'(iid-b)', 10.0:'(dir-u 10.0)', 5.0:'(dir-u 5.0)', 1.0:'(dir-u 1.0)', 0.5:'(dir-u 0.5)' }
        else:
            dist_dict = { 0:'(iid-b)', 1:'(pat2-b 1.0)', 2:'(pat2-b 2.0)', 3:'(pat2-b 3.0)', 4:'(pat2-b 4.0)', 5:'(pat2-b 5.0)', 6:'(pat2-b 6.0)', 8:'(pat2-b 8.0)'}
        
        if self.args.d == 'cifar10' or self.args.d == 'cifar100':
            fl_model_setup = 'vgg11'
            sl_model_setup = 'vgg11(4)'
            fl_alg_setup = 'flv2({} 1 1.0)'.format(self.args.clients)
        else:
            fl_model_setup = 'lenet5'
            sl_model_setup = 'lenet5(2)'
            fl_alg_setup = 'fl({} 1 1.0)'.format(self.args.clients)
        sl_alg_setup = 'sl({} 1 1.0)'.format(self.args.clients)

        fl_files = []
        sl_files = []
        self.fl_legend = []
        self.sl_legend = []
        for lr in self.args.flr:
            for d in self.args.dist:
                dist = dist_dict[d]
                fl_temp = 'base {} {} {}{} sgd({} 0.9 0.0001) hp({})'.format(fl_alg_setup, fl_model_setup, self.args.d, dist, lr, args.b)
                dlengend = 'iid' if d == 0 else d
                if '{}.csv'.format(fl_temp) in raw_files:
                    fl_files.append(fl_temp)
                    if len(self.args.flr) > 1:
                        self.fl_legend.append('FL({}) lr={}'.format(dlengend, lr))
                    else:
                        if unbalanced == 1 and d != 0:
                            self.fl_legend.append('FL($\\alpha={}$)'.format(dlengend))
                        else:
                            self.fl_legend.append('FL($C={}$)'.format(dlengend))
        
        for lr in self.args.slr:
            for d in self.args.dist:
                dist = dist_dict[d]
                sl_temp = 'base {} {} {}{} sgd({} 0.9 0.0001) hp({})'.format(sl_alg_setup, sl_model_setup, self.args.d, dist, lr, args.b)
                dlengend = 'iid' if d == 0 else d
                if '{}.csv'.format(sl_temp) in raw_files:
                    sl_files.append(sl_temp)
                    if len(self.args.slr) > 1:
                        self.sl_legend.append('SL({}) lr={}'.format(dlengend, lr))
                    else:
                        if d == 0:
                            self.sl_legend.append('SL({})'.format(dlengend))
                        elif unbalanced == 1:
                            self.sl_legend.append('SL($\\alpha={}$)'.format(dlengend))
                        else:
                            self.sl_legend.append('SL($C={}$)'.format(dlengend))
        
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
parser.add_argument('--flr', type=float, nargs='*', default=[1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], help='sl learning rate list')
parser.add_argument('--slr', type=float, nargs='*', default=[1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], help='fl learning rate list')
#parser.add_argument('--time', type=int, default=0, help='x-axis is time ?')
#parser.add_argument('-f', type=str, nargs='*', default='', help='folder')
#parser.add_argument('--choose', type=str, nargs='*', default=[], help='chooses')
#parser.add_argument('--ban', type=str, nargs='*', default=[], help='ban')
args = parser.parse_args()
print(args)

# python cmp_noniid.py -x "round" -d mnist --dist 0 8 5 2 --clients 10 -b 1000 -t 0 --slr 0.01 --fl 0 -e 30
# python cmp_noniid.py -x "round" -d fashionmnist --dist 0 8 5 2 --clients 10 -b 1000 -t 0 --slr 0.01 --fl 0
# python cmp_noniid.py -x "round" -d cifar10 --dist 0 8 5 2 --clients 10 -b 100 -t 0 --slr 0.001 --fl 0

# 1e-05 5e-05 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1
def plotcurve(args):
    fncurve = FileNameCurve(args=args)
    path = fncurve.path
    title = fncurve.get_title()
    xlabel = fncurve.get_xlabel(args.x)
    ylabel = fncurve.get_ylabel(args.t)

    fl_files, sl_files = fncurve.select_files()
    fl_legend, sl_legend = fncurve.get_legend()

    fig = plt.figure(figsize=(6, 4)) #figsize=(8, 6)
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
    
    if args.sl == 1:
        for i in range(len(sl_files)):
            df = read_fromcsv(sl_files[i], path)
            end_epoch = min(args.e, len(df))
            x_axis = range(1, end_epoch+1)
            plt.plot(x_axis, df.iloc[start_epoch-1:end_epoch, args.t].values, color=None, lw=2)
            lines.append('{}'.format(sl_legend[i]))
    
    #ax1.set_xlabel('x')
    #ax1.set_ylabel('y')
    #ax1.set_title('title inside 1')
    
    #plt.title(title, fontsize=15)        
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    if args.d == 'cifar10':
        plt.ylim(ymin=0, ymax=2.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ncol = 2 if args.fl == 1 and args.sl == 1 else 1
    plt.legend(lines, loc=None, ncol= ncol, prop={'size': 14}) # loc = 1 or 4
    plt.grid()

    #df = read_fromcsv('base sl(100 1 1.0) lenet5(2) mnist(pat2-b 1.0) sgd(0.003 0.9 0.0001) hp(10)', path)
    #ax1 = fig.add_axes([0.4, 0.6, 0.2, 0.2])  # inside axes
    #x_axis = range(1, end_epoch+1)
    #ax1.plot(x_axis, df.iloc[start_epoch-1:end_epoch, args.t].values, color='orange', lw=2)

    #plt.savefig('{}/{} {}.png'.format(pathplus[0], title, ylabel))
    plt.savefig('{}/{} {}.pdf'.format(fncurve.pdfpath, title, ylabel), bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plotcurve(args) 

