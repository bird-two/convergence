'''
Plot FL and SL
STANDARD SIZE: title=15, legend=14, label=14, ticks=12
'''
from msilib.schema import Error
import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

from outputs import read_fromcsv
    
class TrainData():
    '''select the files as the curves'''
    def __init__(self, args):
        self.folder = '../save/Global learning rate/'
        self.dataset = 'cifar10'
        self.path = '{}/{}/'.format(self.folder, self.dataset)
        self.glr = args.glr
        self.llr = args.llr        

    def get_title(self):
        if self.dataset == 'mnist':
            self.title_dataset = 'MNIST'
        elif self.dataset == 'fashionmnist':
            self.title_dataset = 'Fashion-MNIST'
        elif self.dataset == 'cifar10':
            self.title_dataset = 'CIFAR-10'
        if self.glr == None:
            self.pdfname = '{}, 100 clients, 5 classes, llr={}'.format(self.title_dataset, self.llr)
            self.title = '{}, 100 clients, 5 classes, $\eta_l$={}'.format(self.title_dataset, self.llr)
        elif self.llr == None:
            self.pdfname = '{}, 100 clients, 5 classes, glr={}'.format(self.title_dataset, self.glr)
            self.title = '{}, 100 clients, 5 classes, $\eta_g$={}'.format(self.title_dataset, self.glr)
        return self.title, self.pdfname
    
    def get_legend(self):
        return self.sl_legend
    
    def get_xlabel(self, x='round'):
        xlabel_map = {'round': 'Rounds', 'iteration': 'Iterations'}
        return xlabel_map[x]
    
    def get_ylabel(self, t=0):
        ylabel_name = ['Train Loss', 'Test Loss', 'Train Top1 Accuracy', 'Test Top1 Accuracy', 'Train Top5 Accuracy', 'Test Top5 Accuracy']
        return ylabel_name[t]

    def get_raw_files(self):
        self.raw_files = os.listdir(self.path)
        return self.raw_files

    def select_files(self):
        # base fl(100 1 1.0) lenet5 mnist(pat2-b 5.0) sgd(0.01 0.9 0.0001) hp(1.0 10)
        # (alg_setup, dcml_setup, model_setup, dtype_setup, optim_setup, hp_setup)
        raw_files = self.get_raw_files()
        sl_files = []
        if self.glr == None:
            self.choice = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
            self.sl_legend = []
            for i in self.choice:
                if self.dataset == 'cifar10':
                    t = t = 'base sl(100 1 {}) vgg11(4) {}(pat2-b 5.0) sgd({} 0.9 0.0001) hp(10)'.format(self.glr, self.dataset, i)
                else:
                    t = 'base sl(100 1 {}) lenet5(2) {}(pat2-b 5.0) sgd({} 0.9 0.0001) hp(10)'.format(self.glr, self.dataset, i)
                if '{}{}'.format(t, '.csv') in raw_files:
                    sl_files.append(t)
                    self.sl_legend.append('$\eta_g$={}'.format(i))
        elif self.llr == None:
            self.choice = [1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
            self.sl_legend = []
            for i in self.choice:
                if self.dataset == 'cifar10':
                    t = t = 'base sl(100 1 {}) vgg11(4) {}(pat2-b 5.0) sgd({} 0.9 0.0001) hp(10)'.format(self.glr, self.dataset, i)
                else:
                    t = 'base sl(100 1 {}) lenet5(2) {}(pat2-b 5.0) sgd({} 0.9 0.0001) hp(10)'.format(self.glr, self.dataset, i)
                print(t)
                if '{}{}'.format(t, '.csv') in raw_files:
                    sl_files.append(t)
                    self.sl_legend.append('$\eta_l$={}'.format(i))
        return sl_files
    

parser = argparse.ArgumentParser()
parser.add_argument('-t', type=int, default=0, help='test accuracy or train loss')
parser.add_argument('-x', type=str, default=0, choices=['round', 'iteration', 'amount'], help='X-axis')
parser.add_argument('--glr', type=float, default=None, help='global learning rate')
parser.add_argument('--llr', type=float, default=None, help='local learning rate')
parser.add_argument('-s', type=int, default=1, help='start epoch')
parser.add_argument('-e', type=int, default=1, help='end epoch')
args = parser.parse_args()
print(args)   
# python plot_global_lr.py -x "round" -t 0 --glr 1.0
# python plot_global_lr.py -x "round" -t 0 --llr 0.1

def plotcurve(args):
    #mpl.style.use('seaborn')

    tdata = TrainData(args)
    title, pdfname = tdata.get_title()
    xlabel = tdata.get_xlabel(args.x)
    ylabel = tdata.get_ylabel(args.t)
    path = tdata.path
    
    sl_files = tdata.select_files()
    sl_legend = tdata.get_legend()
    
    plt.figure()
    plt.title(title, fontsize=15)
    start_epoch = args.s # 1
    #epochs = range(start_epoch, end_epoch+1)
    lines = []
    
    for i in range(len(sl_files)):
        df = read_fromcsv(sl_files[i], path)
        end_epoch = min(args.e, len(df))
        x_axis = range(1, end_epoch+1)
        plt.plot(x_axis, df.iloc[start_epoch-1:end_epoch, args.t].values, color=None, lw=2)
        lines.append('{}'.format(sl_legend[i]))

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(lines, loc=None, ncol=1, prop={'size': 14}) # loc = 1 or 4
    plt.grid()
    #plt.savefig('{}/{} {}.png'.format(pathplus[0], title, ylabel))
    plt.savefig('{}/{} {}.pdf'.format(tdata.path, pdfname, ylabel), bbox_inches='tight')
    plt.show()

    
if __name__ == '__main__':
    plotcurve(args) 


