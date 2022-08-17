from msilib.schema import Error
import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

from outputs import read_fromcsv

def resolvefilename(filename = 'gld(0.5 u-50-200 1-1) splitfed(1 16 1) vgg11(1) cifar10(iid) hp(0.0001 0.0001 32) server'):
    r'''resolve the file name to get the related information.
    Examples:
        ['gld(0.5 u-50-200 1-1)', 'splitfed(1 16 1)', 'vgg11(1)', 'cifar10(iid)', 'hp(0.0001 0.0001 32)', 'server']
    '''
    pattern = r'[\+\w-]+(?:\([\w \-\.]+\))*'
    a = re.findall(pattern, filename)
    return a
    

capsule = {}

class FileNameCurve():
    '''select the files as the curves'''
    def __init__(self, capsule={}):
        pass

    def get_title(self):
        title = 'MNIST, 100 clients, 5 classes'
        return title
    
    def get_ylabel(self, t=0):
        ylabel_name = ['Train Loss', 'Test Loss', 'Train Top1 Accuracy', 'Test Top1 Accuracy', 'Train Top5 Accuracy', 'Test Top5 Accuracy']
        return ylabel_name[t]

    def get_raw_files(self, folder):
        raw_files = os.listdir(folder)
        return raw_files

    def select_files(self, raw_files):
        # base fl(100 1 1.0) lenet5 mnist(pat2-b 5.0) sgd(0.01 0.9 0.0001) hp(1.0 10)
        # (alg_setup, dcml_setup, model_setup, dtype_setup, optim_setup, hp_setup)
        local_epochs_choice = [1, 2, 5, 10]
        sl_files = []
        for i in local_epochs_choice:
            sl_files.append('base sl(100 {} 1.0) lenet5(2) mnist(pat2-b 5.0) sgd(0.01 0.9 0.0001) hp(10)'.format(i))
        return sl_files
    
    def get_legend(self):
        sl_legend = ['1 local epoch', '2 local epochs', '5 local epochs', '10 local epochs']
        return sl_legend


class Xaxis():
    '''select the files as the curves'''
    def __init__(self):
        pass

    def get_xlabel(self, x):
        xlabel_map = {'round': 'Rounds', 'iteration': 'Iterations'}
        return xlabel_map[x]


parser = argparse.ArgumentParser()
parser.add_argument('-t', type=int, default=0, help='test accuracy or train loss')
parser.add_argument('-x', type=str, default=0, choices=['round', 'iteration', 'amount'], help='X-axis')
parser.add_argument('-s', type=int, default=1, help='start epoch')
args = parser.parse_args()
print(args)
# python plot_local_epoch.py -x "iteration" -t 0  

def plotcurve(args):
    #mpl.style.use('seaborn')

    path = '../save/Choice of K/'

    fncurve = FileNameCurve()
    title = fncurve.get_title()
    ylabel = fncurve.get_ylabel(args.t)
    
    x_axis = Xaxis()
    xlabel = x_axis.get_xlabel(args.x)

    raw_files = fncurve.get_raw_files(path)
    sl_files = fncurve.select_files(raw_files)
    sl_legend = fncurve.get_legend()
    
    local_epochs_choice = [1, 2, 5, 10]
    plt.figure()
    plt.title(title, fontsize=12)
    start_epoch = args.s # 1
    #epochs = range(start_epoch, end_epoch+1)
    lines = []
    
    for i in range(len(sl_files)):
        df = read_fromcsv(sl_files[i], path)
        end_epoch = len(df)
        x_axis = range(1*local_epochs_choice[i]*60, end_epoch*local_epochs_choice[i]*60+1, local_epochs_choice[i]*60)
        plt.plot(x_axis, df.iloc[start_epoch-1:end_epoch, args.t].values, color=None, lw=2, marker='d')
        lines.append('{}'.format(sl_legend[i]))

    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.legend(lines, loc=None, ncol=1, prop={'size': 15}) # loc = 1 or 4
    plt.grid()
    #plt.savefig('{}/{} {}.png'.format(pathplus[0], title, ylabel))
    plt.savefig('{}/{} {}.pdf'.format(path, title, ylabel), bbox_inches='tight')
    plt.show()

    
if __name__ == '__main__':
    plotcurve(args) 


