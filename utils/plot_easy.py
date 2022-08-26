from msilib.schema import Error
import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

from outputs import read_fromcsv


class FileNameCurve():
    '''select the files as the curves'''
    def __init__(self, capsule={}):
        pass
    
    def get_ylabel(self, t=0):
        ylabel_name = ['Train Loss', 'Test Loss', 'Train Top1 Accuracy', 'Test Top1 Accuracy', 'Train Top5 Accuracy', 'Test Top5 Accuracy']
        return ylabel_name[t]

    def get_raw_files(self, folder):
        all_files = os.listdir(folder)
        raw_files = []
        for i in all_files:
            if '.csv' in i:
                raw_files.append(i[0:-4])
        return raw_files


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
parser.add_argument('-r', type=int, default=0, help='all the raw files or not')
parser.add_argument('-s', type=int, default=1, help='start epoch')
parser.add_argument('-e', type=int, default=400, help='end epoch')
#parser.add_argument('--time', type=int, default=0, help='x-axis is time ?')
#parser.add_argument('-f', type=str, nargs='*', default='', help='folder')
#parser.add_argument('--choose', type=str, nargs='*', default=[], help='chooses')
#parser.add_argument('--ban', type=str, nargs='*', default=[], help='ban')
args = parser.parse_args()
print(args)


def plotcurve(args):
    #mpl.style.use('seaborn')

    path = 'D:\\study\\birds\\Convegence Analysis of SL and FL\\EXP\\save\\Global learning rate\\'
    #path = 'D:\\download\\'

    fncurve = FileNameCurve()
    title = 'test'
    ylabel = fncurve.get_ylabel(args.t)
    
    x_axis = Xaxis()
    xlabel = x_axis.get_xlabel(args.x)

    raw_files = fncurve.get_raw_files(path)
    if args.r == 1:
        files = raw_files
    else:
        #files = ['base flv2(100 1 1.0) vgg11 cifar10(iid-b) sgd(0.0001 0.9 0.0001) hp(10)','base flv2(100 1 1.0) vgg11 cifar10(iid-b) sgd(0.01 0.9 0.0001) hp(10)','base flv2(100 1 1.0) vgg11 cifar10(iid-b) sgd(0.001 0.9 0.0001) hp(10)'] # please write the files by yourself.
        files = ['base sl(100 1 0.1) lenet5(2) mnist(pat2-b 5.0) sgd(0.001 0.9 0.0001) hp(10)',
                 'base sl(100 1 1.0) lenet5(2) mnist(pat2-b 5.0) sgd(0.001 0.9 0.0001) hp(10)',
                 'base sl(100 1 1.5) lenet5(2) mnist(pat2-b 5.0) sgd(0.001 0.9 0.0001) hp(10)',
                 'base sl(100 1 2.0) lenet5(2) mnist(pat2-b 5.0) sgd(0.001 0.9 0.0001) hp(10)',
                 'base sl(100 1 5.0) lenet5(2) mnist(pat2-b 5.0) sgd(0.001 0.9 0.0001) hp(10)',
                 'base sl(100 1 10.0) lenet5(2) mnist(pat2-b 5.0) sgd(0.001 0.9 0.0001) hp(10)',
                 #'base sl(100 1 1.0) vgg11(4) cifar10(iid-b) sgd(0.0001 0.9 0.0001) hp(10)',
                 #'base sl(100 1 1.0) vgg11(4) cifar10(iid-b) sgd(0.001 0.9 0.0001) hp(10)',
                 #'base sl(100 1 1.0) vgg11(4) cifar10(pat2-b 5.0) sgd(0.001 0.9 0.0001) hp(10)',
                 #'base sl(100 1 1.0) vgg11(4) cifar10(pat2-b 5.0) sgd(0.001 0.9 0.0001) hp(10)',
                 #'base sl(100 1 1.0) vgg11(4) cifar10(iid-b) sgd(0.005 0.9 0.0001) hp(10)',
                 #'base sl(100 1 1.0) vgg11(4) cifar10(pat2-b 5.0) sgd(0.005 0.9 0.0001) hp(10)',
                 #'base sl(100 1 1.0) vgg11(4) cifar10(iid-b) sgd(0.01 0.9 0.0001) hp(10)',
                 #'base sl(100 1 1.0) vgg11(4) cifar10(iid-b) sgd(0.01 0.9 0.0001) hp(10)',
                 #'base flv2(100 1 1.0) vgg11 cifar10(iid-b) sgd(0.0001 0.9 0.0001) hp(10)',
                 #'base flv2(100 1 1.0) vgg11 cifar10(iid-b) sgd(0.001 0.9 0.0001) hp(10)',
                 #'base flv2(100 1 1.0) vgg11 cifar10(pat2-b 5.0) sgd(0.005 0.9 0.0001) hp(10)',
                 #'base flv2_2(100 1 1.0) vgg11 cifar10(pat2-b 5.0) sgd(0.005 0.9 0.0001) hp(10)',
                 #'base flv2(100 1 1.0) vgg11 cifar10(iid-b) sgd(0.005 0.9 0.0001) hp(10)',
                 #'base flv2_2(100 1 1.0) vgg11 cifar10(iid-b) sgd(0.005 0.9 0.0001) hp(10)',
                 #'base flv2(100 1 1.0) vgg11 cifar10(iid-b) sgd(0.01 0.9 0.0001) hp(10)', 
                 #'base flv2(100 1 1.0) vgg11 cifar10(iid-b) sgd(0.015 0.9 0.0001) hp(10)',
                 #'base flv2(100 1 1.0) vgg11 cifar10(iid-b) sgd(0.02 0.9 0.0001) hp(10)',
                 #'base flv2(100 1 1.0) vgg11 cifar10(iid-b) sgd(0.025 0.9 0.0001) hp(10)',
                 #'base flv2(100 1 1.0) vgg11 cifar10(iid-b) sgd(0.03 0.9 0.0001) hp(10)',
                 #'base flv2(100 1 1.0) vgg11 cifar10(iid-b) sgd(0.05 0.9 0.0001) hp(10)',
                 #'base flv2(100 1 1.0) vgg11 cifar10(iid-b) sgd(0.1 0.9 0.0001) hp(10)',
                 #'base flv2(100 1 1.0) vgg11 cifar10(iid-b) adam(0.001 0.0001) hp(10)',
                 ]

    plt.figure()
    plt.title(title, fontsize=12)
    start_epoch = args.s # 1
    #epochs = range(start_epoch, end_epoch+1)
    lines = []
    for i in range(len(files)):
        print(files[i])
        df = read_fromcsv(files[i], path)
        end_epoch = min(args.e, len(df))
        x_axis = range(1, end_epoch+1)
        plt.plot(x_axis, df.iloc[start_epoch-1:end_epoch, args.t].values, color=None, lw=2, linestyle='--')
        lines.append('{}'.format(files[i]))
            
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.legend(lines, loc=None, ncol=1) # loc = 1 or 4, prop={'size': 12}
    plt.grid()
    #plt.savefig('{}/{} {}.png'.format(pathplus[0], title, ylabel))
    plt.savefig('{}/{} {}.pdf'.format(path, title, ylabel), bbox_inches='tight')
    plt.show()

    
if __name__ == '__main__':
    plotcurve(args) 


