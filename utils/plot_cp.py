'''
Plot FL and SL
SIZE: title=15, legend=14, label=14, ticks=12
'''
from msilib.schema import Error
import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

from outputs import read_fromcsv

mnist_capsule = {
            'alg'  : ['base'],
            'dcml' : [],
            'model': [],
            'dtype': ['mnist(iid-b)', 'mnist(pat2-b 1.0)', 'mnist(pat2-b 2.0)', 'mnist(pat2-b 5.0)'],
            'optim': ['sgd(0.01 0.9 0.0001)'],
            'hp'   : ['hp(10)']
        }
# llr = 0.01
# fashion_capsule = {
#             'alg'  : ['base'],
#             'dcml' : [],
#             'model': [],
#             'dtype': ['fashionmnist(iid-b)', 'fashionmnist(pat2-b 1.0)', 'fashionmnist(pat2-b 2.0)', 'fashionmnist(pat2-b 5.0)'],
#             'optim': ['sgd({} 0.9 0.0001)'.format(self.llr)],
#             'hp'   : ['hp(10)']
#         }
cifar10_capsule = {
            'alg'  : ['base'],
            'dcml' : [],
            'model': [],
            'dtype': ['cifar10(iid-b)', 'cifar10(pat2-b 1.0)', 'cifar10(pat2-b 2.0)', 'cifar10(pat2-b 5.0)'],
            'optim': ['sgd(0.005 0.9 0.0001)'],
            'hp'   : ['hp(10)']
        }

class FileNameCurve():
    '''select the files as the curves'''
    def __init__(self, args, capsule={}):
        self.args = args
        self.capsule = capsule

    def get_title(self, dataset, num_clients, llr):
        self.num_clients = num_clients
        self.llr = llr
        if dataset == 'mnist':
            title = 'MNIST, {} clients'.format(num_clients)
        elif dataset == 'fashionmnist':
            title = 'FashionMNIST, {} clients'.format(num_clients)
        elif dataset == 'cifar10':
            title = 'CIFAR10, {} clients'.format(num_clients)
        return title
    
    def get_ylabel(self, t=0):
        ylabel_name = ['Train Loss', 'Test Loss', 'Train Top1 Accuracy', 'Test Top1 Accuracy', 'Train Top5 Accuracy', 'Test Top5 Accuracy']
        return ylabel_name[t]

    def get_raw_files(self, folder):
        raw_files = os.listdir(folder)
        return raw_files

    def select_files(self, raw_files):
        shared_items = {
            'alg'  : ['base'],
            'dcml' : [],
            'model': [],
            'dtype': ['cifar10(iid-b)', 'cifar10(pat2-b 1.0)', 'cifar10(pat2-b 2.0)', 'cifar10(pat2-b 5.0)', 'cifar10(pat2-b 8.0)'],
            'optim': ['sgd({} 0.9 0.0001)'.format(self.llr)],
            'hp'   : ['hp(10)']
        }
        fl_other_items = {
            'alg'  : ['base'],
            'dcml' : [],
            'model': [],
            'dtype': ['cifar10(iid-b)', 'cifar10(pat2-b 1.0)', 'cifar10(pat2-b 2.0)', 'cifar10(pat2-b 5.0)', 'cifar10(pat2-b 8.0)'],
            'optim': ['sgd(0.1 0.9 0.0001)'],
            'hp'   : ['hp(10)']
        }
        fl_items = {
            'alg'  : [],
            'dcml' : ['flv2({} 1 1.0)'.format(self.num_clients)],
            'model': ['vgg11'], 
            'dtype': [],
            'optim': [],
            'hp'   : []
        }
        sl_items = {
            'alg'  : [],
            'dcml' : ['sl({} 1 1.0)'.format(self.num_clients)],
            'model': ['vgg11(4)'],#lenet5(2)
            'dtype': [],
            'optim': [],
            'hp'   : []
        }
        fl_names = self.assemble_curve_name(shared_items, fl_items)
        if self.args.o != None:
            fl_other_names = self.assemble_curve_name(fl_other_items, fl_items)
            fl_names= fl_names+fl_other_names
        sl_names = self.assemble_curve_name(shared_items, sl_items)
        
        fl_exist = []
        for i in fl_names:
            if '{}{}'.format(i, '.csv') in raw_files:
                fl_exist.append(i)
        
        sl_exist = []
        for i in sl_names:
            if '{}{}'.format(i, '.csv') in raw_files:
                sl_exist.append(i)
        
        return fl_exist, sl_exist
    
    def assemble_curve_name(self, shared, personal):
        '''personal: sl or fl'''
        assemble = {
            'alg'  : [],
            'dcml' : [],
            'model': [],
            'dtype': [],
            'optim': [],
            'hp'   : []
        }
        varying_key = 'alg'
        for key in shared.keys():
            if len(shared[key]) == 0:
                assemble[key].extend(personal[key])
            else:
                assemble[key].extend(shared[key])
            if len(assemble[key]) > 1:
                varying_key = key
            
        curve_names = [] 
        t = []
        for i in range(len(assemble[varying_key])):
            t = []
            for key in assemble.keys():
                if key == varying_key:
                    t.append(assemble[key][i])
                else:
                    t.append(assemble[key][0])
            curve_names.append(' '.join(t))
        return curve_names
    
    def get_legend(self):
        fl_legend = ['FL(iid)', 'FL(1)', 'FL(2)', 'FL(5)']
        #fl_legend = ['FL(iid)', 'FL(1)', 'FL(2)', 'FL(5)', 'FL(8)']
        fl_other_legend = ['FL(iid)-', 'FL(1)-', 'FL(2)-', 'FL(5)-']
        fl_legend = fl_legend + fl_other_legend
        sl_legend = ['SL(iid)', 'SL(1)', 'SL(2)', 'SL(5)']
        #sl_legend = ['SL(iid)', 'SL(1)', 'SL(2)', 'SL(5)', 'SL(8)']
        return fl_legend, sl_legend


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
parser.add_argument('-f', type=int, default=0, help='Use FL or not')
parser.add_argument('--sl', type=int, default=0, help='Use SL or not')
parser.add_argument('-s', type=int, default=1, help='start epoch')
parser.add_argument('-e', type=int, default=400, help='end epoch')
parser.add_argument('-o', type=int, default=None, help='others')
#parser.add_argument('--time', type=int, default=0, help='x-axis is time ?')
#parser.add_argument('-f', type=str, nargs='*', default='', help='folder')
#parser.add_argument('--choose', type=str, nargs='*', default=[], help='chooses')
#parser.add_argument('--ban', type=str, nargs='*', default=[], help='ban')
args = parser.parse_args()
print(args)
# python plot_cp.py -x "round" -t 3 -f 1 -e 100

def plotcurve(args):
    #mpl.style.use('seaborn')
    dataset = 'cifar10' # mnist, cifar10
    num_clients = 1000
    llr = 0.005
    path = '../save/SL and FL/{}/'.format(dataset)

    fncurve = FileNameCurve(args=args)
    title = fncurve.get_title(dataset, num_clients, llr)
    ylabel = fncurve.get_ylabel(args.t)
    
    x_axis = Xaxis()
    xlabel = x_axis.get_xlabel(args.x)

    raw_files = fncurve.get_raw_files(path)
    fl_files, sl_files = fncurve.select_files(raw_files)
    fl_legend, sl_legend = fncurve.get_legend()

    plt.figure()
    start_epoch = args.s # 1
    #epochs = range(start_epoch, end_epoch+1)
    lines = []
    if args.f == 1:
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
    
    plt.title(title, fontsize=15)        
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    #plt.ylim(ymin=0, ymax=3.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ncol = 2 if args.f == 1 and args.sl == 1 else 1
    plt.legend(lines, loc=None, ncol= ncol, prop={'size': 14}) # loc = 1 or 4
    plt.grid()
    #plt.savefig('{}/{} {}.png'.format(pathplus[0], title, ylabel))
    plt.savefig('{}/{} {}.pdf'.format(path, title, ylabel), bbox_inches='tight')
    plt.show()

    
if __name__ == '__main__':
    plotcurve(args) 


