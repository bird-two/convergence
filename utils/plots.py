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
        title = 'MNIST, 100 clients'
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
        shared_items = {
            'alg'  : ['base'],
            'dcml' : [],
            'model': [],
            'dtype': ['mnist(iid-b)', 'mnist(pat2-b 1.0)', 'mnist(pat2-b 2.0)', 'mnist(pat2-b 5.0)'],
            'optim': ['sgd(0.01 0.9 0.0001)'],
            'hp'   : ['hp(10)']
        }
        fl_items = {
            'alg'  : [],
            'dcml' : ['fl(100 1 1.0)'],
            'model': ['lenet5'],
            'dtype': [],
            'optim': [],
            'hp'   : []
        }
        sl_items = {
            'alg'  : [],
            'dcml' : ['sl(100 1 1.0)'],
            'model': ['lenet5(2)'],
            'dtype': [],
            'optim': [],
            'hp'   : []
        }
        fl_names = self.assemble_curve_name(shared_items, fl_items)
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
        sl_legend = ['SL(iid)', 'SL(1)', 'SL(2)', 'SL(5)']
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
parser.add_argument('-s', type=int, default=1, help='start epoch')
parser.add_argument('-e', type=int, default=300, help='end epoch')
#parser.add_argument('--time', type=int, default=0, help='x-axis is time ?')
#parser.add_argument('-f', type=str, nargs='*', default='', help='folder')
#parser.add_argument('--choose', type=str, nargs='*', default=[], help='chooses')
#parser.add_argument('--ban', type=str, nargs='*', default=[], help='ban')
args = parser.parse_args()
print(args)


def getmarker(self, file):
    marker = ['^', 'v', 's', 'o', 'D', 'x', '.', '*', '+', 'd']
    temp = resolvefilename(file)
    if temp[0] == 'base':
        return 's'
    elif temp[0] == 'glocal':
        return 'x'
    elif temp[0] == 'gdelay':
        return '^'
    elif temp[0] == 'replay' or temp[0] == 'grey':
        return 'd'
    elif 'gld' in temp[0]:
        return 'v'
    else:
        return '.'

def getcolor(i):
    color = ['xkcd:red', 'xkcd:blue', 'xkcd:green', 'xkcd:orange', 'xkcd:brown', 'xkcd:violet', 'xkcd:grey', 'xkcd:cyan', 'xkcd:pink']
    return color[i]
    

def plotcurve(args):
    #mpl.style.use('seaborn')

    path = '../save/'

    fncurve = FileNameCurve()
    title = fncurve.get_title()
    ylabel = fncurve.get_ylabel(args.t)
    
    x_axis = Xaxis()
    xlabel = x_axis.get_xlabel(args.x)

    raw_files = fncurve.get_raw_files(path)
    fl_files, sl_files = fncurve.select_files(raw_files)
    fl_legend, sl_legend = fncurve.get_legend()

    plt.figure()
    plt.title(title, fontsize=12)
    start_epoch = args.s # 1
    #epochs = range(start_epoch, end_epoch+1)
    lines = []
    for i in range(len(fl_files)):
        df = read_fromcsv(fl_files[i], path)
        end_epoch = min(args.e, len(df))
        x_axis = range(1, end_epoch+1)
        plt.plot(x_axis, df.iloc[start_epoch-1:end_epoch, args.t].values, linestyle='dashed', color=None, lw=2)
        lines.append('{}'.format(fl_legend[i]))
    
    for i in range(len(sl_files)):
        df = read_fromcsv(sl_files[i], path)
        end_epoch = min(args.e, len(df))
        x_axis = range(1, end_epoch+1)
        plt.plot(x_axis, df.iloc[start_epoch-1:end_epoch, args.t].values, color=None, lw=2)
        lines.append('{}'.format(sl_legend[i]))
            
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.legend(lines, loc=None, ncol=2, prop={'size': 12}) # loc = 1 or 4
    plt.grid()
    #plt.savefig('{}/{} {}.png'.format(pathplus[0], title, ylabel))
    plt.savefig('{}/{} {}.pdf'.format(path, title, ylabel), bbox_inches='tight')
    plt.show()

    
if __name__ == '__main__':
    plotcurve(args) 


