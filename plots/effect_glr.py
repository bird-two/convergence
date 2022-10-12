'''
Plot FL and SL
STANDARD SIZE: title=15, legend=14, label=14, ticks=12
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import argparse

from outputs import read_fromcsv

class HeatMapData():
    def __init__(self, args):
        self.args = args
        self.dataset = args.d
        self.path = '../save/effect_glr/{}/'.format(self.dataset)
        self.pdfpath = '../save/effect_glr/'
        if self.dataset == 'cifar10':
            if self.args.r == 1:
                self.glr_choice = [1.0, 1.2, 1.4, 1.6, 1.8]
            else:
                self.glr_choice = [1.0, 2.0, 3.0, 4.0, 5.0]
            self.llr_choice = [0.0005, 0.001, 0.005, 0.01, 0.05]
        else:
            if self.args.r == 1:
                self.glr_choice = [1.0, 1.2, 1.4, 1.6, 1.8]
            elif self.args.r == 2:
                self.glr_choice = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
            else:
                self.glr_choice = [1.0, 2.0, 3.0, 4.0, 5.0]
            self.llr_choice = [0.0005, 0.001, 0.005, 0.01, 0.05]
            
    
    def collect_files(self):
        files = {i: [] for i in self.glr_choice[::-1]} # reverse the list glr_choice
        for i in range(len(self.glr_choice)):
            for j in range(len(self.llr_choice)):
                if self.dataset == 'cifar10':
                #t = 'base sl(100 1 {}) lenet5(2) {}(pat2-b 5.0) sgd({} 0.9 0.0001) hp(10)'.format(self.glr_choice[i], self.dataset, self.llr_choice[j])
                    t = 'base sl({} 1 {}) vgg11(4) {}(dir-u 10.0) sgd({} 0.9 0.0001) hp({})'.format(self.args.clients, self.glr_choice[i], self.dataset, self.llr_choice[j], self.args.b)
                else:
                    t = 'base sl({} 1 {}) lenet5(2) {}(dir-u 10.0) sgd({} 0.9 0.0001) hp({})'.format(self.args.clients, self.glr_choice[i], self.dataset, self.llr_choice[j], self.args.b)
                if '{}{}'.format(t, '.csv') in self.raw_files:
                    files[self.glr_choice[i]].append(t)
                else:
                    files[self.glr_choice[i]].append('')
        self.files = files
        return self.files    

    def get_raw_files(self):
        self.raw_files = os.listdir(self.path)
        return self.raw_files

    def gen_heatmapdata(self):
        self.get_raw_files()
        self.collect_files()
        data_point_map = np.zeros((len(self.glr_choice), len(self.llr_choice)))
        for i, key in enumerate(self.files.keys()):
            for j in range(len(self.files[key])):
                if self.files[key][j] == '':
                    data_point_map[i,j] = 0
                else:
                    df = read_fromcsv(self.files[key][j], self.path)
                    if self.dataset == 'cifar10':
                        t = df.iloc[0:30, 3].values # we use 100 epochs for cifar100 50/100
                        data_point = t[-10:].mean(axis=0)
                    elif self.dataset == 'mnist':
                        t = df.iloc[0:30, 3].values # 30/50
                        data_point = t[-10:].mean(axis=0)
                    else:
                        t = df.iloc[0:20, 3].values # 30/50
                        data_point = t[-10:].mean(axis=0)
                    #data_point = t[-10:].var(axis=0)
                    data_point_map[i,j] = data_point
        self.data_point_map = data_point_map
        return data_point_map              

def gen_heatmap(hmdata, row_ticks, col_ticks):
    # this time the data points in the figure are too crowded
    # so we set the figsize (7.2, 5.4) instead of (6.4, 4.8), nice :)
    data = hmdata.data_point_map
    dataset = hmdata.dataset
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(data, cmap = 'Greens')

    # Create colorbar
    cbarlabel = 'Test Top1 Accuracy'
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=14)
    
    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_ticks)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_ticks)

    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Loop over data dimensions and create text annotations.
    textcolors=("black", "white")
    threshold = im.norm(data.max())/2
    for i in range(len(row_ticks)):
        for j in range(len(col_ticks)):
            #color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = ax.text(j, i, '{:.2f}'.format(data[i, j]),
                        ha="center", va="center", color=textcolors[int(im.norm(data[i, j]) > threshold)])

    title_dict = {"mnist":"MNIST", "fashionmnist":"Fashion-MNIST", "cifar10":"CIFAR-10"}
    #title = "MNIST, 100 clients, 5 classes"
    #title = "FashionMNIST, 100 clients, 5 classes"
    title = title_dict[dataset]

    #ax.set_title(title, fontsize=15)
    #plt.ylabel('Global Learning Rate', fontsize=14)
    plt.xlabel('Local Learning Rate ($f(x)=log_{10}(x)$)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('{}/{} {}.pdf'.format(hmdata.pdfpath, title, args.r), bbox_inches='tight')
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default='mnist', help='dataset')
#parser.add_argument('--dist', type=int, default=0, help='distribution')
#parser.add_argument('--dist', type=int, nargs='*', default=[0, 1, 2, 3, 4, 5, 6, 8], help='different non-IID distribution')
parser.add_argument('--clients', type=int, default=10, help='number of clients')
#parser.add_argument('-t', type=int, default=0, help='test accuracy or train loss')
parser.add_argument('-b', type=int, default=10, help='mini-batch')
parser.add_argument('-r', type=int, default=1, help='refine')
#parser.add_argument('-x', type=str, default=0, choices=['round', 'iteration', 'amount'], help='X-axis')
#parser.add_argument('--fl', type=int, default=1, help='Use FL or not')
#parser.add_argument('--sl', type=int, default=1, help='Use SL or not')
#parser.add_argument('-s', type=int, default=1, help='start epoch')
#parser.add_argument('-e', type=int, default=400, help='end epoch')
#parser.add_argument('--lr', type=float, nargs='*', default=[1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], help='learning rate list')
args = parser.parse_args()
print(args)

#python effect_glr.py -d mnist --clients 10 -b 1000 -r 1
#python effect_glr.py -d fashionmnist --clients 10 -b 1000 -r 1
#python effect_glr.py -d cifar10 --clients 10 -b 100 -r 1

def main():
    hmdata = HeatMapData(args)
    hmdata.gen_heatmapdata()
    if hmdata.dataset == 'cifar10':
        #row_ticks = [0.1, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.5, 2.0, 2.5, 3.0, 5.0] # not use
        #row_ticks = [1.0, 2.0, 3.0, 4.0, 5.0]
        row_ticks = hmdata.glr_choice
        #col_ticks = ['5f(-5)','f(-4)','5f(-4)','f(-3)','5f(-3)','f(-2)','5f(-2)']
        col_ticks = ['5f(-4)','f(-3)','5f(-3)','f(-2)','5f(-2)']
    else:
        row_ticks = hmdata.glr_choice
        #col_ticks = [1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
        #col_ticks = ['f(-5)','5f(-5)','f(-4)','5f(-4)','f(-3)','5f(-3)','f(-2)','5f(-2)','f(-1)']
        col_ticks = ['5f(-4)','f(-3)','5f(-3)','f(-2)','5f(-2)']
        #col_ticks = ['$f(-5)$','$5f(-5)$','$f(-4)$','$5f(-4)$','$f(-3)$','$5f(-3)$','$f(-2)$','$5f(-2)$','$f(-1)$']
    gen_heatmap(hmdata, row_ticks[::-1], col_ticks)


if __name__ == '__main__':
    main()