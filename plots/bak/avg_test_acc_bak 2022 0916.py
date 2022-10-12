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


mnist_temps = [
    'base fl({} 1 1.0) lenet5 {}(iid-b) {}{} hp({})',      'base sl({} 1 1.0) lenet5(2) {}(iid-b) {}{} hp({})',
    'base fl({} 1 1.0) lenet5 {}(pat2-b 5.0) {}{} hp({})', 'base sl({} 1 1.0) lenet5(2) {}(pat2-b 5.0) {}{} hp({})',
    'base fl({} 1 1.0) lenet5 {}(pat2-b 2.0) {}{} hp({})', 'base sl({} 1 1.0) lenet5(2) {}(pat2-b 2.0) {}{} hp({})',
    'base fl({} 1 1.0) lenet5 {}(pat2-b 1.0) {}{} hp({})', 'base sl({} 1 1.0) lenet5(2) {}(pat2-b 1.0) {}{} hp({})',]

cifar10_temps = [
    'base flv2({} 1 1.0) vgg11 {}(iid-b) {}{} hp({})',      'base sl({} 1 1.0) vgg11(4) {}(iid-b) {}{} hp({})',
    'base flv2({} 1 1.0) vgg11 {}(pat2-b 5.0) {}{} hp({})', 'base sl({} 1 1.0) vgg11(4) {}(pat2-b 5.0) {}{} hp({})',
    'base flv2({} 1 1.0) vgg11 {}(pat2-b 2.0) {}{} hp({})', 'base sl({} 1 1.0) vgg11(4) {}(pat2-b 2.0) {}{} hp({})',
    'base flv2({} 1 1.0) vgg11 {}(pat2-b 1.0) {}{} hp({})', 'base sl({} 1 1.0) vgg11(4) {}(pat2-b 1.0) {}{} hp({})',]

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default=0, help='dataset')
parser.add_argument('-b', type=int, default=10, help='minibatch')
parser.add_argument('--alg', type=str, default='sgd', help='sgd, adam')
parser.add_argument('--clients', type=int, default=10, help='number of clients')
args = parser.parse_args()
print(args)

# python avg_test_acc.py -d fashionmnist -b 10 --alg sgd --clients 10

class HeatMapData():
    def __init__(self, args):
        self.args = args
        self.dataset = args.d
        self.path = '../save/cmp_lr/{}/'.format(self.dataset)
        epochs_dict = {'mnist':50, 'fashionmnist':50, 'cifar10':400, 'cifar100': 400}
        select_epochs_dict = {50:10, 400:40}
        self.epochs = epochs_dict[self.dataset]
        self.select_epochs = select_epochs_dict[self.epochs]
        if self.dataset == 'mnist' or self.dataset == 'fashionmnist':
            self.temps = mnist_temps
        elif self.dataset == 'cifar10' or self.dataset == 'cifar100':
            self.temps = cifar10_temps
        self.llr_choice = [1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
        #self.col_ticks = [1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
        self.col_ticks = ['f(-5)','5f(-5)','f(-4)','5f(-4)','f(-3)','5f(-3)','f(-2)','5f(-2)','f(-1)']
            
    def collect_files(self):
        files = {i: [] for i in range(len(self.temps))}
        for i in range(len(self.temps)):
            for j in range(len(self.llr_choice)):
                if self.args.alg == 'sgd':
                    t = self.temps[i].format(self.args.clients, self.dataset, self.args.alg, '({} 0.9 0.0001)'.format(self.llr_choice[j]), self.args.b)
                elif self.args.alg == 'adam':
                    t = self.temps[i].format(self.args.clients, self.dataset, self.args.alg, '({} 0.0001)'.format(self.llr_choice[j]), self.args.b)
                if '{}{}'.format(t, '.csv') in self.raw_files:
                    files[i].append(t)
                else:
                    files[i].append('')
        self.files = files
        #return self.files    

    def get_raw_files(self):
        self.raw_files = os.listdir(self.path)
        #return self.raw_files

    def gen_heatmapdata(self):
        self.get_raw_files()
        self.collect_files()
        data_point_map = np.zeros((len(self.temps), len(self.llr_choice)))
        for i, key in enumerate(self.files.keys()):
            for j in range(len(self.files[key])):
                if self.files[key][j] == '':
                    data_point_map[i,j] = 0
                else:
                    df = read_fromcsv(self.files[key][j], self.path)
                    t = df.iloc[0:self.epochs, 3].values # we use 100 epochs for cifar100
                    data_point = t[-self.select_epochs:].mean(axis=0)
                    #data_point = t[-10:].var(axis=0)
                    data_point_map[i,j] = data_point
        self.data_point_map = data_point_map
        self.print_data2()  

    def print_data(self):
        for i in range(len(self.data_point_map)):
            for j in range(len(self.llr_choice)):
                print('&{:.2f}'.format(self.data_point_map[i,j]), end=' ') if self.data_point_map[i,j] != 0 else print('&-', end=' ')
            print('')
    
    def print_data2(self):
        if self.dataset == 'mnist':
            a = ['L-M &{} &IID &{} &{}'.format(self.args.clients, self.args.alg, self.args.b), 
                 'L-M &{} &5 &{} &{}'.format(self.args.clients, self.args.alg, self.args.b), 
                 'L-M &{} &2 &{} &{}'.format(self.args.clients, self.args.alg, self.args.b), 
                 'L-M &{} &1 &{} &{}'.format(self.args.clients, self.args.alg, self.args.b)]
        elif self.dataset == 'fashionmnist':
            a = ['L-F &{} &IID &{} &{}'.format(self.args.clients, self.args.alg, self.args.b), 
                 'L-F &{} &5 &{} &{}'.format(self.args.clients, self.args.alg, self.args.b), 
                 'L-F &{} &2 &{} &{}'.format(self.args.clients, self.args.alg, self.args.b), 
                 'L-F &{} &1 &{} &{}'.format(self.args.clients, self.args.alg, self.args.b)]
        elif self.dataset == 'cifar10':
            a = ['V-10 &{} &IID &{} &{}'.format(self.args.clients, self.args.alg, self.args.b), 
                 'V-10 &{} &5 &{} &{}'.format(self.args.clients, self.args.alg, self.args.b), 
                 'V-10 &{} &2 &{} &{}'.format(self.args.clients, self.args.alg, self.args.b), 
                 'V-10 &{} &1 &{} &{}'.format(self.args.clients, self.args.alg, self.args.b)]
        elif self.dataset == 'cifar100':
            a = ['V-100 &{} &IID &{} &{}'.format(self.args.clients, self.args.alg, self.args.b), 
                 'V-100 &{} &5 &{} &{}'.format(self.args.clients, self.args.alg, self.args.b), 
                 'V-100 &{} &2 &{} &{}'.format(self.args.clients, self.args.alg, self.args.b), 
                 'V-100 &{} &1 &{} &{}'.format(self.args.clients, self.args.alg, self.args.b)]
        for i in range(len(self.temps)//2):
            print(a[i], end=' ')
            if self.dataset == 'cifar100':
                min0 = np.where((self.data_point_map[2*i+0]>0.0)&(self.data_point_map[2*i+0]<2.0))[0]
                min1 = np.where((self.data_point_map[2*i+1]>0.0)&(self.data_point_map[2*i+1]<2.0))[0]
            else:
                min0 = np.where((self.data_point_map[2*i+0]>0.0)&(self.data_point_map[2*i+0]<12.0))[0]
                min1 = np.where((self.data_point_map[2*i+1]>0.0)&(self.data_point_map[2*i+1]<12.0))[0]
            min0 = 300 if len(min0)==0 else min0[0]
            min1 = 300 if len(min1)==0 else min1[0]
            #print(min0, min1)

            best0 = np.argmax(self.data_point_map[2*i+0], axis=0)
            best1 = np.argmax(self.data_point_map[2*i+1], axis=0)
            
            #print(max0, max1)
            for j in range(len(self.llr_choice)):
                # if best0 == min0 here use best0
                if j == best0:
                    format0 = '&\\tcb{{{:.1f}}}'
                elif j == min0:
                    format0 = '&\\ulb{{{:.1f}}}'
                else:
                    format0 = '&{:.1f}'
                
                if j == best1:
                    format1 = '&\\tcr{{{:.1f}}}'
                elif j == min1:
                    format1 = '&\\ulr{{{:.1f}}}'
                else:
                    format1 = '&{:.1f}'
                print(format0.format(self.data_point_map[2*i+0,j]), end=' ') if self.data_point_map[2*i+0,j] != 0.0 else print('&-', end=' ')
                print(format1.format(self.data_point_map[2*i+1,j]), end=' ') if self.data_point_map[2*i+1,j] != 0.0 else print('&-', end=' ')
            print('\\\\')

def gen_heatmap(heatmapdata):
    # this time the data points in the figure are too crowded
    # so we set the figsize (7.2, 5.4) instead of (6.4, 4.8), nice :)
    data = heatmapdata.data_point_map
    col_ticks = heatmapdata.col_ticks
    row_ticks = ['FL(iid)', 'SL(iid)', 'FL(5)', 'SL(5)', 'FL(2)', 'SL(2)', 'FL(1)', 'SL(1)']
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap = 'Greens')

    # Create colorbar
    cbarlabel = 'Test Top1 Accuracy'
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=14)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_ticks)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_ticks)

    # Loop over data dimensions and create text annotations.
    textcolors=("black", "white")
    threshold = im.norm(data.max())/2
    for i in range(len(row_ticks)):
        for j in range(len(col_ticks)):
            #color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = ax.text(j, i, '{:.2f}'.format(data[i, j]),
                        ha="center", va="center", color=textcolors[int(im.norm(data[i, j]) > threshold)])

    #title_dict = {"mnist":"MNIST, 100 clients", "fashionmnist":"FashionMNIST, 100 clients, 5 cla", "cifar10":"CIFAR-10, 100 clients, 5 classes"}
    #title = "MNIST, 100 clients, 5 classes"
    #title = "FashionMNIST, 100 clients, 5 classes"
    #title = title_dict[heatmapdata.dataset]
    title = heatmapdata.temps[1][5:]
    print(title)
    ax.set_title(title)
    #plt.ylabel('Global Learning Rate', fontsize=14)
    plt.xlabel('Local Learning Rate ($f(x)=log_{10}(x)$)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('{}/{}.pdf'.format('../save/cmp_lr/', title), bbox_inches='tight')
    plt.show()

def main():
    heatmapdata = HeatMapData(args=args)
    heatmapdata.gen_heatmapdata()
    #col_ticks = ['$f(-5)$','$5f(-5)$','$f(-4)$','$5f(-4)$','$f(-3)$','$5f(-3)$','$f(-2)$','$5f(-2)$','$f(-1)$']
    gen_heatmap(heatmapdata)


if __name__ == '__main__':
    main()