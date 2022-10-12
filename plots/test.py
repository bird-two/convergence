import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import argparse
import seaborn as sns

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
            else:
                self.glr_choice = [1.0, 2.0, 3.0, 4.0, 5.0]
            self.llr_choice = [0.0005, 0.001, 0.005, 0.01, 0.05]
            
    
    def collect_files(self, glr_choice=None, llr_choice=None):
        if glr_choice == None:
            glr_choice = self.glr_choice 
        elif llr_choice == None:
            llr_choice = self.llr_choice
        files = {i: [] for i in glr_choice[::-1]} # reverse the list glr_choice
        for i in range(len(glr_choice)):
            for j in range(len(llr_choice)):
                if self.dataset == 'cifar10':
                #t = 'base sl(100 1 {}) lenet5(2) {}(pat2-b 5.0) sgd({} 0.9 0.0001) hp(10)'.format(self.glr_choice[i], self.dataset, self.llr_choice[j])
                    t = 'base sl({} 1 {}) vgg11(4) {}(dir-u 10.0) sgd({} 0.9 0.0001) hp({})'.format(self.args.clients, glr_choice[i], self.dataset, llr_choice[j], self.args.b)
                else:
                    t = 'base sl({} 1 {}) lenet5(2) {}(dir-u 10.0) sgd({} 0.9 0.0001) hp({})'.format(self.args.clients, glr_choice[i], self.dataset, llr_choice[j], self.args.b)
                if '{}{}'.format(t, '.csv') in self.raw_files:
                    files[glr_choice[i]].append(t)
                else:
                    files[glr_choice[i]].append('')
        return files

    def get_raw_files(self):
        self.raw_files = os.listdir(self.path)
        return self.raw_files

    def gen_heatmapdata(self, files=None, glr_choice=None, llr_choice=None):
        self.get_raw_files()
        if files == None:
            files = self.collect_files()
            glr_choice = self.glr_choice
            llr_choice = self.llr_choice
        data_point_map = np.zeros((len(glr_choice), len(llr_choice)))
        for i, key in enumerate(files.keys()):
            for j in range(len(files[key])):
                if files[key][j] == '':
                    data_point_map[i,j] = 0
                else:
                    df = read_fromcsv(files[key][j], self.path)
                    if self.dataset == 'cifar10':
                        t = df.iloc[0:100, 3].values # we use 100 epochs for cifar100 50/100
                        data_point = t[-20:].mean(axis=0)
                    elif self.dataset == 'mnist':
                        t = df.iloc[0:30, 3].values # 30/50
                        data_point = t[-10:].mean(axis=0)
                    else:
                        t = df.iloc[0:30, 3].values # 30/50
                        data_point = t[-10:].mean(axis=0)
                    #data_point = t[-10:].var(axis=0)
                    data_point_map[i,j] = data_point
        self.data_point_map = data_point_map
        return data_point_map 

    def gen_two(self):
        self.get_raw_files()
        files1 = self.collect_files(glr_choice=[1.0, 2.0, 3.0, 4.0, 5.0], llr_choice=[0.0005, 0.001, 0.005, 0.01, 0.05])
        data1 =  self.gen_heatmapdata(files=files1, glr_choice=[1.0, 2.0, 3.0, 4.0, 5.0], llr_choice=[0.0005, 0.001, 0.005, 0.01, 0.05])

        files2 = self.collect_files(glr_choice=[1.0, 1.2, 1.4, 1.6, 1.8], llr_choice=[0.0005, 0.001, 0.005, 0.01, 0.05])
        data2 =  self.gen_heatmapdata(files=files2, glr_choice=[1.0, 1.2, 1.4, 1.6, 1.8], llr_choice=[0.0005, 0.001, 0.005, 0.01, 0.05]) 

        return data1, data2


def plot():
    
    hmdata = HeatMapData(args)
    datas = hmdata.gen_two()
    row_ticks1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    row_ticks2 = [1.0, 1.2, 1.4, 1.6, 1.8]

    #datas[0].pivot("month", "year", "passengers")
    

    plt.rcParams["figure.figsize"] = [8, 3.6]
    plt.rcParams["figure.autolayout"] = True

    fig, axs = plt.subplots(ncols=2, gridspec_kw=dict(width_ratios=[4, 4]))
    
    #plt.ylabel("Y-Axis")
    #fig.subplots_adjust(wspace=0.01)

    s1 = sns.heatmap(datas[0], cmap="Greens", 
                    ax=axs[0], 
                    xticklabels=[0.0005, 0.001, 0.005, 0.01, 0.05],
                    yticklabels=row_ticks1[::-1], 
                    linewidths=.8, 
                    annot=True,
                    annot_kws={
                'fontsize': 12,
            },
                    fmt=".2f", 
                    cbar=False)
    s2 = sns.heatmap(datas[1], cmap="Greens", 
                    ax=axs[1], 
                    xticklabels=[0.0005, 0.001, 0.005, 0.01, 0.05],
                    yticklabels=row_ticks2[::-1], 
                    linewidths=.8, 
                    annot=True, 
                    annot_kws={
                'fontsize': 12,
            },
                    fmt=".2f",
                    cbar=False)

    #fig.colorbar(axs[1].collections[0], cax=axs[2])
    s1.set_xlabel('Local Learning Rate', fontsize=12)
    s2.set_xlabel('Local Learning Rate', fontsize=12)
    s1.set_ylabel('Global Learning Rate', fontsize=12)
    
    #fig.supxlabel('Local Learning Rate ($f(x)=log_{10}(x)$)', fontsize=14)
    #fig.supylabel('Global Learning Rate', fontsize=14)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.001)
    plt.savefig('{}/{}.pdf'.format(hmdata.pdfpath, 'MNIST'), bbox_inches='tight')
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
    plot()

if __name__ == "__main__":
    main()
