'''
Plot FL and SL
STANDARD SIZE: title=15, legend=14, label=14, ticks=12
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from outputs import read_fromcsv

class HeatMapData():
    def __init__(self):
        self.dataset = 'fashionmnist'
        self.path = '../save/Global learning rate/{}/'.format(self.dataset)
        if self.dataset == 'cifar10':
            self.glr_choice = [0.1, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 5.0]
            self.llr_choice = [1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005]
        else:
            self.glr_choice = [0.1, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]
            self.llr_choice = [1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
            
    
    def collect_files(self):
        files = {i: [] for i in self.glr_choice[::-1]} # reverse the list glr_choice
        for i in range(len(self.glr_choice)):
            for j in range(len(self.llr_choice)):
                if self.dataset == 'cifar10':
                #t = 'base sl(100 1 {}) lenet5(2) {}(pat2-b 5.0) sgd({} 0.9 0.0001) hp(10)'.format(self.glr_choice[i], self.dataset, self.llr_choice[j])
                    t = 'base sl(100 1 {}) vgg11(4) {}(pat2-b 5.0) sgd({} 0.9 0.0001) hp(10)'.format(self.glr_choice[i], self.dataset, self.llr_choice[j])
                else:
                    t = 'base sl(100 1 {}) lenet5(2) {}(pat2-b 5.0) sgd({} 0.9 0.0001) hp(10)'.format(self.glr_choice[i], self.dataset, self.llr_choice[j])
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
                        t = df.iloc[0:100, 3].values # we use 100 epochs for cifar100
                        data_point = t[-50:].mean(axis=0)
                    else:
                        t = df.iloc[0:50, 3].values
                        data_point = t[-30:].mean(axis=0)
                    #data_point = t[-10:].var(axis=0)
                    data_point_map[i,j] = data_point
        return data_point_map              

def gen_heatmap(data, dataset, row_ticks, col_ticks):
    # this time the data points in the figure are too crowded
    # so we set the figsize (7.2, 5.4) instead of (6.4, 4.8), nice :)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
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

    title_dict = {"mnist":"MNIST, 100 clients, 5 classes", "fashionmnist":"FashionMNIST, 100 clients, 5 classes", "cifar10":"CIFAR-10, 100 clients, 5 classes"}
    #title = "MNIST, 100 clients, 5 classes"
    #title = "FashionMNIST, 100 clients, 5 classes"
    title = title_dict[dataset]

    ax.set_title(title, fontsize=15)
    plt.ylabel('Global Learning Rate', fontsize=14)
    plt.xlabel('Local Learning Rate ($f(x)=log_{10}(x)$)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('{}/{}.pdf'.format('../save/Global learning rate/', title), bbox_inches='tight')
    plt.show()

def main():
    heatmapdata = HeatMapData()
    dataset = heatmapdata.dataset
    data_point_map = heatmapdata.gen_heatmapdata()
    if dataset == 'cifar10':
        #row_ticks = [0.1, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.5, 2.0, 2.5, 3.0, 5.0] # not use
        row_ticks = [0.1, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 5.0]
        col_ticks = ['f(-5)','5f(-5)','f(-4)','5f(-4)','f(-3)','5f(-3)']
    else:
        row_ticks = [0.1, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]
        #col_ticks = [1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
        col_ticks = ['f(-5)','5f(-5)','f(-4)','5f(-4)','f(-3)','5f(-3)','f(-2)','5f(-2)','f(-1)']
        #col_ticks = ['$f(-5)$','$5f(-5)$','$f(-4)$','$5f(-4)$','$f(-3)$','$5f(-3)$','$f(-2)$','$5f(-2)$','$f(-1)$']
    gen_heatmap(data_point_map, dataset, row_ticks[::-1], col_ticks)


if __name__ == '__main__':
    main()