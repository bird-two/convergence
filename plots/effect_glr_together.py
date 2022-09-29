'''
Plot FL and SL
STANDARD SIZE: title=15, legend=14, label=14, ticks=12
https://blog.csdn.net/u011699626/article/details/108477880
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import argparse
from mpl_toolkits.axes_grid1 import ImageGrid

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

class Visualazation:
    def mainProgram(self):
        # Set up figure and image grid
        fig = plt.figure(figsize=(8, 4))
        
        grid = ImageGrid(fig, 111,
                          nrows_ncols=(1,2),
                          axes_pad=0.5,
                          share_all=False,
                          label_mode='all',
                          cbar_location="right",
                          cbar_mode="single",
                          cbar_size="7%",
                          cbar_pad=0.15,
                         )
        
        col_ticks = ['5f(-4)','f(-3)','5f(-3)','f(-2)','5f(-2)']
        #row_ticks_list = [[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 1.2, 1.4, 1.6, 1.8]]
        row_ticks_list = [[1.0, 1.2, 1.4, 1.6, 1.8], [1.0, 2.0, 3.0, 4.0, 5.0]]
        
        hmdata = HeatMapData(args)
        datas = hmdata.gen_two()

        # Add data to image grid
        for i, ax in enumerate(grid):
            #data = np.random.random((5,5))
            data = datas[i]
            #row_ticks = 
            #print(row_ticks)
            im = ax.imshow(data, cmap = 'Greens')
            #ax.set_xticks(fontsize=12)
            #ax.set_yticks(fontsize=12)

            # Show all ticks and label them with the respective list entries.
            ax.set_xticks(np.arange(data.shape[1]), labels=col_ticks, fontsize=12)
            ax.set_yticks(np.arange(data.shape[0]), labels=row_ticks_list[i], fontsize=12)

            ax.spines[:].set_visible(False)
            ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
            ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
            ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
            ax.tick_params(which="minor", bottom=False, left=False)

            # Loop over data dimensions and create text annotations.
            textcolors=("black", "white")
            threshold = im.norm(data.max())/2
            for i in range(len(row_ticks_list[i])):
                for j in range(len(col_ticks)):
                    #color=textcolors[int(im.norm(data[i, j]) > threshold)])
                    text = ax.text(j, i, '{:.2f}'.format(data[i, j]),
                                ha="center", va="center", color=textcolors[int(im.norm(data[i, j]) > threshold)])
        
        # Colorbar
        #cbarlabel = 'Test Top1 Accuracy'
        cbar = ax.cax.colorbar(im)
        #cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=14)
        #ax.cax.toggle_label(True)

        title = 'MNIST'

        plt.xlabel('Local Learning Rate ($f(x)=log_{10}(x)$)', fontsize=14)
        #plt.xticks(fontsize=12)
        #plt.yticks(fontsize=12)
        plt.savefig('{}/{}.pdf'.format(hmdata.pdfpath, title), bbox_inches='tight')
        plt.show()
        
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

if __name__ == "__main__":
    main = Visualazation()
    main.mainProgram()


