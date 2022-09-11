import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import argparse

from outputs import read_fromcsv

class TestData():
    def __init__(self):
        self.dataset = 'mnist'
        self.path = '../save/Global learning rate/{}/'.format(self.dataset)
        self.pdfname = 'MNIST, 100 clients, 5 classes'
        self.glr_choice = [1, 10, 15, 20, 25, 30, 50, 100]
        self.true_glr = [0.1, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]
        self.llr_choice = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
        self.true_llr = [1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]  
    
    def gen_lr(self):
        self.lr_choice = set()
        for glr in self.glr_choice:
            for llr in self.llr_choice:
                #print('lr={} with glr={} and llr={}'.format(llr*glr, glr, llr))
                self.lr_choice.add(glr*llr)
        #print(self.lr_choice)
        return self.lr_choice
    
    def classify(self):
        self.lr_dict = { i:[] for i in self.lr_choice }
        for lr in self.lr_dict.keys():
            for i, glr in enumerate(self.glr_choice):
                for j, llr in enumerate(self.llr_choice):
                    if glr*llr == lr:
                        self.lr_dict[lr].append((i, j))
        for i, lr in enumerate(self.lr_dict.keys()):
            if len(self.lr_dict[lr]) > 2:
                print(i, len(self.lr_dict[lr]))
        return self.lr_dict

    def get_raw_files(self):
        self.raw_files = os.listdir(self.path)
        return self.raw_files

    def collect_files(self, c):
        self.gen_lr()
        self.classify()
        self.get_raw_files()
        keys = list(self.lr_dict.keys())
        #c = 0
        self.legend = []
        files = []
        self.title = '{}, $\eta$={}'.format(self.pdfname, keys[c]/1000000)
        self.pdfname = '{}, eta={}'.format(self.pdfname, keys[c]/1000000)
        for i,j in self.lr_dict[keys[c]]:
            filename = 'base sl(100 1 {}) lenet5(2) {}(pat2-b 5.0) sgd({} 0.9 0.0001) hp(10)'.format(self.true_glr[i], self.dataset, self.true_llr[j])
            print(filename)
            #print(self.raw_files)
            if '{}.csv'.format(filename) in self.raw_files:
                #df = read_fromcsv(filename, self.path)
                #t = df.iloc[0:50, 3].values
                #data_point = t[-10:].var(axis=0)
                #data_point = t[-30:].mean(axis=0)
                files.append(filename)
                self.legend.append('$\eta_g$={}, $\eta_l$={}'.format(self.true_glr[i], self.true_llr[j]))
        return files
 
    def gen_heatmapdata(self):
        #self.gen_lr()
        self.get_raw_files()
        self.collect_files()
        data_point_map = np.zeros((len(self.glr_choice), len(self.llr_choice)))
        print(self.files.keys())
        for i, key in enumerate(self.files.keys()):
            for j in range(len(self.files[key])):
                if self.files[key][j] == '':
                    data_point_map[i,j] = 0
                else:
                    #df = read_fromcsv(self.files[key][j], self.path)
                    #t = df.iloc[0:50, 3].values
                    #data_point = t[-10:].var(axis=0)
                    #data_point = t[-30:].mean(axis=0)
                    #data_point_map[i,j] = data_point
                    data_point_map[i,j] = key*self.llr_choice[j]
        return data_point_map              

def plotcurve(args):
    #mpl.style.use('seaborn')

    fncurve = TestData()
    path = fncurve.path
    #ylabel = fncurve.get_ylabel(args.t)
    
    #x_axis = Xaxis()
    #xlabel = x_axis.get_xlabel(args.x)

    #raw_files = fncurve.get_raw_files(path)
    sl_files = fncurve.collect_files(args.c)
    sl_legend = fncurve.legend
    title = fncurve.title
    pdfname = fncurve.pdfname

    plt.figure()
    plt.title(title, fontsize=14)
    start_epoch = args.s # 1
    #epochs = range(start_epoch, end_epoch+1)
    lines = []
    
    for i in range(len(sl_files)):
        df = read_fromcsv(sl_files[i], path)
        end_epoch = len(df)
        x_axis = range(1, end_epoch+1)
        plt.plot(x_axis, df.iloc[start_epoch-1:end_epoch, args.t].values, color=None, lw=2)
        lines.append('{}'.format(sl_legend[i]))

    #plt.ylabel(ylabel, fontsize=12)
    #plt.xlabel(xlabel, fontsize=12)
    plt.legend(lines, loc=None, ncol=1, prop={'size': 12}) # loc = 1 or 4
    plt.grid()
    #plt.savefig('{}/{} {}.png'.format(pathplus[0], title, ylabel))
    plt.savefig('{}/{}.pdf'.format(path, pdfname), bbox_inches='tight')
    plt.show()


def gen_heatmap(data, row_ticks, col_ticks):
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap = 'Greens')

    # Create colorbar
    cbarlabel = 'Test Top1 Accuracy'
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=12)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_ticks)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_ticks)

    # Loop over data dimensions and create text annotations.
    textcolors=("black", "white")
    threshold = im.norm(data.max())/2
    for i in range(len(row_ticks)):
        for j in range(len(col_ticks)):
            #color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = ax.text(j, i, '{:.5f}'.format(data[i, j]),
                        ha="center", va="center", color=textcolors[int(im.norm(data[i, j]) > threshold)])

    title = "MNIST, 100 clients, 5 classes"
    ax.set_title(title, fontsize=12)
    plt.ylabel('Global Learning Rate', fontsize=12)
    plt.xlabel('Local Learning Rate (log 10)', fontsize=12)
    fig.tight_layout()
    plt.savefig('{}/{}.pdf'.format('../save/Global learning rate/', title), bbox_inches='tight')
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('-t', type=int, default=0, help='test accuracy or train loss')
parser.add_argument('-x', type=str, default=0, choices=['round', 'iteration', 'amount'], help='X-axis')
#parser.add_argument('--glr', type=float, default=None, help='global learning rate')
#parser.add_argument('--llr', type=float, default=None, help='local learning rate')
parser.add_argument('-s', type=int, default=1, help='start epoch')
parser.add_argument('-c', type=int, default=0, help='which eta')
args = parser.parse_args()
print(args)   


def main():
    plotcurve(args) # 2 5 12 18 27 33 34 37
    #data = TestData()
    #lr = data.gen_lr()
    #lr_dict = data.classify()
    
    #row_ticks = [0.1, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]
    #col_ticks = [1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    #gen_heatmap(data_point_map, row_ticks[::-1], col_ticks)


if __name__ == '__main__':
    main()