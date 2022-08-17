import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from outputs import read_fromcsv

class HeatMapData():
    def __init__(self):
        self.path = '../save/Global learning rate/'
        self.glr_choice = [0.1, 1.0, 1.5, 2.0, 5.0]
        self.llr_choice = [1e-5, 0.0001, 0.001, 0.01, 0.1]
    
    def collect_files(self):
        files = {i: [] for i in self.glr_choice[::-1]} # reverse the list glr_choice
        for i in range(len(self.glr_choice)):
            for j in range(len(self.llr_choice)):
                t = 'base sl(100 1 {}) lenet5(2) mnist(pat2-b 5.0) sgd({} 0.9 0.0001) hp(10)'.format(self.glr_choice[i], self.llr_choice[j])
                if '{}{}'.format(t, '.csv') in self.raw_files:
                    files[self.glr_choice[i]].append(t)
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
                df = read_fromcsv(self.files[key][j], self.path)
                t = df.iloc[0:50, 3].values
                #data_point = t[-10:].var(axis=0)
                data_point = t[-30:].mean(axis=0)
                data_point_map[i,j] = data_point
        return data_point_map              

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
            text = ax.text(j, i, '{:.2f}'.format(data[i, j]),
                        ha="center", va="center", color=textcolors[int(im.norm(data[i, j]) > threshold)])

    title = "MNIST, 100 clients, 5 classes"
    ax.set_title(title, fontsize=12)
    plt.ylabel('Global Learning Rate', fontsize=12)
    plt.xlabel('Local Learning Rate (log 10)', fontsize=12)
    fig.tight_layout()
    plt.savefig('{}/{}.pdf'.format('../save/Global learning rate/', title), bbox_inches='tight')
    plt.show()

def main():
    heatmapdata = HeatMapData()
    data_point_map = heatmapdata.gen_heatmapdata()
    row_ticks = [0.1, 1.0, 1.5, 2.0, 5.0]
    col_ticks = [-5, -4, -3, -2, -1]
    gen_heatmap(data_point_map, row_ticks[::-1], col_ticks)


if __name__ == '__main__':
    main()