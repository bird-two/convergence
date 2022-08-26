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
    

class MyPlot():
    def __init__(self):
        print('plot --- ')

    def plotgap(self):
        plt.figure()
        color = ['xkcd:red', 'xkcd:blue', 'xkcd:violet', 'xkcd:orange','xkcd:green', 'xkcd:cyan', 'xkcd:pink', 'xkcd:grey']
        #line_labels = self.gen_linetitle()
        
        title = 'gmix+(1.0 0.001 0.0) splitfed(1 16 1) alex(1) cifar10(non 0.1) hp(0.0001 0.001 32) gap'

        plt.title(title)
        path = '../save/splitfed(1 16 1) alex(1) cifar10(non 0.1) hp(0.0001 0.001 32)/'
        # plot curve
        start_epoch = 1
        end_epoch = 200
        epochs = range(start_epoch, end_epoch+1)
        df = read_fromcsv(title, path)
        plt.plot(epochs, df.iloc[start_epoch-1:end_epoch, 0].values)      
        plt.ylabel('gap')
        plt.xlabel('Epoch')
        plt.grid(linestyle='dotted')
        plt.savefig('{}/{}.png'.format(path, title))
        plt.show()
    
    def gettitle(self, t, folder):
        ylabel_name = ['Train Loss', 'Test Loss', 'Train Top1 Accuracy', 'Test Top1 Accuracy', 'Train Top5 Accuracy', 'Test Top5 Accuracy']
        title = '{}'.format(folder)
        ylabel = ylabel_name[t]
        return title, ylabel
    
    def banfiles(self, files, paths, bans):
        for i in range(len(files)-1, -1, -1):
            for j in bans:
                if j in resolvefilename(files[i]):
                    files.remove(files[i])
                    paths.remove(paths[i])

    def choosefiles(self, files, paths, chooses):
        temp = []
        path_temp = []
        for i in range(0, len(files)):
            for j in chooses:
                if j in resolvefilename(files[i]):
                    temp.append(files[i])
                    path_temp.append(paths[i])
        return temp, path_temp
    
    def whichpart(self, files):
        whichpart = []    
        flags = resolvefilename(files[0])
        if len(flags) != 6:
            raise
        for i in range(1, len(files)):
            if 'client' in files[i] or 'server' in files[i]:
                t = resolvefilename(files[i])
                # len(t)-1 to remove 'server' and 'client'
                for j in range(0, len(t)-1):
                    if t[j] != flags[j] and j not in whichpart:
                        whichpart.append(j)
        return whichpart
    
    def getlegendname(self, file, whichpart):
        partlist = resolvefilename(file)
        # gmix(0.0) is alse called as glocal
        if partlist[0] == 'gmix(0.0)': #bd2 `gmix(0.0)` is the previous name of `glocal` 2022 05 05
            partlist[0] = 'glocal'
        elif partlist[0] == 'grey':
            partlist[0] = 'replay'
        temp = []
        for i in whichpart:
            temp.append(partlist[i])
        return ' '.join(temp)
    
    def getmarker(self, file):
        marker = ['^', 'v', 's', 'o', 'D', 'x', '.', '*', '+', 'd']
        temp = resolvefilename(file)
        if temp[0] == 'base':
            return 's'
        elif temp[0] == 'glocal':
            return 'x'
        elif temp[0] == 'gdelay':
            return '^'
        elif temp[0] == 'replay' or temp[0] == 'grey': #bd2 `grey` is the previous name of `replay` 2022 05 05
            return 'd'
        elif 'gld' in temp[0]:
            return 'v'
        else:
            return '.'
    
    def getcolor(self, file):
        color = ['xkcd:red', 'xkcd:blue', 'xkcd:green', 'xkcd:orange', 'xkcd:brown', 'xkcd:violet', 'xkcd:grey', 'xkcd:cyan', 'xkcd:pink']
        color_dict = {'base':'xkcd:blue', 'glocal':'xkcd:orange', 'gmix(0.0)':'xkcd:blue', 
                      'gdelay':'xkcd:green', 'replay':'xkcd:brown', 'grey':'xkcd:brown'}
        temp = resolvefilename(file)
        alg = temp[0]
        if alg in color_dict.keys():
            return color_dict[alg]
        elif 'gld' in alg:
            return 'xkcd:orange'
        else:
            return 'xkcd:violet'

    def sortfiles(self, files, pathplus):
        r'''sort files based on ['base', 'glocal', 'gdelay', 'replay', others, ...]
        Returns:
            sorted_files: a new files list with order above
        '''
        sorted_files = []
        sorted_pathplus = []
        order = ['base', 'glocal', 'gdelay', 'replay']
        for i in range(0, len(order)):
            # Reverse traversal
            for j in range(len(files)-1, -1, -1):
                if order[i] in files[j]:
                    sorted_files.append(files[j])
                    sorted_pathplus.append(pathplus[j])
                    del files[j]
                    del pathplus[j]
        sorted_files.extend(files)
        sorted_pathplus.extend(pathplus)
        return sorted_files, sorted_pathplus
    
    def gettime(self, round_time):
        time = []
        for i in range(0, len(round_time)):
            if i > 0:
                time.append(round_time[i]+time[i-1]) 
            else:
                time.append(round_time[i])
        return time

    def plotloss(self):
        '''
        different algorithm
        '''
        #mpl.style.use('seaborn')
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-t', type=int, default=0, help='type')
        parser.add_argument('-s', type=int, default=1, help='start epoch')
        parser.add_argument('-e', type=int, default=300, help='end epoch')
        parser.add_argument('--time', type=int, default=0, help='x-axis is time ?')
        parser.add_argument('-f', type=str, nargs='*', default='splitfed(1 16 1) alex(1) cifar10(non 0.1) hp(0.0001 0.001 32)', help='folder')
        parser.add_argument('--choose', type=str, nargs='*', default=[], help='chooses')
        parser.add_argument('--ban', type=str, nargs='*', default=[], help='ban')
        args = parser.parse_args()
        print(args)

        t = args.t
        folder = args.f
        path = ['../save/{}/'.format(x) for x in folder]
        title, ylabel = self.gettitle(t, folder[0])
        
        files = []
        pathplus = []
        for i in range(0, len(path)):
            files.extend(os.listdir(path=path[i]))
            pathplus.extend([path[i] for _ in range(len(os.listdir(path=path[i])))])
        for i in range(0, len(files)):
            files[i] = os.path.splitext(files[i])[0]

        # ban some line labels
        if args.ban != []:
            self.banfiles(files=files, paths=pathplus, bans=args.ban)

        # choose some line labels
        if args.choose != []:
            files, pathplus = self.choosefiles(files=files, paths=pathplus, chooses=args.choose)

        # choose the different partions based on the filenames as legend name
        whichpart = self.whichpart(files=files)
        if whichpart == []:
            whichpart = [0]
        print(whichpart)

        # sort the files based on ['base', 'glocal', 'gdelay', 'replay',...]
        files,pathplus = self.sortfiles(files=files, pathplus=pathplus)
        
        plt.figure()
        
        plt.title(' '.join(resolvefilename(title)[:-1]))
        #plt.title(title)
        start_epoch = args.s # 1
        #epochs = range(start_epoch, end_epoch+1)
        lines = []
        k = 0
        for i in range(0, len(files)):
            print(files[i])
            # if 'client' in files[i]:
            #     df = read_fromcsv(files[i], pathplus[i])
            #     end_epoch = min(args.e, len(df))
            #     if args.time == 1:
            #         x_axis = self.gettime(df.iloc[start_epoch-1:end_epoch, 6])
            #     else:
            #         x_axis = range(1, end_epoch+1)
            #     plt.plot(x_axis, df.iloc[start_epoch-1:end_epoch, t].values, linestyle='dashed', color=None, \
            #             marker=self.getmarker(files[i]), markerfacecolor='none', markevery=100, markersize=6, markeredgewidth=2)
            #     lines.append('{}'.format(self.getlegendname(files[i], whichpart)))
            if 'server' in files[i]:
                df = read_fromcsv(files[i], pathplus[i])
                end_epoch = min(args.e, len(df))
                if args.time == 1:
                    if 'splitfed(1 5 1)' in files[i]:
                        c = 7
                    else:
                        c = 6
                    x_axis = self.gettime(df.iloc[start_epoch-1:end_epoch, c])
                else:
                    x_axis = range(1, end_epoch+1)
                
                plt.plot(x_axis, df.iloc[start_epoch-1:end_epoch, t].values, color=None, \
                        marker=self.getmarker(files[i]), markerfacecolor='none', markevery=100, markersize=6, markeredgewidth=2)
                lines.append('{}'.format(self.getlegendname(files[i], whichpart))) # self.getcolor(files[i])
                k = k+1
                
        plt.ylabel(ylabel)
        if args.time == 1:
            plt.xlabel('Time (s)')
        else:
            plt.xlabel('Rounds')
        if args.t in [0, 1]:
            loc = 1
        else:
            loc = 4
        plt.legend(lines, loc=loc)
        plt.grid()
        #plt.savefig('{}/{} {}.png'.format(pathplus[0], title, ylabel))
        plt.savefig('{}/{} {}.pdf'.format(pathplus[0], title, ylabel))
        plt.show()
    
    def plotsubplots(self):
        t = 3
        fig, axs = plt.subplots(3, 3)
        folders = ['splitfed(1 100 1) alexnet(1) cifar10(iid) hp(0.0001 0.001 32)', 
                 'splitfed(1 100 1) alexnet(1) cifar10(non 0.5) hp(0.0001 0.001 32)',
                 'splitfed(1 100 1) alexnet(1) cifar10(non 0.1) hp(0.0001 0.001 32)',
                 'splitfed(1 100 1) vgg11(1) cifar10(iid) hp(0.0001 0.001 32)',
                 'splitfed(1 100 1) vgg11(1) cifar10(non 0.5) hp(0.0001 0.001 32)',
                 'splitfed(1 100 1) vgg11(1) cifar10(non 0.1) hp(0.0001 0.001 32)',
                 'splitfed(1 100 1) resnet20(1) cifar10(iid) hp(0.0001 0.001 32)',
                 'splitfed(1 100 1) resnet20(1) cifar10(non 0.5) hp(0.0001 0.001 32)',
                 'splitfed(1 100 1) resnet20(1) cifar10(non 0.2) hp(0.0001 0.001 32)']
        algs = ['base', 'glocal', 'gdelay']
        path = '../save/'
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    file = '{} {} {}'.format(algs[k], folders[i*3+j], 'server')
                    file_path = '{}/{}/'.format(path, folders[i*3+j])
                    print(file_path)
                    df = read_fromcsv(file, file_path)
                    x_axis = [i for i in range(0, len(df))]
                    axs[i, j].plot(x_axis, df.iloc[0:len(df), t].values, linestyle='solid', color=None, \
                        marker=self.getmarker(file), markerfacecolor='none', markevery=100, markersize=6, markeredgewidth=2)
                    axs[i, j].set_title(' '.join(resolvefilename(file)[2:-2]))
        
        k = 0
        for ax in axs.flat:
            if k in [0, 3, 6]:
                ax.set(ylabel='Test Top1 Accuracy')
                #ax.set(ylabel='Train Loss')
            if k in [6, 7, 8]:
                ax.set(xlabel='Rounds')
            ax.grid(axis='y')
            k = k+1
        
        # custom fig
        fig.set_figwidth(12)
        fig.set_figheight(9)
        
        fig.tight_layout()
        fig.legend(algs, loc=8, ncol=3, prop={'size': 12}, frameon=False)
        fig.subplots_adjust(bottom=0.1)
        
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        # for ax in axs.flat:
        #     ax.label_outer()
        plt.show()

    def plotsubplots2(self):
        fig, axs = plt.subplots(2, 4)
        folders = ['splitfed(1 100 1) vgg11(1) cifar10(non 0.5) hp(0.0001 0.001 32)',
                   'splitfed(1 100 1) vgg11(1) cifar100(non 0.5) hp(0.0001 0.001 32)',
                   'splitfed(1 100 1) resnet20(1) cifar10(non 0.2) hp(0.0001 0.001 32)', 
                   'splitfed(1 100 1) resnet20(1) cifar100(non 0.2) hp(0.0001 0.001 32)']
        algs = ['base', 'glocal', 'gdelay']
        path = '../save/'
        
        for i in range(0, 2):
            for j in range(0, 2):
                for k in range(0, 3):
                    file = '{} {} {}'.format(algs[k], folders[i*2+j], 'server')
                    file_path = '{}/{}/'.format(path, folders[i*2+j])
                    df = read_fromcsv(file, file_path)
                    x_axis = [i for i in range(0, len(df))]
                    axs[i, j].plot(x_axis, df.iloc[0:len(df), 0].values, linestyle='solid', color=None, \
                        marker=self.getmarker(file), markerfacecolor='none', markevery=100, markersize=6, markeredgewidth=2)
                    axs[i, j].set_title(' '.join(resolvefilename(file)[2:-2]))

                    axs[i, j+2].plot(x_axis, df.iloc[0:len(df), 3].values, linestyle='solid', color=None, \
                        marker=self.getmarker(file), markerfacecolor='none', markevery=100, markersize=6, markeredgewidth=2)
                    axs[i, j+2].set_title(' '.join(resolvefilename(file)[2:-2]))
        
        k = 0
        for ax in axs.flat:
            if k in [0, 1, 4, 5]:
                ax.set(ylabel='Train Loss')
            if k in [2, 3, 6, 7]:
                ax.set(ylabel='Test Top1 Accuracy')
            
            ax.set(xlabel='Rounds')
            ax.grid(axis='y')
            k = k+1
        
        # custom fig
        fig.set_figwidth(16)
        fig.set_figheight(7)
        
        fig.tight_layout()
        fig.legend(algs, loc=8, ncol=3, prop={'size': 12}, frameon=False)
        fig.subplots_adjust(bottom=0.12)
        
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        # for ax in axs.flat:
        #     ax.label_outer()
        plt.show()
    

if __name__ == '__main__':
    a = MyPlot()
    a.plotloss()  
    #a.plotsubplots2() 

# def gen_linetitle2(self, folder):
#     algs = ['lgl', 'olp', 'olpl2(0.01)']
#     ends = ['client', 'server']

#     # add alg
#     a = ['{} {}'.format(x, folder) for x in algs]
#     # add ends
#     b = []
#     for i in range(0, len(a)):
#         b.extend('{} {}'.format(a[i], x) for x in ends)
#     #print(b)
#     return b


