'''
Data distribution.
iid { -
    { balanced
non-iid 
    dirichlet [1]    { unbalanced 'dir'
                     { -
        alpha [float]: a smaller `alpha` means more heterogeous data distribution.   
    pathological     { unbalanced [2] 'pat'
                     { balanced * [3][4] 'pat2'
        class [int]: most clients will only have examples of two classes (digits).
Notices:
1.The anotations are not fully unified since we get the code from the references directly. 
  We want to retain the code characteristic so we make a few adjustments. 
2*.Reference [2] has used `balance` in their code, but their number of samples per client is affected by the initial distribution,
  which is explained in Appendix A.
References:
[1] https://github.com/FedML-AI/FedML/blob/2ee0517a7fa9ec7d6a5521fbae3e17011683eecd/fedml_api/data_preprocessing/cifar10/data_loader.py
[2] https://github.com/TsingZ0/PFL-Non-IID
[3] McMahan B, Moore E, Ramage D, et al. Communication-efficient learning of deep networks from decentralized data[C]//Artificial intelligence and statistics. PMLR, 2017: 1273-1282.
[4] https://github.com/AshwinRJ/Federated-Learning-PyTorch
2022 08 05
'''
from multiprocessing.sharedctypes import Value
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import json


def build_distribution(data_path='./datasets/in_use/', dtype='mnist', num_users=16, way='iid-b', alpha=10):
    r'''build distributions on training data. Allocate data idx to users(clients).
    Returns:
        net_dataidx_map (dict): {int: numpy.array*}, e.g. {0: [1,2,3], 1: [4,5,6]}
            * we use `list` or `numpy.array`(priority) here, but in fact `set` it should be, for duplicate elements are not allowed.
        way (str): the way to generate the data distribution. 'iid-b', 'iid-u', 'dir-u', 'pat-u', 'pat2-b' are valid.
    '''
    partition, balance = way.split('-')
    net_dataidx_map = DatasetDistributed.read_net_dataidx_map(data_path=data_path, dtype=dtype, partition=partition, balance=balance, n_nets=num_users, alpha=alpha)
    return net_dataidx_map


class DatasetDistributed():
    def __init__(self, dtype='mnist', data_path='../data/'):
        r'''
        Args:
            dtype: mnist, fashionmnist, cifar10, cifar100
        '''
        self.dtype = dtype
        # only keep train_dataset
        train_dataset, _ = build_dataset(self.dtype, data_path) # use test dataset, please comment these two lines below
        self.y_train = train_dataset.targets     

    def check_args(self, partition, balance):
        if partition == 'iid':
            white_list = ['b', 'u']
        elif partition == 'dir':
            white_list = ['u']
        elif partition == 'pat':
            white_list = ['u', 'unr']
        elif partition == 'pat2':
            white_list = ['b']
        else:
            raise ValueError
        
        if balance not in white_list:
            raise ValueError
    
    def gen_net_dataidx_map(self, check=True, partition='iid', balance='b', n_nets=16, alpha=10):
        r'''Generate net_dataidx_map.
        Args:
            check: If it's True, call check_net_dataidx_map to show the information of the distribution.
            partition: 'iid' or 'dir' or 'pat'
            balance: 'b(alanced)' or 'u(nbalanced)'
            n_nets: the number of clients or nets or users.
            alpha: the alpha of dirichlet distribution
        '''
        # check_args
        self.check_args(partition=partition, balance=balance)

        net_dataidx_map = self.partition_data(partition, balance, n_nets, alpha)
        # check 
        if check == True:
            '''bd2 Apendix A
            a = [0 for _ in range(0, 20)]
            b = [1 for _ in range(0, 20)]
            c = [2 for _ in range(0, 40)]
            d = [3 for _ in range(0, 20)]
            y_train = a + b + c + d
            2022 08 06'''
            y_train = self.y_train
            self.check_net_dataidx_map(ready=True, net_dataidx_map=net_dataidx_map, y_train=y_train, dtype=self.dtype, partition=partition, balance=balance, n_nets=n_nets, alpha=alpha)
        
        if partition == "iid":
            filepath = "./distribution/{}/{}-{}-{}({}).txt".format(self.dtype, self.dtype, partition, balance, n_nets)
        elif partition in ['dir', 'pat', 'pat2']:
            filepath = "./distribution/{}/{}-{}-{}({} {}).txt".format(self.dtype, self.dtype, partition, balance, n_nets, alpha)
        else:
            raise ValueError
        
        self.dumpmap(net_dataidx_map, filepath)

    def partition_data(self, partition, balance, n_nets, alpha):
        r'''partition data to different clients based on distribution.
        Returns:
            net_dataidx_map (dict {client index: idxs of examples}): {int: np.array (or list)}, e.g. {0: [1,2,3]}
        '''
        print("*********partition data***************")
        '''bd2 Appendix A
        a = [0 for _ in range(0, 20)]
        b = [1 for _ in range(0, 20)]
        c = [2 for _ in range(0, 40)]
        d = [3 for _ in range(0, 20)]
        y_train = a + b + c + d
        2022 08 06'''
        y_train = self.y_train
        # to pay attention that cifar10, 100 data is a list, but mnist is a tensor
        if isinstance(y_train, list):
            y_train = np.array(y_train)
        # n_train = y_train.shape[0] can't work when dtype=cifar
        n_train = len(y_train) # the number of training samples
        least_samples = n_train // (10*n_nets) # least samples for each client, `10` is the parameter we set, which can be replaced.
        map_dtype_numclass = {'mnist': 10, 'fashionmnist': 10, 'cifar10': 10, 'cifar100': 100, 'test': 4}
        if self.dtype in map_dtype_numclass.keys():
            num_classes = map_dtype_numclass[self.dtype]
        else:
            raise ValueError
        net_dataidx_map = {}

        if partition == "iid":
            # now 'balance' is ready, 'unbalance' is not completed.
            idxs = np.random.permutation(n_train)
            batch_idxs = np.array_split(idxs, n_nets)
            net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}
        
        elif partition == "dir":
            min_size = 0
            K = num_classes
            N = n_train
            # ensure per client' sampling size >= least_samples (is set to 10 originally in [1])
            while min_size < least_samples:
                idx_batch = [[] for _ in range(n_nets)]
                # for each class in the dataset
                for k in range(K):
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                    # Balance
                    proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))] 
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(n_nets):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]
        
        elif 'pat' in partition:
            # collect the idxs by their labels (or class)
            idxs = np.array(range(n_train))    
            idxs_each_class = []
            for k in range(num_classes):
                #bd2 we add shuffle on the code of [2] to add randomness.
                # `idx_for_each_class.append(idxs[dataset_label == i])` -> three lines below
                # 2022 08 05
                idx_k = idxs[y_train == k]
                np.random.shuffle(idx_k)
                idxs_each_class.append(idx_k)
            class_per_client = int(alpha) # when 'pat', alpha used as class

            if partition == 'pat':
                '''the size of shards depends on the initial distribution of the dataset'''
                num_selected_clients = n_nets*class_per_client // num_classes
                class_per_client_list = np.array([class_per_client for _ in range(n_nets)]) # a list to record the remaining num of classes (or shards) of each client
            
                for i in range(num_classes):
                    num_all_samples = len(idxs_each_class[i])
                    n_sample_per_shard = num_all_samples / num_selected_clients
                    # Decide the clients who will be allocated the idxs of this class.
                    selected_clients = []
                    for client in range(n_nets): # net or client or user are the same meaning
                        if class_per_client_list[client] > 0:
                            selected_clients.append(client)
                        if len(selected_clients) == num_selected_clients:
                            break
                    # produce the number of samples of each shard
                    if balance == 'unr':
                        # unbalanced but nonrandom (affected by the initial sample distribution)
                        num_sample_list = [int(n_sample_per_shard) for _ in range(num_selected_clients-1)]
                    elif balance == 'u':
                        # unbalanced and random
                        num_sample_list = np.random.randint(max(n_sample_per_shard/10, least_samples/num_classes), n_sample_per_shard, num_selected_clients-1).tolist()
                    else:
                        raise ValueError
                    num_sample_list.append(num_all_samples-sum(num_sample_list))
                    # generate the net_dataidx_map and modify the class_per_client_list
                    idx = 0
                    for client, num_sample in zip(selected_clients, num_sample_list):
                        if client not in net_dataidx_map.keys():
                            net_dataidx_map[client] = idxs_each_class[i][idx:idx+num_sample]
                        else:
                            net_dataidx_map[client] = np.append(net_dataidx_map[client], idxs_each_class[i][idx:idx+num_sample], axis=0)
                        idx += num_sample
                        class_per_client_list[client] -= 1
        
            elif partition == 'pat2':
                # decide the number of samples of one shard
                n_sample_per_shard = n_train // (n_nets*class_per_client)
                # compute the number of shards per class
                n_shard_per_class_list = np.array([0 for _ in range(num_classes)])
                for i in range(num_classes):
                    n_shard_per_class = len(idxs_each_class[i]) // n_sample_per_shard
                    n_shard_per_class_list[i] = n_shard_per_class
                bidx_per_class_list = [0 for _ in range(num_classes)] # the begenning index of remaining samples per class
                net_dataidx_map = {i: np.array([], dtype=int) for i in range(n_nets)}
                
                net_i = 0
                for i in range(n_nets):
                    # get the number of shards of this class to decide how many clients to select
                    class_with_surplus_shards = np.where(n_shard_per_class_list > 0)[0]
                    if len(class_with_surplus_shards) >= class_per_client:
                        selected_classes = np.random.choice(class_with_surplus_shards, class_per_client, replace=False)
                        for j in selected_classes:
                            net_dataidx_map[i] = np.concatenate((net_dataidx_map[i], idxs_each_class[j][bidx_per_class_list[j]: bidx_per_class_list[j]+n_sample_per_shard]), axis=0)
                            n_shard_per_class_list[j] -= 1
                            bidx_per_class_list[j] += n_sample_per_shard   
                    else:
                        net_i = i
                        break
                # the method modified from function `mnist_noniid` in the sampling.py in [4]
                if net_i > 0:
                    remaining_idxs = np.concatenate([idxs_each_class[j][bidx_per_class_list[j]:] for j in range(num_classes)], axis=0)
                    n_shard_remaining_idxs = len(remaining_idxs) // n_sample_per_shard
                    distribute_order = np.random.permutation(n_shard_remaining_idxs)
                    for i in range(net_i, n_nets):
                        t = class_per_client*(i-net_i)
                        net_dataidx_map[i] = np.concatenate([remaining_idxs[(distribute_order[t+j])*n_sample_per_shard: 
                                                        (distribute_order[t+j]+1)*n_sample_per_shard] for j in range(class_per_client)], axis=0)             
        else:
            raise ValueError
        return net_dataidx_map
    
    @classmethod
    def check_net_dataidx_map(cls, ready=True, net_dataidx_map=None, y_train=None, data_path="../data/", dtype="mnist", partition="hetero", balance='b', n_nets=16, alpha=10):
        r'''check if the net_dataidx_map is reasonable.
        Args:
            ready: if ready is True, the function will receive net_dataidx_map and y_train; else, data_path will work.
        '''
        if ready == True:
            pass
        else:
            net_dataidx_map = cls.read_net_dataidx_map(dtype, partition, balance, n_nets, alpha)
            train_dataset, _ = build_dataset(dtype, data_path)
            y_train = train_dataset.targets

        map_numclass_dtype = {'mnist': 10, 'fashionmnist': 10, 'cifar10': 10, 'cifar100': 100, 'test': 4}
        if dtype in map_numclass_dtype.keys():
            num_classes = map_numclass_dtype[dtype]
        else:
            raise ValueError

        # count the number of samples per label per client
        n_sample_label_per_client = { i: [] for i in range(n_nets)} # x: client id, y: label, value: number of images
        order = list(net_dataidx_map.keys())
        for i in order:
            num = [0 for _ in range(num_classes)]
            for j in range(0, len(net_dataidx_map[i])):
                num[int(y_train[net_dataidx_map[i][j]])] += 1
            #print(num)
            n_sample_label_per_client[i] = num
        print("*****the number of samples per label per client*****")
        print(n_sample_label_per_client)

        # count the number of samples per client
        n_sample_per_client = [] # x: client id, value: number of images distributed to user x
        for i in range(0, n_nets):
            n_sample_per_client.append(sum(n_sample_label_per_client[i]))
        print("*****the number of samples per client*****")
        print(n_sample_per_client)

        # count the number of samples per label
        n_sample_per_label = [] # x: client id, value: number of images distributed to user x
        for i in range(0, num_classes):
            s = 0
            for j in range(0, n_nets):
                s = s + n_sample_label_per_client[j][i]
            n_sample_per_label.append(s)
        print("*****the number of samples per label*****")
        print(n_sample_per_label)
        
        cls.distribution_bubble(n_sample_label_per_client, dtype, partition, balance, n_nets, alpha, num_classes)    

    @classmethod
    def distribution_bubble(cls, n_sample_label_per_client, dtype, partition, balance, n_nets, alpha, num_classes):
        r'''draw distribution bubble chart.
        Args:
            n_sample_label_per_client: set {int: list}: { x: [y0, y1, ..., yz, ...]}, 
                `x` is the client id [0, n_nets), `yz` is the sample number of label `z` [0, num_classes).
                e.g. {0: [100, 10, 10, ... ]} 
        '''
        if partition == 'iid': 
            b = 'balanced'
            title = "Training Label Distribution {}-{}({})".format(partition, b, n_nets)
        elif partition == 'dir':
            b = 'unbalanced'
            title = "Training Label Distribution {}-{}({} {})".format(partition, b, n_nets, alpha)
        elif partition == 'pat':
            b = 'unbalanced'
            title = "Training Label Distribution {}-{}({} {})".format(partition, b, n_nets, alpha)
        elif partition == 'pat2':
            b = 'balanced'
            title = "Training Label Distribution {}-{}({} {})".format(partition, b, n_nets, alpha)
        else:
            raise ValueError

        x = []
        for i in range(0, n_nets):
            x.extend([i for _ in range(0, num_classes)])
        #print(x)

        y = []
        for i in range(0, n_nets):
            y.extend([j for j in range(0, num_classes)])
        #print(y)

        size = []
        for i in range(0, len(x)):
            size.append(n_sample_label_per_client[x[i]][y[i]])
        size = [i*0.2 for i in size]

        plt.figure()
        plt.scatter(x, y, s=size, alpha=1)
        plt.title(title)
        plt.xlabel("Client ID")
        plt.ylabel("Training Labels")
        plt.savefig('./distribution/{}/{}.png'.format(dtype, title))
        plt.show()

    @classmethod
    def read_net_dataidx_map(cls, data_path='./datasets/distribution/', dtype="mnist", partition="iid", balance='b', n_nets=16, alpha=10):
        r'''read net_dataidx_map.'''
        if partition == "iid":
            filepath = "{}/{}-{}-{}({}).txt".format(data_path, dtype, partition, balance, n_nets)
        elif partition in ['dir', 'pat', 'pat2']:
            filepath = "{}/{}-{}-{}({} {}).txt".format(data_path, dtype, partition, balance, n_nets, alpha)
        net_dataidx_map = cls.loadmap(filepath)

        return net_dataidx_map
    
    @classmethod
    def dumpmap(cls, net_dataidx_map, filepath):
        for i in range(0, len(net_dataidx_map)):   
            if isinstance(net_dataidx_map[i], list) == False:
                net_dataidx_map[i] = net_dataidx_map[i].tolist()
        #print(net_dataidx_map)
        with open(filepath, 'w') as f:
            json.dump(net_dataidx_map, f)
    
    @classmethod
    def loadmap(cls, filepath):
        with open(filepath, 'r') as f:
            temp = json.load(f)
        # Badly, for json.load will form dict{'0': []}, instead of dict{0: []}
        # dict{'0': []} -> dict{0: []}
        net_dataidx_map = dict()
        for i in range(0, len(temp)):
            net_dataidx_map[i] = np.array(temp[str(i)]) 
        return net_dataidx_map


class SubDataset(Dataset):
    r'''Generate Sub-dataset of the dataset given based on idxs.
    Args:
        dataset (torch.utils.data.Dataset): datasets
        idx (list): A part indexes of the dataset given.
    '''
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


# python distributions.py -d mnist -n 100 --partition pat2 --balance b -a 2 
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='mnist', help='dataset')
    parser.add_argument('-n', type=int, default=100, help='divide into n portions')
    parser.add_argument('--partition', type=str, default='iid', help='iid or dir or pat or pat2')
    parser.add_argument('--balance', type=str, default='b', help='balanced or unbalanced')
    parser.add_argument('-a', type=float, default=1.0, help='the alpha of dirichlet distribution or the classes of pathological partition')
    args = parser.parse_args()
    print(args)
    

    data_path = '../data/'
    n_nets = args.n # divide into n portions
    alpha = args.a
    dtype = args.d

    a = DatasetDistributed(dtype, data_path)
    a.gen_net_dataidx_map(check=True, partition=args.partition, balance=args.balance, n_nets=n_nets, alpha=alpha)
    #a.check_net_dataidx_map(net_dataidx_map=None, y_train=None, data_path=data_path, dtype=dtype, partition=partition, n_nets=n_nets, alpha=alpha)


if __name__ == '__main__':
    from datasets import build_dataset
    main()


'''Appendix A
the pathological balanced is not balanced at all. The reasons are shown below. [2] has not said they make method 
'for pathological noniid and unbalanced setting', but there is `if balance` in their code, which confused me a lot.
If we use the code from [2], we will have the example below.
four labels[number of samples]: 0[20], 1[20], 2[40], 3[20]
n_nets: 10
class per client: 2
then we will have the results client [labels|number of samples]:
0 [0,1|8];  1 [0,1|8];  2 [0,1|8];  3 [0,1|8];  4 [0,1|8];
5 [2,3|12]; 6 [2,3|12]; 7 [2,3|12]; 8 [2,3|12]; 9 [2,3|12].
That is to say the pathological balance distribution of [2] is determined by the distribution of the initial data distribution.
To reproduce the code, uncomment the mark `Appendix A`.
2022 08 06
'''





