'''
The implementation of Federated Learning[1]. 
Here we average the weight after receiving the weight from any client immediately to save the memory.
Class AverageAcrossClients is main modification.
Some examples are shown below, which indicates that this modification makes running time longer with improvement of memory consumption.
But the difference is a little, so we adopt the vanilla one. By the way, we find it bad that the first running task needs the least time.
References:
[1] McMahan B, Moore E, Ramage D, et al. Communication-efficient learning of deep networks from decentralized data[C]//Artificial intelligence and statistics. PMLR, 2017: 1273-1282.
[2] https://github.com/chandra2thapa/SplitFed
[3] https://github.com/AshwinRJ/Federated-Learning-PyTorch
2022 08 03
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD

import numpy as np
import random
import time
import copy

from utils.utils import AverageMeter, accuracy, group_clients, average_weights
from utils.outputs import record_tocsv, switchonlog, loginfo
from datasets.datasets import build_dataset
from datasets.distributions import SubDataset, build_distribution
from models.build_models import build_model


import argparse
parser = argparse.ArgumentParser()
model_names = ['lenet5', 'vgg11']
alg_names = ['base']
parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg11', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: vgg11)')
parser.add_argument('-d', default='cifar10', type=str, metavar='N', choices=['mnist', 'fashionmnist', 'cifar10', 'cifar100'], help='dataset')
parser.add_argument('-s', default=1, type=int, metavar='N', help='split layer')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--local-epochs', default=1, type=int, metavar='N', help='number of local epochs to run')
parser.add_argument('--clients', default=100, type=int, metavar='N', help='number of total clients')
parser.add_argument('--servers', default=1, type=int, metavar='N', help='number of total servers')
parser.add_argument('--alg', default='base', type=str, choices=alg_names, metavar='N', help='the algorithm')
parser.add_argument('--distribution', default='iid-b', type=str, metavar='N', choices=['dir-u', 'iid-b', 'pat2-b'], help='way to sample')
parser.add_argument('--alpha', default=10, type=float, metavar='N', help='the alpha of dirichlet distribution')
parser.add_argument('--optim', default='sgd', type=str, choices=['sgd', 'adam'], metavar='N', help='the optimizer')
parser.add_argument('--lr', default=0.0001, type=float, metavar='LR', help='initial learning rate of client')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='N', help='weight decay of client optimizer')
parser.add_argument('--momentum', default=0, type=float, metavar='N', help='momentum of client optimizer')
parser.add_argument('--global-lr', default=1, type=float, metavar='LR', help='global learning rate, i.e. `\eta_g` of the paper')
parser.add_argument('-b', '--batch-size', default=50, type=int, metavar='N',
                    help='mini-batch size (default: 50), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=1234, type=int, help='seed for initializing training. ')
args = parser.parse_args()

#python fl.py -a lenet5 -d mnist --epochs 50 --local-epochs 1 --clients 100 --distribution pat2-b --alpha 2 --optim sgd --lr 0.01 --momentum 0.9 --global-lr 1 --batch-size 10 

# hyperparameters
rounds = args.epochs
local_epochs = args.local_epochs
users = args.clients # number of clients

# seed
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    #print(torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

topacc = [1, 5]
# collect server's loss and tops
train_loss = []
val_loss = []
train_tops = [[] for _ in range(len(topacc))]
val_tops = [[] for _ in range(len(topacc))]


def main():
    global args, topacc, device
    # log
    switchonlog('../convergence/save/', getprojectname(args))
    loginfo('args: {}\ndevice: {}'.format(args, device))
    
    # datasets and generate train local datasets
    train_dataset, test_dataset = build_dataset(args.d, data_path='../data/')
    
    train_loaders = []
    localdataset_idxs_train = build_distribution('../convergence/datasets/in_use/', args.d, users, way=args.distribution, alpha=args.alpha)
    for i in range(users):
        train_loaders.append(DataLoader(SubDataset(train_dataset, localdataset_idxs_train[i]), batch_size=args.batch_size, shuffle=True))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
    # Dataset size for the client-size -- client wise averaging   
    datasetsize_client = []
    for i in range(users):
        datasetsize_client.append(len(localdataset_idxs_train[i]))  
    loginfo('datasetsize_client\n{}'.format(datasetsize_client.__str__()))

    # model
    global_model = build_model(arch=args.arch, dtype=args.d, alg=args.alg)
    global_model.to(device)
    loginfo('model\n{}'.format(global_model))

    # loss function
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()    # store start time
    print("timmer start!")

    global_model.train()
    record_weight = AverageAcrossClients(global_model.state_dict())
    
    for r in range(rounds):
        record_weight.reset()
        for u in range(users):
            local_model = localupdate(args, train_loaders[u], model=copy.deepcopy(global_model), criterion=criterion, device=device)
            record_weight.update(local_model.state_dict(), datasetsize_client[u])
        # global update, use `\eta_g` in the paper.
        w_server = globalupdate(global_model.state_dict(), record_weight.average(), args)
        global_model.load_state_dict(w_server)   
        #global_model.load_state_dict(record_weight.average())
        
        # Train accuracy and loss check at each round (this is for the gloabl model -- not local model)
        inference(train_loaders, global_model, criterion, r)
        # Test accuracy and loss evaluate at each round
        evaluate(test_loader, global_model, criterion, r)

    end_time = time.time()  # store end time
    loginfo("TrainingTime: {} sec".format(end_time - start_time))
    recorddata()


def localupdate(args, train_loader, model, criterion, device):
    r'''local_epoch `E` is the number of training passes each client makes over its local dataset on each round [1].
    It is worth noting that `K = E(n_k/B)`, 
    where `K` denotes the local steps defined in our paper, `n_k` is the local examples of client `k`, `B` denotes the Mini-batch size.
    '''
    if args.optim == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    model.train()
    for local_epoch in range(args.local_epochs):
        for input, labels in train_loader:
            # get the inputs; data is a list of [inputs, labels]
            input = input.to(device)
            labels = labels.to(device)

            # client forward training
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    return model


def globalupdate(w_server, round_weight, args):
    r'''Update the global model with the round model and `eta_g` (see algorithm in our paper).
    Two conditions are adopted here, because if using the second, true but different result will be gotten (maybe caused by float caculation of computer).'''
    global_lr = torch.tensor(args.global_lr)
    if global_lr == 1.0: # Although float, "==" is ok.
        # update the global model
        for key in w_server.keys():
            w_server[key] = round_weight[key]
    else:
        for key in w_server.keys():
            w_server[key] = w_server[key] + global_lr*(round_weight[key]-w_server[key])

    return w_server


def inference(train_loaders, model, criterion, round):
    r'''compute training accuracy and loss.'''
    global users, train_loss, train_tops, topacc

    model.eval()
    # train acc for each client's training dataset
    with torch.no_grad():
        # to record every local datasets' server top1, top5, losses and caculate avg.
        avglosses = AverageMeter()
        avgtops = [AverageMeter() for _ in range(0, len(topacc))]
        
        for u in range(users):

            # Meters for server
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
        
            for j, trn in enumerate(train_loaders[u]):
                trn_x, trn_label = trn
                trn_x = trn_x.to(device)
                trn_label = trn_label.to(device)

                trn_output = model(trn_x) # forward
                loss = criterion(trn_output, trn_label)
                acc1, acc5 = accuracy(trn_output.data, trn_label.data, topk=topacc)
                losses.update(loss.item(), trn_label.size(0))
                top1.update(acc1.item(), trn_label.size(0))
                top5.update(acc5.item(), trn_label.size(0))
            
            avglosses.update(losses.avg, losses.count)
            avgtops[0].update(top1.avg, top1.count)
            avgtops[1].update(top5.avg, top5.count)

        loginfo("rounds {}'s server train acc avg: {:.2f}%, train loss avg: {:.4f}".format(round + 1, avgtops[0].avg, avglosses.avg), 'green')
                    
        train_loss.append(avglosses.avg)
        for i in range(0, len(topacc)):
            train_tops[i].append(avgtops[i].avg) 


def evaluate(val_loader, model, criterion, round):
    r'''compute test accuracy and loss.'''
    global val_loss, val_tops, topacc # to record servers' loss and tops
    
    model.eval()
    with torch.no_grad():
        losses = AverageMeter()
        tops = [AverageMeter() for _ in range(0, len(topacc))]
        
        for j, val in enumerate(val_loader):
            val_x, val_label = val
            val_x = val_x.to(device)
            val_label = val_label.to(device)

            val_output = model(val_x) # forward
            loss = criterion(val_output, val_label)
            acc1, acc5 = accuracy(val_output.data, val_label.data, topk=topacc)
            losses.update(loss.item(), val_label.size(0))
            tops[0].update(acc1.item(), val_label.size(0))
            tops[1].update(acc5.item(), val_label.size(0))
        
        loginfo("rounds {}'s server test acc: {:.2f}%, test loss: {:.4f}".format(round + 1, tops[0].avg, losses.avg), 'red')
        
        val_loss.append(losses.avg)
        for i in range(0, len(topacc)):
            val_tops[i].append(tops[i].avg)


class AverageAcrossClients():
    r'''Average the weights across selected clients based on the weight (the first weight is the model parameters) on the fly,
    which is different from the way collecting all the weights and averaging at last.'''
    def __init__(self, initweights):
        self.sum = copy.deepcopy(initweights)
        self.reset()
    
    def reset(self):
        for key in self.sum.keys():
            # if we use "w[i][key] *= float(data)", when resnet parameters may have different type that is not float
            self.sum[key].zero_()
        self.sum_weight = torch.tensor(0)

    def update(self, weights, weightclient):
        r'''Since `weights` has been used for the model parameters, we use `weightclient` to denote the weight (for weighted average) of the client.
        Record the sum of model parameters. Return the average parameters only when needed.
        Args:
            weights: the model parameters;
            weightclient (int):
        '''
        weightclient = torch.tensor(weightclient)
        self.weighted_sum(weights, weightclient)
        self.sum_weight += weightclient
    
    def average(self):
        r'''Return the average parameters only when needed.'''
        avg = copy.deepcopy(self.sum)
        
        for key in avg.keys():
            avg[key] = torch.div(self.sum[key], float(self.sum_weight))
        return avg

    def weighted_sum(self, weights, weightclient):
        for key in weights.keys():
            # if we use "w[i][key] *= float(data)", when resnet parameters may have different type that is not float
            weights[key] *= weightclient.type_as(weights[key])
            self.sum[key] += weights[key]
        

def AverageAcrossRounds():
    r'''Average the weights across the all the rounds based on the weight (the first weight is the model parameters) on the fly.
    We only use weights of the last round instead of averaging here.'''
    pass


def getprojectname(args):
    alg_setup = '{}'.format(args.alg)
    # distributed collaborative machine learning 
    dcml_setup = '{}({} {} {})'.format('flv2', args.clients, args.local_epochs, args.global_lr)
    model_setup = '{}'.format(args.arch)
    # data distribution
    if args.distribution in ['iid-b', 'iid-u']:
        dtype_setup = '{}({})'.format(args.d, args.distribution)
    elif args.distribution in ['dir-u', 'pat-u', 'pat2-b']:
        dtype_setup = '{}({} {})'.format(args.d, args.distribution, args.alpha)
    else:
        raise ValueError
    
    if args.optim == 'sgd':
        optim_setup = '{}({} {} {})'.format(args.optim, args.lr, args.momentum, args.weight_decay)
    elif args.optim == 'adam':
        optim_setup = '{}({} {})'.format(args.optim, args.lr, args.weight_decay)
    else:
        raise ValueError
    
    hp_setup = 'hp({})'.format(args.batch_size)

    return '{} {} {} {} {} {}'.format(alg_setup, dcml_setup, model_setup, dtype_setup, optim_setup, hp_setup)


def recorddata(): 
    global args, train_loss, val_loss, train_tops, val_tops

    projectname = getprojectname(args)
    
    # save the output data to .csv file (for comparision plots)   
    record_tocsv(name='{}'.format(projectname), path='../convergence/save/', 
        train_loss=train_loss, val_loss=val_loss, train_top1=train_tops[0], val_top1=val_tops[0], 
        train_top5=train_tops[1], val_top5=val_tops[1])


if __name__ == '__main__':
    main()