'''
The implementation of Split Learning[1].
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

from utils.utils import AverageMeter, accuracy
from utils.outputs import record_tocsv, switchonlog, loginfo
from datasets.datasets import build_dataset
from datasets.distributions import SubDataset, build_distribution
from models.build_models import build_model_split


import argparse
parser = argparse.ArgumentParser()
model_names = ['lenet5', 'vgg11']
alg_names = ['base']
parser.add_argument('-a', '--arch', metavar='ARCH', default='lenet5', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: vgg11)')
parser.add_argument('-d', default='cifar10', type=str, metavar='N', choices=['mnist', 'fashionmnist', 'cifar10', 'cifar100'], help='dataset')
parser.add_argument('-s', default=2, type=int, metavar='N', help='split layer')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--local-epochs', default=1, type=int, metavar='N', help='number of local epochs to run')
parser.add_argument('--clients', default=100, type=int, metavar='N', help='number of total clients')
parser.add_argument('--servers', default=1, type=int, metavar='N', help='number of total servers')
parser.add_argument('--alg', default='base', type=str, choices=alg_names, metavar='N', help='the algorithm')
parser.add_argument('--distribution', default='iid-b', type=str, metavar='N', choices=['dir-u', 'iid-b', 'pat2-b'], help='the data distribution')
parser.add_argument('--alpha', default=10, type=float, metavar='N', help='the alpha of dirichlet distribution')
parser.add_argument('--optim', default='sgd', type=str, choices=['sgd', 'adam'], metavar='N', help='the optimizer')
parser.add_argument('--lr', default=0.0001, type=float, metavar='LR', help='initial learning rate of client and server')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='N', help='weight decay of client optimizer')
parser.add_argument('--momentum', default=0, type=float, metavar='N', help='momentum of client optimizer')
parser.add_argument('--global-lr', default=1, type=float, metavar='LR', help='global learning rate, i.e. `\eta_g` of the paper')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=1234, type=int, help='seed for initializing training. ')
args = parser.parse_args()

# python sl.py -a lenet5 -d mnist -s 2 --epochs 50 --local-epochs 1 --clients 100 --distribution pat2-b --alpha 2 --optim sgd --lr 0.01 --momentum 0.9 --global-lr 1 --batch-size 10

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

# collect server's loss and tops
topacc = [1, 5]
train_loss = []
val_loss = []
train_tops = [[] for _ in range(len(topacc))]
val_tops = [[] for _ in range(len(topacc))]


def main():
    global args, topacc, device
    # log
    switchonlog(logname=getprojectname(args))
    loginfo('args: {}\ndevice: {}'.format(args, device))
    
    # datasets and generate train local datasets
    train_dataset, test_dataset = build_dataset(args.d)
    
    train_loaders = []
    localdataset_idxs_train = build_distribution(dtype=args.d, num_users=users, way=args.distribution, alpha=args.alpha)
    for i in range(users):
        train_loaders.append(DataLoader(SubDataset(train_dataset, localdataset_idxs_train[i]), batch_size=args.batch_size, shuffle=True))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
    # Dataset size for the client-size -- client wise averaging   
    datasetsize_client = []
    for i in range(users):
        datasetsize_client.append(len(localdataset_idxs_train[i]))  
    loginfo('datasetsize_client\n{}'.format(datasetsize_client.__str__()))

    # model
    global_model_client, global_model_server = build_model_split(arch=args.arch, dtype=args.d, alg=args.alg, split=args.s)
    global_model_client.to(device)
    global_model_server.to(device)
    loginfo('model_client{}\nmodel_server{}'.format(global_model_client, global_model_server))

    # loss function
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()    # store start time
    print("timmer start!")

    global_model_client.train()
    global_model_server.train()
    
    round_model_client = copy.deepcopy(global_model_client)
    round_model_server = copy.deepcopy(global_model_server)
    
    for r in range(rounds):
        #note round_model_client = global_model_client.load_state_dict(global_model_client.state_dict())
        # https://github.com/lukemelas/EfficientNet-PyTorch/issues/119
        #2022 08 11
        round_model_client.load_state_dict(global_model_client.state_dict())
        round_model_server.load_state_dict(global_model_server.state_dict())
        for u in range(users):
            localupdate(train_loaders[u], model_client=round_model_client, model_server=round_model_server, criterion=criterion, device=device, args=args)
        
        # global update, use `\eta_g` in the paper.   
        w_client, w_server = globalupdate(global_model_client.state_dict(), global_model_server.state_dict(), 
                                            round_model_client.state_dict(), round_model_server.state_dict(), args)
        global_model_client.load_state_dict(w_client)
        global_model_server.load_state_dict(w_server)
        
        # Train accuracy and loss check at each round (this is for the gloabl model -- not local model)
        inference(train_loaders, global_model_client, global_model_server, criterion, r)
        # Test accuracy and loss evaluate at each round
        evaluate(test_loader, global_model_client, global_model_server, criterion, r)

    end_time = time.time()  # store end time
    loginfo("TrainingTime: {} sec".format(end_time - start_time))
    recorddata()


#def localupdate(local_epochs, train_loaders, model_client, model_server, optimizer, criterion, device):
def localupdate(train_loader, model_client, model_server, criterion, device, args):
    r'''local_epoch `E` is the number of training passes each client makes over its local dataset on each round [1].
    It is worth noting that `K = E(n_k/B)`, 
    where `K` denotes the local steps defined in our paper, `n_k` is the local examples of client `k`, `B` denotes the Mini-batch size.
    '''
    if args.optim == 'adam':
        optimizer_client = Adam(model_client.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_server = Adam(model_server.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer_client = SGD(model_client.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer_server = SGD(model_server.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    model_client.train()
    model_server.train()
    
    for local_epoch in range(local_epochs):
        for inputs, labels in train_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs = inputs.to(device)
            labels = labels.to(device)

            #------------ client forward training -----------
            optimizer_client.zero_grad()
            intermediate = model_client(inputs)
            # send to server
            activation = intermediate.clone().detach().requires_grad_(True)
                
            # ---------- server forward/backward training ------------
            optimizer_server.zero_grad()
            output = model_server(activation)
            loss = criterion(output, labels)
            loss.backward()
            client_grad = activation.grad.clone().detach()
            optimizer_server.step()

            # ----------- client backward training ------------
            intermediate.backward(client_grad)
            optimizer_client.step()


def globalupdate(w_client, w_server, round_client_weight, round_server_weight, args):
    r'''Update the global model with the round model and `eta_g` (see algorithm in our paper).
    Two conditions are adopted here, because if using the second, true but different result will be gotten (maybe caused by float caculation of computer).'''
    global_lr = torch.tensor(args.global_lr)
    if global_lr == 1.0: # Although float, "==" is ok.
        # update the global client model
        for key in w_client.keys():
            w_client[key] = round_client_weight[key]
        # update the global server model
        for key in w_server.keys():
            w_server[key] = round_server_weight[key]
    else:
        # update the global client model
        for key in w_client.keys():
            w_client[key] = w_client[key] + global_lr*(round_client_weight[key]-w_client[key])
        # update the global server model
        for key in w_server.keys():
            w_server[key] = w_server[key] + global_lr*(round_server_weight[key]-w_server[key])

    return w_client, w_server


def inference(train_loaders, model_client, model_server, criterion, round):
    r'''compute training accuracy and loss.'''
    global users, train_loss, train_tops, topacc

    model_client.eval()
    model_server.eval()

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
                trn_label = trn_label.clone().detach().long().to(device)

                # client model forward
                trn_intermediate = model_client(trn_x)
                activation = trn_intermediate.clone().detach()

                # server model forward
                trn_output = model_server(activation)
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


def evaluate(val_loader, model_client, model_server, criterion, round):
    global val_loss, val_tops, topacc # to record servers' loss and tops
    
    model_client.eval()
    model_server.eval()

    # test acc
    with torch.no_grad():
        losses = AverageMeter()
        tops = [AverageMeter() for _ in range(0, len(topacc))]
        
        for j, val in enumerate(val_loader):
            val_x, val_label = val
            val_x = val_x.to(device)
            val_label = val_label.clone().detach().long().to(device)

            # client forward
            val_intermediate = model_client(val_x)
            activation = val_intermediate.clone().detach()

            # server forward
            val_output = model_server(activation)
            loss = criterion(val_output, val_label)
            acc1, acc5 = accuracy(val_output.data, val_label.data, topk=topacc)
            losses.update(loss.item(), val_label.size(0))
            tops[0].update(acc1.item(), val_label.size(0))
            tops[1].update(acc5.item(), val_label.size(0))
        
        loginfo("rounds {}'s server test  acc    : {:.2f}%, test  loss    : {:.4f}".format(round + 1, tops[0].avg, losses.avg), 'red')
        
        val_loss.append(losses.avg)
        for i in range(0, len(topacc)):
            val_tops[i].append(tops[i].avg)


def getprojectname(args):
    alg_setup = '{}'.format(args.alg)
    # distributed collaborative machine learning
    dcml_setup = '{}({} {} {})'.format('sl', args.clients, args.local_epochs, args.global_lr)
    model_setup = '{}({})'.format(args.arch, args.s)
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
    record_tocsv(name='{}'.format(projectname),
        train_loss=train_loss, val_loss=val_loss, train_top1=train_tops[0], val_top1=val_tops[0], 
        train_top5=train_tops[1], val_top5=val_tops[1])


if __name__ == '__main__':
    main()
