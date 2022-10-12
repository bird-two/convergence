'''
Split LeNet5(MNIST) to client model and server model.
The split method is from [1].
References:
[1] https://github.com/ZHUANGHP/FDG/blob/master/DG_models.py
2022 08 04
'''
import torch
import torch.nn as nn

# note Comment the two lines if you want to run in this file (__name__ == '__main__').
from .lenet5_mnist import LeNet5
from .model_utils import BuildClient, BuildServer


def splitmodel(model, split):
    r'''Pay attention that `split` \in [1,2], not to go beyond this range.
    Args:
        split: retain `split` block (one or more layers)s in client
    '''
    if split == 1:
        # front net
        net_client = nn.Sequential(model.features[0:3])
        # end net
        net_server = nn.Sequential(model.features[3:6], model.flatten, model.classifier)

    elif split == 2:
        net_client = nn.Sequential(model.features[0:6])
        net_server = nn.Sequential(model.flatten, model.classifier)
        
    model_client = BuildClient(net_client)
    model_server = BuildServer(net_server)

    return model_client, model_server


def lenet5_split(split=2, dtype='mnist', num_classes=10, **kwargs):
    r'''split LeNet5
    Args:
        split: make this argument first, for it is handled here.
        dtype, num_classes: these two args are necessary to constuct AlexNet, so we specify here.
    '''
    model = LeNet5(**kwargs)
    return splitmodel(model, split)


if __name__ == '__main__':
    from pytorch_model_summary import summary
    from lenet5_mnist import LeNet5
    from model_utils import BuildClient, BuildServer, count_parameters
    #from model_utils import count_parameters
    model = LeNet5()
    model1, model2 = lenet5_split(split=2)
    #count_parameters(model)
    print(summary(model1, torch.zeros((1, 1, 28, 28)), show_input=True)) # 1, 3, 32, 32
    print(summary(model, torch.zeros((1, 1, 28, 28)), show_input=True)) # 1, 3, 32, 32