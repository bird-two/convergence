'''
Split VGG(CIFAR10/100) to client model and server model.
The split method is from [1].
References:
[1] https://github.com/ZHUANGHP/FDG/blob/master/DG_models.py
2022 05 01
'''
import torch
import torch.nn as nn
from .vgg_cifar import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn 
from .model_utils import BuildClient, BuildServer

def splitmodel(model, split):
    r'''Pay attention that `split` \in [1,2,3,4,5,6], not to go beyond this range.
    Args:
        split: retain `split` block (one or more layers)s in client
    '''
    if split == 1:
        # front net
        net_client = nn.Sequential(model.features[0:3])
        # end net
        net_server = nn.Sequential(model.features[3:], model.avgpool, model.flatten, model.classifier)
    elif split == 2:
        net_client = nn.Sequential(model.features[0:6])
        net_server = nn.Sequential(model.features[6:], model.avgpool, model.flatten, model.classifier)
    elif split == 3:
        net_client = nn.Sequential(model.features[0:8])
        net_server = nn.Sequential(model.features[8:], model.avgpool, model.flatten, model.classifier)
    elif split == 4:
        net_client = nn.Sequential(model.features[0:11])
        net_server = nn.Sequential(model.features[11:], model.avgpool, model.flatten, model.classifier)
    elif split == 5:
        net_client = nn.Sequential(model.features[0:13])
        net_server = nn.Sequential(model.features[13:], model.avgpool, model.flatten, model.classifier)
    elif split == 6:
        net_client = nn.Sequential(model.features[0:16])
        net_server = nn.Sequential(model.features[16:], model.avgpool, model.flatten, model.classifier)
        
    model_client = BuildClient(net_client)
    model_server = BuildServer(net_server)

    return model_client, model_server


def vgg11_split(split, dtype, num_classes, **kwargs):
    r'''split vgg11
    Args:
        split: make this argument first, for it is handled here.
        dtype, num_classes: these two args are necessary to constuct VGG, so we specify here.
    '''
    model = vgg11(dtype=dtype, num_classes=num_classes, **kwargs)
    return splitmodel(model, split)


def vgg11_bn_split(split, dtype, num_classes, **kwargs):
    r'''split vgg11_bn
    Args:
        split: make this argument first, for it is handled here.
        dtype, num_classes: these two args are necessary to constuct VGG, so we specify here.
    '''
    model = vgg11_bn(dtype=dtype, num_classes=num_classes, **kwargs)
    return splitmodel(model, split)

#bd2 in our experiments, we only use vgg11, so other vgg models are commented to avoid being mixed. 2022 05 02
# def vgg13_split(split, dtype, num_classes, **kwargs):
#     r'''split alexnet
#     Args:
#         split: make this argument first, for it is handled here.
#         dtype, num_classes: these two args are necessary to constuct AlexNet, so we specify here.
#     '''
#     model = vgg13(dtype=dtype, num_classes=num_classes, **kwargs)
#     return splitmodel(model, split)


# def vgg13_bn_split(split, dtype, num_classes, **kwargs):
#     r'''split alexnet
#     Args:
#         split: make this argument first, for it is handled here.
#         dtype, num_classes: these two args are necessary to constuct AlexNet, so we specify here.
#     '''
#     model = vgg13_bn(dtype=dtype, num_classes=num_classes, **kwargs)
#     return splitmodel(model, split)


if __name__ == '__main__':
    from pytorch_model_summary import summary
    from model_utils import count_parameters
    
    model_client = vgg11_split(dtype='cifar10', num_classes=10, split=4)
    model_server = vgg11_split(dtype='cifar10', num_classes=10, split=4)
    #print(model_client)
    print(model_client.model)
    count_parameters(model_client.model)

    #print(summary(model_client, torch.zeros((1, 3, 32, 32)), show_input=True)) # 1, 3, 32, 32
    #print(summary(model_server, torch.zeros((1, 256, 4, 4)), show_input=True)) # 1, 256, 4, 4

