'''
VGG for cifar10/cifar100
The implementation and structure of AlexNet is from [1].
The cifar10 hyper parameters are from [2,3], cifar100 from [4].
References: 
[1] https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
[2] https://github.com/JYWa/FedNova/blob/master/models/vgg.py
[3] https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py
[4] https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/vgg.py
'''
from typing import Union, List, Dict, Any, cast
import torch
import torch.nn as nn

def get_params(dtype = 'cifar10'):
    if dtype == 'cifar10':
        p = '512,512,512,512,512'
    elif dtype == 'cifar100':
        p = '512,4096,4096,4096,4096'
    p = p.split(',')
    for i in range(0,len(p)):
        p[i] = int(p[i])
    return p


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)


class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, dtype: str = 'cifar10', num_classes: int = 10, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        p = get_params(dtype=dtype)

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(p[0], p[1]),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(p[2], p[3]),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(p[4], num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(cfg: str, batch_norm: bool, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model


def vgg11(**kwargs: Any) -> VGG:
    return _vgg("A", False, **kwargs)


def vgg11_bn(**kwargs: Any) -> VGG:
    return _vgg("A", True, **kwargs)


def vgg13(**kwargs: Any) -> VGG:
    return _vgg("B", False, **kwargs)


def vgg13_bn(**kwargs: Any) -> VGG:
    return _vgg("B", True, **kwargs)


def vgg16(**kwargs: Any) -> VGG:
    return _vgg("D", False, **kwargs)


def vgg16_bn(**kwargs: Any) -> VGG:
    return _vgg("D", True, **kwargs)


def vgg19(**kwargs: Any) -> VGG:
    return _vgg("E", False, **kwargs)


def vgg19_bn(**kwargs: Any) -> VGG:
    return _vgg("E", True, **kwargs)


if __name__ == '__main__':
    from pytorch_model_summary import summary
    from model_utils import count_parameters
    model = vgg11(dtype='cifar100', num_classes=100)
    count_parameters(model)
    print(summary(model, torch.zeros((1, 3, 32, 32)), show_input=True)) # 1, 3, 32, 32