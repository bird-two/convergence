'''
LeNet5 for MNIST.
The implementation is mainly from [1]. We only make a little modifications for split. Codes from [2][3] are relevant. 
References: 
[1] https://github.com/ChawDoe/LeNet5-MNIST-PyTorch/blob/master/model.py
[2] https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
[3] https://blog.csdn.net/sunqiande88/article/details/80089941
2022 08 04
'''
import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        y = self.features(x)
        y = self.flatten(y)
        y = self.classifier(y)
        return y


if __name__ == '__main__':
    from pytorch_model_summary import summary
    from model_utils import count_parameters
    model = LeNet5()
    count_parameters(model)
    print(summary(model, torch.zeros((1, 1, 28, 28)), show_input=True)) # 1, 3, 32, 32