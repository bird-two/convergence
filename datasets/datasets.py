'''
Data preprocessing. MNIST's is from [1]. FashionMNIST is from [5], [6]. 
CIFAR10's is from [2], [3]. CIFAR100's from [4], but we don not do `cutout`. Reference [5] is also helpful.
References:
[1] https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/utils.py
[2] https://github.com/JYWa/FedNova/blob/master/util_v4.py
[3] https://github.com/FedML-AI/FedML/blob/master/python/fedml/data/cifar10/data_loader.py
[4] https://github.com/FedML-AI/FedML/blob/master/python/fedml/data/cifar100/data_loader.py
[5] https://github.com/Lornatang/pytorch-alexnet-cifar100/blob/master/utils/datasets.py
[6] https://github.com/felisat/federated-learning/blob/master/data_utils.py
2022 08 11
'''
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
import torchvision.transforms as transforms

def build_dataset(dtype='mnist', data_path = '../data/'):
    if dtype == 'mnist':
        train_dataset, test_dataset = dataset_mnist(data_path)
    elif dtype == 'fashionmnist':
        train_dataset, test_dataset = dataset_fashionmnist(data_path)
    elif dtype == 'cifar10':
        train_dataset, test_dataset = dataset_cifar10(data_path)
    elif dtype == 'cifar100':
        train_dataset, test_dataset = dataset_cifar100(data_path)
    return train_dataset, test_dataset


def dataset_mnist(data_path):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]) # mean: 0.13066235184669495, std:0.30810782313346863
    train_dataset = MNIST(root=data_path, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=data_path, train=False, download=True, transform=transform)

    return train_dataset, test_dataset


def dataset_fashionmnist(data_path):
    #bd2 mean and std are generated manually
    # https://www.freesion.com/article/77151167393/
    # 2022 08 17
    transform_train = transforms.Compose([
                        transforms.Resize(28),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.2860,), (0.3530,))])
    
    transform_test = transforms.Compose([
                        transforms.Resize(28),
                        transforms.ToTensor(),
                        transforms.Normalize((0.2860,), (0.3530,)),
                        ])

    train_dataset = FashionMNIST(root=data_path, train=True, download=True, transform=transform_train)
    test_dataset = FashionMNIST(root=data_path, train=False, download=True, transform=transform_test)

    return train_dataset, test_dataset


def dataset_cifar10(data_path):
    mean = [0.49139968, 0.48215827, 0.44653124]
    std = [0.24703233, 0.24348505, 0.26158768]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    train_dataset = CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

    return train_dataset, test_dataset


def dataset_cifar100(data_path):
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    train_dataset = CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
    
    return train_dataset, test_dataset


if __name__ == '__main__':
    for i in range(0, 3):
        # to judge if the sample sequence is the same at different times
        train_dataset, test_dataset = dataset_mnist('../data/')
        print(train_dataset.targets[:30])
   
    
    
