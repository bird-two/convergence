def build_model_split(arch='lenet5', dtype='mnist', alg='base', split=2):
    r'''build model for split learning based on arch, dtype, alg, split. 
    All the options are below. 
    arch        dataset(dtype)      split layer
    LeNet5      mnist/fashionmnist  [1,2]
    Vgg11       cifar10/100         [1,2,3,4,5,6]
    '''
    map_dtype_numclass = {'mnist': 10, 'fashionmnist': 10, 'cifar10': 10, 'cifar100': 100, 'test': 4}
    num_classes = map_dtype_numclass[dtype]

    if arch == 'lenet5':
        from .lenet5_mnist_split import lenet5_split
        model_client, model_server = lenet5_split(split=split, dtype=dtype,  num_classes=num_classes)
    elif arch == 'vgg11':
        from .vgg_cifar_split import vgg11_split
        model_client, model_server = vgg11_split(split=split, dtype=dtype,  num_classes=num_classes)
    
    return model_client, model_server


def build_model(arch='lenet5', dtype='mnist', alg='base'):
    r'''build model for federated learning based on arch, dtype, alg.'''
    map_dtype_numclass = {'mnist': 10, 'fashionmnist': 10, 'cifar10': 10, 'cifar100': 100, 'test': 4}
    num_classes = map_dtype_numclass[dtype]

    if arch == 'lenet5':
        from .lenet5_mnist import LeNet5
        model = LeNet5()
    elif arch == 'vgg11':
        from .vgg_cifar import vgg11
        model = vgg11(dtype=dtype, num_classes=num_classes)
    
    return model