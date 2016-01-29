from fuel.datasets.cifar10 import CIFAR10

def data():
    train = CIFAR10(("train",), subset=slice(None, 40000))
    validation = CIFAR10(("train",), subset=slice(40000, None))
    test = CIFAR10(("test",))

    return train, validation, test

