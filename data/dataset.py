import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import medmnist
from medmnist import INFO, Evaluator

transform = T.Compose([T.ToTensor()])
traindata = datasets.MNIST(root='./data', train=True,
                           download=True, transform=transform)
trainloader = DataLoader(traindata, batch_size=64)


def get_loader(args_dataset):
    if args_dataset == "mnist":
        train_dataloader = DataLoader(
            datasets.MNIST(root='./data', train=True,
                           download=True, transform=transform),
            batch_size=64,
            shuffle=True
        )
        val_dataloader = DataLoader(
            datasets.MNIST(root='./data', train=False,
                           download=True, transform=transform),
            batch_size=64,
            shuffle=False
        )
    elif args_dataset == "cifar10":
        train_dataloader = DataLoader(
            datasets.CIFAR10(root='./data', train=True,
                             download=True, transform=transform),
            batch_size=64,
            shuffle=True
        )
        val_dataloader = DataLoader(
            datasets.CIFAR10(root='./data', train=False,
                             download=True, transform=transform),
            batch_size=64,
            shuffle=False
        )
    elif args_dataset == "medmnist":

        data_flag = 'dermamnist'
        # data_flag = 'breastmnist'
        download = True

        info = INFO[data_flag]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])

        DataClass = getattr(medmnist, info['python_class'])

        # preprocessing
        data_transform = T.Compose([
            T.ToTensor(),
            # transforms.Lambda(lambda x: x.permute(1, 2, 0)),
            T.Normalize(mean=[.5], std=[.5])
        ])

        # load the data
        train_dataset = DataClass(
            split='train', transform=data_transform, download=download, size=224)
        test_dataset = DataClass(
            split='test', transform=data_transform, download=download, size=224)

        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(
            dataset=test_dataset, batch_size=64, shuffle=False)
    else:
        raise NotImplementedError
    return train_dataloader, val_dataloader
