import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

transform = T.Compose([T.ToTensor()])
traindata = datasets.MNIST(root='./data', train=True,download=True,transform=transform)
trainloader = DataLoader(traindata,batch_size = 64)

def get_loader(args_dataset):
    if args_dataset == "mnist":
        train_dataloader = DataLoader(
            datasets.MNIST(root='./data', train=True, download=True, transform=transform),
            batch_size = 64,
            shuffle=True
            )
        val_dataloader = DataLoader(
            datasets.MNIST(root='./data', train=False, download=True, transform=transform),
            batch_size = 64,
            shuffle=False
            )
    if args_dataset == "cifar10":
        train_dataloader = DataLoader(
            datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
            batch_size = 64,
            shuffle=True
            )
        val_dataloader = DataLoader(
            datasets.CIFAR10(root='./data', train=False, download=True, transform=transform),
            batch_size = 64,
            shuffle=False
            )
    else:
        raise NotImplementedError
    return train_dataloader, val_dataloader


