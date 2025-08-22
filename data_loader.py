import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# Define data loader function
def get_CIFAR10(train_ratio=0.9, batch_size=128):
    temp_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    loader = DataLoader(temp_set, batch_size=len(temp_set), shuffle=False)
    temp_data = next(iter(loader)) # temp_data[0]은 X, temp_data[1]은 label
    mean = temp_data[0].mean(dim=[0, 2, 3])
    std = temp_data[0].std(dim=[0, 2, 3]) # 관례적인 std 값 = (0.2023, 0.1994, 0.2010)

    # for Train set
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # for Valid, Test set
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    valset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_size = int(len(trainset) * train_ratio)
    val_size = len(trainset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_subset, _ = random_split(trainset, [train_size, val_size], generator=generator)
    _, val_subset = random_split(valset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader