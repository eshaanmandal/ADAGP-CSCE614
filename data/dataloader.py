import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

def get_dataloader(batch_size=64, validation_split=0.1):
    # Define transformation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load the CIFAR-10 dataset
    full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Calculate sizes for train and validation sets
    num_train = int((1 - validation_split) * len(full_train_dataset))
    num_val = len(full_train_dataset) - num_train
    
    # Split the dataset
    train_dataset, val_dataset = random_split(full_train_dataset, [num_train, num_val])
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Test dataset and loader
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
