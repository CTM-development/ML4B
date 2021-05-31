'''
MNIST dataset
DataLoader, Transformations
Multilayer Neural Net, activation function
Loss and optimizer
Training loop (batch training)
Model evaluation
GPU support
'''

import torch
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# hyper params
input_size = 28*28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
lr = 0.001

# import MNIST
train_dataset = torchvision.datasets.MNIST(
    root='../../data/MNIST',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
test_dataset = torchvision.datasets.MNIST(
    root='../../data/MNIST',
    train=False,
    transform=transforms.ToTensor()
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

for i in range(6):
    plt.subplot(2, 3, i-1)
    plt.imshow(samples[i][0], cmap='gray')
    plt.show()
