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
import sys

from torch.utils.tensorboard import SummaryWriter

# device config
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
# SummaryWriter
writer =  SummaryWriter("../runs/pytorchTut_mnist")

# hyper params
input_size = 28 * 28
hidden_size = 100
num_classes = 10
num_epochs = 10
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
example_data, example_targets = examples.next()

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(example_data[i][0], cmap='gray')
# plt.show()

img_grid = torchvision.utils.make_grid(example_data)
writer.add_image('mnist_images', img_grid)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, 50)
        self.l3 = nn.Linear(50, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out


model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

writer.add_graph(model, example_data.reshape(-1, 28*28).to(device))

# training loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28 * 1).to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, step {i + 1} / {n_total_steps}, loss = {loss.item():.4f}')

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28 * 1).to(device)
        labels = labels.to(device)

        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100 * (n_correct / n_samples)

    print(f'accuracy =', acc)
