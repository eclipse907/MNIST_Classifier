import multiprocessing
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from random import randint
from custom_datasets import CustomLabeledDataset


class MnistClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        self.pool = nn.MaxPool2d((2, 2))
        self.drop1 = nn.Dropout(p=0.25)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(64 * 12 * 12, 128)
        self.drop2 = nn.Dropout(p=0.5)
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = self.flat(x)
        x = F.relu(self.fc(x))
        x = self.drop2(x)
        x = self.out(x)
        return x


def main():
    directory = './data'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Selected device: " + str(device))
    if not os.path.exists(directory):
        os.mkdir(directory)
    train_set = datasets.MNIST(directory, train=True, transform=transforms.ToTensor(), download=True)
    test_set = datasets.MNIST(directory, train=False, transform=transforms.ToTensor(), download=True)
    num_of_data = int(input("Enter number of data to train on: "))
    if num_of_data == len(train_set):
        train_dataset = train_set
    else:
        indexes = list()
        while len(indexes) < num_of_data:
            index = randint(0, len(train_set) - 1)
            if index not in indexes:
                indexes.append(index)
        train_dataset = CustomLabeledDataset(train_set, indexes)
    batch_size = int(input("Enter desired batch size: "))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               pin_memory=True, num_workers=multiprocessing.cpu_count())
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, pin_memory=True,
                                              num_workers=multiprocessing.cpu_count())

    model = MnistClassifier()
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    print("Started training.")

    for epoch in range(10):
        for i, data in enumerate(train_loader, 1):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            # print("Epoch: %d, batch: %d, loss: %f" % (epoch + 1, i, loss.item()))

    print("Finished Training.")
    print("Evaluating model on test set.")

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Evaluation complete. Returning result.")
    # print("Accuracy of the network on the %d test images: %.2f%%" % (len(test_set), 100 * (correct / total)))
    return len(train_dataset), round(correct / total, 2)
