import multiprocessing
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.distributions.categorical import Categorical
from random import randint
from custom_datasets import CustomLabeledDataset
from custom_datasets import CustomUnlabeledDataset


def least_confidence_sampling(num_of_data_to_add, unlabeled_loader, model, device):
    with torch.no_grad():
        new_indexes = list()
        to_delete = list()
        while len(new_indexes) < num_of_data_to_add:
            for data_unlabeled in unlabeled_loader:
                indexes, images = data_unlabeled[0].to(device), data_unlabeled[1].to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                max_probs, _ = torch.max(probs.data, 1)
                diff = torch.ones_like(max_probs) - max_probs
                _, to_return = torch.topk(diff, 5)
                for index in torch.gather(indexes, 0, to_return).tolist():
                    if len(new_indexes) < num_of_data_to_add:
                        new_indexes.append(index)
                        to_delete.append(index)
                    else:
                        break
                if len(new_indexes) >= num_of_data_to_add:
                    break
            for index in to_delete:
                unlabeled_loader.dataset.list_of_indexes.remove(index)
            to_delete.clear()
        return new_indexes


def margin_sampling(num_of_data_to_add, unlabeled_loader, model, device):
    with torch.no_grad():
        new_indexes = list()
        to_delete = list()
        while len(new_indexes) < num_of_data_to_add:
            for data_unlabeled in unlabeled_loader:
                indexes, images = data_unlabeled[0].to(device), data_unlabeled[1].to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                two_biggest_maxs, _ = torch.topk(probs, 2, dim=1)
                diff = torch.abs(two_biggest_maxs[:, 0] - two_biggest_maxs[:, 1])
                _, to_return = torch.topk(diff, 5, largest=False)
                for index in torch.gather(indexes, 0, to_return).tolist():
                    if len(new_indexes) < num_of_data_to_add:
                        new_indexes.append(index)
                        to_delete.append(index)
                    else:
                        break
                if len(new_indexes) >= num_of_data_to_add:
                    break
            for index in to_delete:
                unlabeled_loader.dataset.list_of_indexes.remove(index)
            to_delete.clear()
        return new_indexes


def entropy_sampling(num_of_data_to_add, unlabeled_loader, model, device):
    with torch.no_grad():
        new_indexes = list()
        to_delete = list()
        while len(new_indexes) < num_of_data_to_add:
            for data_unlabeled in unlabeled_loader:
                indexes, images = data_unlabeled[0].to(device), data_unlabeled[1].to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                entropy = Categorical(probs=probs).entropy()
                _, to_return = torch.topk(entropy, 5)
                for index in torch.gather(indexes, 0, to_return).tolist():
                    if len(new_indexes) < num_of_data_to_add:
                        new_indexes.append(index)
                        to_delete.append(index)
                    else:
                        break
                if len(new_indexes) >= num_of_data_to_add:
                    break
            for index in to_delete:
                unlabeled_loader.dataset.list_of_indexes.remove(index)
            to_delete.clear()
        return new_indexes


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


def main(sampling_parameter=None):
    directory = './data'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Selected device: " + str(device))
    if not os.path.exists(directory):
        os.mkdir(directory)
    train_set = datasets.MNIST(directory, train=True, transform=transforms.ToTensor(), download=True)
    test_set = datasets.MNIST(directory, train=False, transform=transforms.ToTensor(), download=True)
    seed_size = int(input("Enter desired number of data in initial seed: "))
    train_test_batch_size = int(input("Enter size of batch for training and testing: "))
    unlabeled_batch_size = int(input("Enter size of batch from which to add unlabeled data: "))
    num_of_data_to_add= int(input("Enter number of new data to add to the train set in each epoch: "))
    while True:
        if not sampling_parameter:
            method = input("Choose sampling method for unlabeled data from the available methods:\n"
                           "1 - Least Confidence Sampling\n"
                           "2 - Margin Sampling\n"
                           "3 - Entropy Sampling\n"
                           "Enter one of the numbers: ")
        else:
            method = sampling_parameter
        try:
            number = int(method)
            if number == 1:
                sampling_method = least_confidence_sampling
                break
            elif number == 2:
                sampling_method = margin_sampling
                break
            elif number == 3:
                sampling_method = entropy_sampling
                break
            else:
                raise ValueError
        except ValueError:
            print("Wrong argument entered.")
    labeled_data_indexes = list()
    while len(labeled_data_indexes) < seed_size:
        index = randint(0, len(train_set) - 1)
        if index not in labeled_data_indexes:
            labeled_data_indexes.append(index)
    labeled_dataset = CustomLabeledDataset(train_set, labeled_data_indexes)
    unlabeled_data_indexes = list()
    for index in range(0, len(train_set)):
        if index not in labeled_data_indexes:
            unlabeled_data_indexes.append(index)
    unlabeled_dataset = CustomUnlabeledDataset(train_set, unlabeled_data_indexes)
    labeled_loader = data.DataLoader(labeled_dataset, batch_size=train_test_batch_size, shuffle=True, pin_memory=True,
                                     num_workers=multiprocessing.cpu_count())
    unlabeled_loader = data.DataLoader(unlabeled_dataset, batch_size=unlabeled_batch_size, shuffle=True,
                                       num_workers=multiprocessing.cpu_count())
    test_loader = data.DataLoader(dataset=test_set, batch_size=train_test_batch_size, pin_memory=True,
                                  num_workers=multiprocessing.cpu_count())

    model = MnistClassifier()
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    print("Started training on initial seed.")
    for data_labeled in labeled_loader:
        inputs, labels = data_labeled[0].to(device), data_labeled[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
    print("Completed training on initial seed.")

    sizes_of_data = list()
    accuracies = list()
    for epoch in range(10):
        print("Started training in epoch %d." % (epoch + 1))
        if len(unlabeled_dataset) > 0:
            labeled_dataset.list_of_indexes += sampling_method(num_of_data_to_add, unlabeled_loader, model, device)
        for i, data_labeled in enumerate(labeled_loader, 1):
            inputs, labels = data_labeled[0].to(device), data_labeled[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            # print("Epoch: %d, batch: %d, loss: %f" % (epoch + 1, i, loss.item()))
        print("Finished training in epoch %d." % (epoch + 1))
        print("Evaluating model on test set.")
        correct = 0
        total = 0
        with torch.no_grad():
            for test_data in test_loader:
                images, labels = test_data[0].to(device), test_data[1].to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(probs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        sizes_of_data.append(len(labeled_dataset))
        accuracies.append(round(correct / total, 2))
        print("Evaluation complete.")
        # print("Accuracy of model is: %.2f%%" % (100 * (correct / total)))

    return sizes_of_data, accuracies
