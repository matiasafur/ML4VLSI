# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------------------------------------------------
Copyright (c) 2021 - 2023
------------------------------------------------------------------------------------------------------------------------
@Author: Diego Gigena Ivanovich - diego.gigena-ivanovich@silicon-austria.com
@File:   tutorial.py
@Time:   2/27/2023 - 9:55 PM
@IDE:    PyCharm
@desc:
------------------------------------------------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
from torchinfo import summary
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from models import Net
import os


if __name__ == '__main__': # -> Necesario solo para ejecutar en windows.
    ########################################################################################################################
    # Parámetros
    ########################################################################################################################
    batch_size = 10
    epochs = 10

    ########################################################################################################################

    dataset_path = os.path.join(os.getcwd(), '..', 'data')

    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)

    ########################################################################################################################
    # Transformaciones
    ########################################################################################################################
    # Es una cadena de operaciones que se aplica a cada muestra del dataset. En este caso, las imágenes se transforman a
    # tensor, y luego se normalizan con media y desviación estándar de 0.5

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))])


    train_set = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root=dataset_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = train_set.classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    net = Net()
    net = net.to(device)

    info = summary(net, (batch_size, 1, 28, 28))
    print(info)
    ########################################################################################################################
    # Optimizer
    ########################################################################################################################
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001) #e-1

    ########################################################################################################################
    # Defino diccionario para almacenar el progreso del entrenamiento

    training_progress = {'train_accuracy': [],
                         'test_accuracy': [],
                         'train_loss': [],
                         'test_loss': [],
                         'epoch_count': []}


    for epoch in range(epochs):  # loop over the dataset multiple times
        # Inicializo variables cada vez que comienza una nueva época
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        count = 0

        accu_train = 0
        loss_train = 0
        ####################################################################################################################
        # Train Loop
        ####################################################################################################################
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            _, train_predicted = torch.max(outputs.data, 1)

            train_total += labels.size(0)
            train_correct += (train_predicted == labels).sum().item()

            running_loss += loss.item()
            count += 1

            accu_train = 100 * train_correct / train_total
            loss_train = running_loss / len(train_loader)

        # Guardo variables de entrenamiento
        training_progress['train_accuracy'].append(accu_train)
        training_progress['train_loss'].append(loss_train)
        ####################################################################################################################
        # Test Loop
        ####################################################################################################################
        correct = 0
        total = 0
        running_loss = 0.0

        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)

                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accu_test = 100 * correct / total
            loss_test = running_loss / len(test_loader)

        training_progress['test_accuracy'].append(accu_test)
        training_progress['test_loss'].append(loss_test)
        training_progress['epoch_count'].append(epoch + 1)
        ####################################################################################################################
        # Printing progress
        if epoch == 0:
            print(90 * '=')
            print('TRAINING STARTED')
            print(90 * '=')
        print(' ' * 7 + '| Train Accuracy: {:6.2f}% | Test Accuracy: {:6.2f}%'.format(
            training_progress['train_accuracy'][-1], training_progress['test_accuracy'][-1]))
        print('  {:3d}  |'.format(epoch + 1) + 82 * '-')
        print(
            ' ' * 7 + '| Train Loss:      {:6.4f} | Test Loss:      {:6.4f}'.format(training_progress['train_loss'][-1],
                                                                                    training_progress['test_loss'][-1]))
        print(90 * '=')
    print('TRAINING FINISHED')
    print(90 * '=')

