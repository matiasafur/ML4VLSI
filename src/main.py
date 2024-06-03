import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from models import conv_net
from plot_training import plot_training


# On Windows:
if __name__ == '__main__':
    # Parameters
    batch_size = 64
    epochs = 10

    dataset_path = os.path.join(os.getcwd(), '..', 'data')

    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)

    # Transformations
    # Training
    transform_training = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )

    # Test
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )

    # Dataset
    train_set = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform_test)
    test_set = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(train_set, batch_size, shuffle=False, num_workers=2)

    # Device selection
    classes = train_set.classes

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("Cuda is not available")
        exit(0)

    print("Device: ", device)

    net = conv_net.ConvNet()
    net = net.to(device)

    info = summary(net, (batch_size, 3, 32, 32))
    print(info)

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)

    training_progress = {
        "train_accuracy": [],
        "test_accuracy": [],
        "train_loss": [],
        "test_loss": [],
        "epoch_count": [],
    }

    # Training the network
    for epoch in range(epochs):
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        accu_train = 0
        loss_train = 0
        count = 0

        # Train Loop
        net.train()
        for i, data in enumerate(train_loader, 0):
            # data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()

            _, train_predicted = torch.max(outputs.data, 1)

            train_total += labels.size(0)
            train_correct += (train_predicted == labels).sum().item()

            running_loss += loss.item()
            count += 1

            accu_train = 100 * train_correct / train_total
            loss_train = running_loss / len(train_loader)

        # Saving training variables
        training_progress["train_accuracy"].append(accu_train)
        training_progress["train_loss"].append(loss_train)

        # Test loop
        correct = 0
        total = 0
        running_loss = 0.0

        net.eval()
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

        # Printing progress
        if epoch == 0:
            print(90 * '=')
            print('TRAINING STARTED')
            print(90 * '=')
        print(' ' * 7 + '| Train Accuracy: {:6.2f}% | Test Accuracy: {:6.2f}%'.format(
            training_progress['train_accuracy'][-1], training_progress['test_accuracy'][-1]))
        print('  {:3d}  |'.format(epoch + 1) + 82 * '-')
        print(
            ' ' * 7 + '| Train Loss:      {:6.4f} | Test Loss:      {:6.4f}'.format(
                training_progress['train_loss'][-1],
                training_progress['test_loss'][-1]))
        print(90 * '=')
    print('TRAINING FINISHED')
    print(90 * '=')
    plot_training(training_progress)
