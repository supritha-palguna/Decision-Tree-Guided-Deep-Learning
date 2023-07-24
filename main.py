from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class Net(nn.Module):
    def __init__(self, n_hidden):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, model_name):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Model: {} \nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(model_name,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def get_indices(arr):
    left_index = [] 
    right_index = []
    for i in range(len(arr)):
        if arr[i] == 1:
            left_index.append(i)
        elif arr[i] == 2:
            right_index.append(i)
    return left_index, right_index

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # DT for train data
    dt = DecisionTreeClassifier(max_depth=1)
    dt.fit(dataset1.data.view(dataset1.data.size(0), -1), dataset1.targets)

    leaf_nodes = dt.apply(dataset1.data.view(dataset1.data.size(0), -1))

    from collections import Counter
    
    print("Labels: ", Counter(leaf_nodes))
        
    print(np.unique(leaf_nodes))
    left_leaf_nodes, right_leaf_nodes = get_indices(leaf_nodes)

    left_dataset = torch.utils.data.Subset(dataset1, left_leaf_nodes)
    right_dataset = torch.utils.data.Subset(dataset1, right_leaf_nodes)

    left_loader = torch.utils.data.DataLoader(left_dataset,**train_kwargs)
    right_loader = torch.utils.data.DataLoader(right_dataset,**train_kwargs)
    
   
    test_leaf_nodes = dt.apply(dataset2.data.view(dataset2.data.size(0), -1))
    print("Labels: ", Counter(test_leaf_nodes))
    print(np.unique(test_leaf_nodes))
    test_left_leaf_nodes, test_right_leaf_nodes = get_indices(test_leaf_nodes) 

    test_left_dataset = torch.utils.data.Subset(dataset2, test_left_leaf_nodes)
    test_right_dataset = torch.utils.data.Subset(dataset2, test_right_leaf_nodes)

    test_left_loader = torch.utils.data.DataLoader(test_left_dataset,**train_kwargs)
    test_right_loader = torch.utils.data.DataLoader(test_right_dataset,**train_kwargs)
    
    
    model = Net(128).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    model_left = Net(64).to(device)
    optimizer1 = optim.Adadelta(model_left.parameters(), lr=args.lr)
    model_right = Net(64).to(device)
    optimizer2 = optim.Adadelta(model_right.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, 'Whole model')
        scheduler.step()

    scheduler1 = StepLR(optimizer1, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model_left, device, left_loader, optimizer1, epoch)
        test(model_left, device, test_left_loader, 'Left DT model')
        scheduler1.step()

    scheduler2 = StepLR(optimizer2, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model_right, device, right_loader, optimizer2, epoch)
        test(model_right, device, test_right_loader, 'Right DT model')
        scheduler2.step()

    test(model, device, test_loader, 'Whole model')
    test(model_left, device, test_left_loader, 'Left DT model')
    test(model_right, device, test_right_loader, 'Right DT model')

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()