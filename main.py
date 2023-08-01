from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from torchinfo import summary
import numpy as np

def model_creator(n_hidden, output, dataset):
    if dataset == 'mnist' or dataset == 'fashion':
        return Net(n_hidden)
    elif dataset == 'cifar10' or dataset == 'cifar100':
        return Cifar_Net(n_hidden, output)
    

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
    

class Cifar_Net(nn.Module):
    def __init__(self, n_hidden, output):
        super(Cifar_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, n_hidden)
        self.fc2 = nn.Linear(n_hidden, output)

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
          
class NN_DT_classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, train_kwargs, test_kwargs, args, device, my_net):
        self.dt_model = DecisionTreeClassifier(max_depth=args.depth)
        self.device = device
        self.train_kwargs = train_kwargs
        self.test_kwargs = test_kwargs
        self.args = args
        self.models = None
        self.net = my_net

    def fit(self, X, y):
        original_data = X
        if type(X.data) is np.ndarray:
             X = torch.stack([torch.Tensor(x) for x, _ in X], dim=0)
        data = X.data
        reshaped_data = data.view(data.size(0), -1)
        print(reshaped_data.size())  
       
        self.dt_model.fit(reshaped_data, y)
        leaf_nodes = self.dt_model.apply(reshaped_data)

        n_models = len(np.unique(leaf_nodes))
        #print(Net(128 // pow(2, self.args.depth)))
        print(self.net)
        
        #self.models = [Net(128 // pow(2, self.args.depth)).to(self.device) for _ in range(n_models)]
        self.models = [self.net.to(self.device) for _ in range(n_models)]

        for index, leaf in enumerate(np.unique(leaf_nodes)):
            indices = get_indices(leaf_nodes, leaf)
            model = self.models[index]
            print(summary(model))
            optimizer = optim.Adadelta(model.parameters(), lr=self.args.lr)
            scheduler = StepLR(optimizer, step_size=1, gamma=self.args.gamma)
            for epoch in range(1, self.args.epochs + 1):
                train(self.args, model, self.device, torch.utils.data.Subset(original_data, indices), optimizer, epoch, self.train_kwargs)
                scheduler.step()
        return self


    def predict(self, X):
        original_data = X
        if type(X.data) is np.ndarray:
             X = torch.stack([torch.Tensor(x) for x, _ in X], dim=0)
        test_leaf_nodes = self.dt_model.apply(X.data.view(X.data.size(0), -1))
        
        for index, leaf in enumerate(np.unique(test_leaf_nodes)):
            indices = get_indices(test_leaf_nodes, leaf)
            test(self.models[index], self.device, torch.utils.data.Subset(original_data, indices), f"{index+1}", self.test_kwargs)


def get_indices(arr, value):
        indices = []
        for i in range(len(arr)):
            if arr[i] == value:
                indices.append(i)
        return indices


def train(args, model, device, train_loader, optimizer, epoch, train_kwargs):
    train_loader = torch.utils.data.DataLoader(train_loader, **train_kwargs)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, model_name, test_kwargs):
    test_loader = torch.utils.data.DataLoader(test_loader,**test_kwargs)
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

def get_dataset(dataset):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    output = 10
    if(dataset == 'mnist'):
            X = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
            Y = datasets.MNIST('./data', train=False,
                       transform=transform)
    elif(dataset == 'fashion'):
            X = datasets.FashionMNIST('./data', train=True, download=True,
                       transform=transform)
            Y = datasets.FashionMNIST('./data', train=False,
                       transform=transform)
    elif(dataset == 'cifar10'):
            X = datasets.CIFAR10('./data', train=True, download=True,
                       transform=transform)
            Y = datasets.CIFAR10('./data', train=False,
                       transform=transform)
    elif(dataset == 'cifar100'):
            X = datasets.CIFAR100('./data', train=True, download=True,
                       transform=transform)
            Y = datasets.CIFAR100('./data', train=False,
                       transform=transform)
            output = 100
    
    return X, Y, output

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
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
    parser.add_argument("-de", "--depth", help="Depth of decision tree", type=int, default=1)
    parser.add_argument("-d", "--dataset", help="Dataset used for experiments. ",type=str, default=["mnist"], nargs='+')
    
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

    for dataset in args.dataset:
        X, Y, output = get_dataset(dataset)
        # model = Cifar_Net(128, 100).to(device)
        # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
   
        # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        # for epoch in range(1, args.epochs + 1):
        #     train(args, model, device, X, optimizer, epoch, train_kwargs)
        #     scheduler.step()

        # test(model, device, Y, 'Whole model', test_kwargs)
        n_hidden=128
        my_net = model_creator(n_hidden // pow(2, args.depth), output, dataset)

        dt = NN_DT_classifier(train_kwargs, test_kwargs, args, device, my_net)
        dt.fit(X, X.targets)
        dt.predict(Y)
        
        
    
    
    


    # DT for train data
    # depth = args.depth

    # dt = DT_classifier(max_depth=depth)
    # dt.fit(dataset1.data.view(dataset1.data.size(0), -1), dataset1.targets)
    
    # leaf_nodes = dt.predict(dataset1.data.view(dataset1.data.size(0), -1))
    # test_leaf_nodes = dt.predict(dataset2.data.view(dataset2.data.size(0), -1))

    # from collections import Counter
    
    # print("Labels: ", Counter(leaf_nodes))
    # print("Labels: ", Counter(test_leaf_nodes))

    # train_leaves = np.unique(leaf_nodes)

    # n_hidden = 128//pow(2, depth)
    # for index, leaf in enumerate(train_leaves):
    #     indices = get_indices(leaf_nodes, leaf)
    #     train_data = torch.utils.data.Subset(dataset1, indices)
    #     train_loader = torch.utils.data.DataLoader(train_data,**train_kwargs)

    #     test_indices = get_indices(test_leaf_nodes, leaf)
    #     test_data = torch.utils.data.Subset(dataset2, test_indices)
    #     test_loader = torch.utils.data.DataLoader(test_data,**test_kwargs)

    #     sub_model = Net(n_hidden).to(device)
    #     optimizer = optim.Adadelta(sub_model.parameters(), lr=args.lr)
    #     scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    #     for epoch in range(1, args.epochs + 1):
    #         train(args, sub_model, device, train_loader, optimizer, epoch)
    #         scheduler.step()
    #     test(sub_model, device, test_loader, f"{index+1}")
    
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()





# if(dataset == 'mnist'):
#             X = datasets.MNIST('./data', train=True, download=True,
#                        transform=transform)
#             labels = X.targets
#             X = X.data.view(X.data.size(0), -1)
#             Y = datasets.MNIST('./data', train=False,
#                        transform=transform)
#             Y = Y.data.view(X.data.size(0), -1)
#     elif(dataset == 'fashion'):
#             X = datasets.FashionMNIST('./data', train=True, download=True,
#                        transform=transform)
#             labels = X.targets
#             X = X.data.view(X.data.size(0), -1)
#             Y = datasets.FashionMNIST('./data', train=False,
#                        transform=transform)
#             Y = Y.data.view(X.data.size(0), -1)
#     elif(dataset == 'cifar10'):
#             X = datasets.CIFAR10('./data', train=True, download=True,
#                        transform=transform)
#             labels = torch.tensor([target for _, target in X])
#             X = torch.stack([torch.Tensor(x) for x, _ in X], dim=0)
#             X = X.data.view(X.data.size(0), -1)
#             Y = datasets.CIFAR10('./data', train=False,
#                        transform=transform)
#             Y = torch.stack([torch.Tensor(x) for x, _ in Y], dim=0)
#             Y = X.data.view(Y.data.size(0), -1)
