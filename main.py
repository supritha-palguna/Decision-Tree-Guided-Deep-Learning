from __future__ import print_function
import warnings
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import os
import argparse
import sys
import matplotlib.pyplot as plt
from torchsummary import summary
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pandas as pd
from collections import Counter

warnings.simplefilter(action='ignore', category=FutureWarning)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

'''path setting'''
parser = argparse.ArgumentParser(description='Train and Test UPANets with PyTorch')
parser.add_argument('--pkg_path', default='./', type=str, help='package path')
parser.add_argument('--save_path', default='./models/', type=str, help='package path')

'''experiment setting'''
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', default=0.0005, type=float, help='weight decay')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument("-d", "--dataset", help="Dataset used for experiments. ",type=str, default=["cifar_10"], nargs='+')
parser.add_argument("-t", "--threshold", help="Confusion matrix threshold for submodels ",type=float, default=0.8)
parser.add_argument("-n", "--num_classes", help="Number of classes",type=int, default=10)
parser.add_argument("-l", "--user_lambda", help="regulates the trade-off between accuracy and model size",type=float, default=[0.5], nargs='+')
#parser.add_argument('--datasets', default='cifar_10', type=str, help='using dataset')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--epochs', default=120, type=int, help='total traing epochs')

'''model setting'''
parser.add_argument('--blocks', default=1, type=int, help='block number in UPANets')
parser.add_argument('--filters', default=16, type=int, help='filter number in UPANets')

'''mode setting'''
parser.add_argument('--mode', default='closure', type=str, choices=['closure', 'separate'], help='block number in UPANets')

parser.add_argument("-de", "--depth", help="Depth of decision tree", type=int, default=1)

args = parser.parse_args()

pkgpath = args.pkg_path
save_path = args.save_path

if os.path.isdir(save_path) == False:
    os.makedirs(save_path)

sys.path.append(pkgpath)

from models.upanets import UPANets

from pugd import PUGD

from utils import progress_bar

def map_values(input_value):
    value_map = {1: 12, 2: 8}
    mapped_value = value_map.get(input_value, None)
    return mapped_value

dataset_name = ''
def get_dataset_model(dataset):
    global dataset_name 
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    img_size=32
    if(dataset == 'cifar_10'):
        trainset = datasets.CIFAR10(
            root='./data/cifar_10', train=True, download=True, transform=transform_train)
        
        testset = datasets.CIFAR10(
            root='./data/cifar_10', train=False, download=True, transform=transform_test)
        classes = 10
        net =  UPANets(map_values(args.depth), classes, args.blocks, img_size)
        dataset_name = 'cifar_10'
    elif(dataset == 'cifar_100'):
        trainset = datasets.CIFAR100(
            root='./data/cifar_100', train=True, download=True, transform=transform_train)
        
        testset = datasets.CIFAR100(
            root='./data/cifar_100', train=False, download=True, transform=transform_test)
        classes = 100
        net =  UPANets(map_values(args.depth), classes, args.blocks, img_size)
        dataset_name = 'cifar_100'
    return trainset, testset, net

def get_indices(arr, value):
        indices = []
        for i in range(len(arr)):
            if arr[i] == value:
                indices.append(i)
        return indices



criterion = nn.CrossEntropyLoss()
configs = []
threshold_data = []
class NN_DT_classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, args, my_net):
        self.dt_model = DecisionTreeClassifier(max_depth=args.depth)
        self.models = None
        self.net = my_net
        self.args = args

    def fit(self, X, y):
        global dataset_name 
        original_data = X
        if type(X.data) is np.ndarray:
             X = torch.stack([torch.Tensor(x) for x, _ in X], dim=0)
        data = X.data
        reshaped_data = data.view(data.size(0), -1)
        print(reshaped_data.size())  
       
        self.dt_model.fit(reshaped_data, y)
        leaf_nodes = self.dt_model.apply(reshaped_data)

        n_models = len(np.unique(leaf_nodes))
        
        self.models = [self.net.to(device) for _ in range(n_models)]
        for index, leaf in enumerate(np.unique(leaf_nodes)):
            indices = get_indices(leaf_nodes.tolist(), leaf)

            PATH = save_path+str(index+1) + dataset_name+'.pt'
            trainData = torch.utils.data.Subset(original_data, indices)
            target_classes = [original_data.targets[idx] for idx in indices]

            train_list = []
            epoch_list = []
            train_acc_list = []

            confmat_list = []
            f1_score_list = []
            train_acc = train_loss = confmat = f1_score = None
            if os.path.exists( PATH ):
                print(PATH)
                model = torch.load(PATH)
                model.eval()
                self.models[index] = model
            else :
                model = self.models[index]
                if device == 'cuda':
                    model = torch.nn.DataParallel(model)
                    cudnn.benchmark = True

                base_optimizer = optim.SGD
                optimizer = PUGD(model.parameters(),
                    base_optimizer,
                    lr=self.args.lr,
                    momentum=self.args.momentum,
                    weight_decay=self.args.wd,
                    )
                
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
                
                for epoch in range(start_epoch, start_epoch+args.epochs):
                    epoch_list.append(epoch)
                    train_loss, train_acc, confmat, f1_score = train(epoch, trainData, model, optimizer)
                    train_list.append(train_loss)
                    train_acc_list.append(train_acc)
                    confmat_list.append(confmat)
                    f1_score_list.append(f1_score)
        
                    epoch_line = 'epoch: {0}/ total epoch: {1} '.format(epoch, args.epochs) 
                    best_acc_line = '| acc: {0} '.format(train_acc)
                    loss_line = '| loss: {0} '.format(train_loss)
                    
                    with open(save_path+'/sub_logs.txt', 'a') as f:
                        f.write(epoch_line + best_acc_line +loss_line+'\n')
                    scheduler.step()
            
            total_params = 0
            for _, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                params = parameter.numel()
                total_params += params
            print(f"Total Trainable Params: {total_params:,}")
            
            configs.append( 
                 {'dataset' : dataset_name,
                    'model': f'submodel {index + 1}',
                    'classes' : len(np.unique(target_classes)),
                    'parameters' : total_params,
                    'data_len' : len(trainData),
                    'class_dist' : Counter(target_classes),
                    'train_acc' : train_acc,
                    'train_loss' : train_loss,
                    'confmat' : confmat_list,
                    'f1_score' : f1_score_list,
                    'train_loss_list' : train_list,
                    'train_acc_list': train_acc_list
                })
            
            torch.save(model, save_path+'/'+str(index+1) + dataset_name+'.pt' )
        return self


    def predict(self, X):
        global dataset_name 
        original_data = X
        if type(X.data) is np.ndarray:
            X = torch.stack([torch.Tensor(x) for x, _ in X], dim=0)
        test_leaf_nodes = self.dt_model.apply(X.data.view(X.data.size(0), -1))
        
        thresholds = np.linspace(0.8, 1.0, 10)
        best_loss = float('inf')
        best_threshold = 0
        user_lambda = np.linspace(0.1, 0.001, 10)
        for lam in args.user_lambda:
            best_loss = float('inf')
            best_threshold = 0
            for threshold in thresholds:
                
                percentage_bigmodel = 0
                total_test_acc = 0
                total_test_loss = 0
                for index, leaf in enumerate(np.unique(test_leaf_nodes)):
                    indices = get_indices(test_leaf_nodes, leaf)
                    test_loss, test_acc, confmat, f1score, percentage_main  = test(self.models[index], torch.utils.data.Subset(original_data, indices), threshold)
                    
                    with open(save_path+'/sub_logs.txt', 'a') as f:
                            f.write('Test model '+f"{index+1} "+dataset_name+' : test_loss '+ str(test_loss) +' | test_acc '+ str(test_acc) 
                                +'\n')
                    new_data = {
                        'test_loss': test_loss,
                        'test_confmat': confmat,
                        'test_f1score': f1score, 
                        'test_acc': test_acc,
                        'threshold': threshold}
                    configs[index+1].update(new_data)
        
                    percentage_bigmodel += percentage_main
                    total_test_loss += test_loss
                    total_test_acc += test_acc

                custom_loss = total_test_loss/2 + (lam * threshold**2)
                if(custom_loss < best_loss):
                    best_loss = custom_loss
                    best_threshold = threshold
                    
                threshold_data.append(
                {
                    'Threshold' : threshold,
                    'lambda': lam,
                    'Accuracy': total_test_acc/2,
                    'Big model percentage' : percentage_bigmodel/2,
                    'loss': total_test_loss/2,
                    'custom_loss': custom_loss,
                    'Best_loss' : best_loss,
                    'Best_threshold' : best_threshold
                })



def train(epoch, train_data, model, optimizer):
    trainloader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, num_workers=6)
    print('Epoch:{0}/{1}'.format(epoch, args.epochs))
    model.train()
    
    train_loss = 0 
    correct = 0
    total = 0
    
    all_predicted = []
    all_targets = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        if args.mode == 'closure':
            
            def closure():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

            outputs = model(inputs)        
            loss = criterion(outputs, targets)       
            loss.backward()
            optimizer.step(closure)
            
        else:
            
            outputs = model(inputs)        
            loss = criterion(outputs, targets)       
            loss.backward()
            optimizer.first_step()
            
            outputs = model(inputs)        
            loss = criterion(outputs, targets)       
            loss.backward()
            optimizer.second_step(zero_grad=True)
            
            
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        all_predicted.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
      
        progress_bar(batch_idx, len(trainloader), 'train_Loss: %.3f | train_Acc: %.3f%% (%d/%d)'
                      % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    confusionmatrix = confusion_matrix(all_targets, all_predicted)
    f1 = f1_score(all_targets, all_predicted, average='weighted')

    return train_loss/(batch_idx+1), 100.*correct/total, confusionmatrix, f1


def test(model, testset, threshold):
    testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=6)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    acc_list = []
    print(len(testloader))
    with torch.no_grad():
        all_predicted = []
        all_targets = []
        count_main_model = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            
            
            main_model = torch.load(save_path+'whole_model_'+ dataset_name +'.pt')
            main_model.eval()
            probabilities = F.softmax(outputs, dim=1)
            
            if(threshold > 0):
                for i in range(inputs.size(0)):
                    if torch.max(probabilities[i]) < threshold:
                        model_input = inputs[i].unsqueeze(0)  
                        model_output = main_model(model_input) 
                        outputs[i] = model_output.clone() 
                        count_main_model += 1
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc_list.append(100.*correct/total)
            
            progress_bar(batch_idx, len(testloader), 'test_Loss: %.3f | test_Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            print()
            print('>>>mean: {0}, std: {1}'.format(round(np.mean(acc_list), 2), round(np.std(acc_list), 2)))
            all_predicted.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        total_samples = len(testloader.dataset)
        percentage_main = (count_main_model / total_samples) * 100
        confusionmatrix = confusion_matrix(all_targets, all_predicted)
        f1 = f1_score(all_targets, all_predicted, average='weighted')
    return test_loss/(batch_idx+1), 100.*correct/total, confusionmatrix, f1, percentage_main

def predict_whole_model(dataset, testset):
    global dataset_name 
    classes = args.num_classes
    PATH = save_path+'whole_model_'+ dataset_name +'.pt'
    configs.append(
                {
                    'dataset' : dataset_name,
                    'model': 'big model',
                    'classes' : classes,
                    'data_len' : len(dataset), 
                }
            )
    if os.path.exists( PATH ):
        whole_model = torch.load(PATH)
        whole_model.eval()
    else :
        whole_model =  UPANets(16, classes, args.blocks, 32)
        whole_model = whole_model.to(device)
        if device == 'cuda':
            whole_model = torch.nn.DataParallel(whole_model)
            cudnn.benchmark = True

        base_optimizer = optim.SGD
        optimizer = PUGD(whole_model.parameters(),
                    base_optimizer,
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.wd,
                    )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
             

        test_loss = 0
        train_list = []
        epoch_list = []
        train_acc_list = []
        confmat_list = []
        f1_score_list = []

        for epoch in range(start_epoch, start_epoch+args.epochs):

            epoch_list.append(epoch)
        
            train_loss, train_acc, confmat, f1_score = train(epoch, dataset, whole_model, optimizer)
            train_list.append(train_loss)
            train_acc_list.append(train_acc)
            confmat_list.append(confmat)
            f1_score_list.append(f1_score)
            
            epoch_line = 'epoch: {0}/ total epoch: {1} '.format(epoch, args.epochs) 
            best_acc_line = 'acc: {0} '.format(train_acc)
            loss_line = 'loss: {0} '.format(train_loss)

            with open(save_path+'/logs.txt', 'a') as f:
                f.write(epoch_line + best_acc_line + loss_line+ '\n')
            scheduler.step()

        torch.save( whole_model, PATH)
        
        new_data = {
                    'train_acc' : train_acc,
                    'train_loss' : train_loss,
                    'confmat' : confmat_list,
                    'f1_score' : f1_score_list,
                    'train_loss_list' : train_list,
                    'train_acc_list': train_acc_list
                }
        configs[0].update(new_data)
            
        
    total_params = 0
    for _, parameter in whole_model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params  

    test_loss, test_acc, test_confmat, f1score, _ = test(whole_model, testset, threshold=0)
    print(test_acc)
    new_data = {
                'parameters' : total_params,
                'test_acc': test_acc,
                'test_confmat': test_confmat,
                'test_f1score': f1score, 
                'test_loss': test_loss}

    configs[0].update(new_data)

    with open(save_path+'/logs.txt', 'a') as f:
                f.write('Test model '+dataset_name+' : test_loss'+ str(test_loss) +' | test_acc'+ str(test_acc) 
                            +'\n')
    

for dataset in args.dataset:
    
    X, Y, my_net = get_dataset_model(dataset)
    predict_whole_model(X, Y)
    dt = NN_DT_classifier(args, my_net)
    dt.fit(X, X.targets)
    dt.predict(Y)
    
    df = pd.DataFrame(configs)
    df.to_csv("results.csv",index=False)
    threshold_df = pd.DataFrame(threshold_data)
    threshold_df.to_csv("threshold_df.csv",index=False)
    with pd.option_context('display.max_rows', None): 
            print(df)

    

