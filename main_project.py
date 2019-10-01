'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
#from utils import progress_bar
import matplotlib.pyplot as plt
from torchsummary import summary

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net1.train()
    train_loss = 0
    correct1 = 0
    
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = net1(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        
        train_loss += loss.item()
        _, predicted1 = output.max(1)
        
        total += targets.size(0)
        correct1 += predicted1.eq(targets).sum().item()

#        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#            % (train_loss/(batch_idx+1), 100.*correct1/total, correct1, total))
    epoch_acc1 = 100.*correct1/total
    
    print(f"Epoch:{epoch}, Train accuracy: {epoch_acc1}, Train loss = {train_loss}")

    
    scheduler.step()
    return epoch_acc1
    
def test(epoch):
    global best_acc
    net1.eval()
    test_loss = 0
    
    correct1 = 0
    
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            output = net1(inputs)
            
            _, predicted1 = output.max(1)
            
            total += targets.size(0)
            correct1 += predicted1.eq(targets).sum().item()

    epoch_acc1 = 100.*correct1/total

    print(f"Epoch:{epoch}, Test accuracy 1: {epoch_acc1}")

    
 #   str_save = 
    # Save checkpoint.
    if epoch_acc1 > best_acc:
        print('Saving..')
        state = {
            'net': net1.state_dict(),
            'acc': epoch_acc1,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = epoch_acc1
    
    return epoch_acc1


def network_constructor(width_multiplier=1, depth_multiplier=1, connection_type=1, relu=1):
    net1 = ResNet18(width_multiplier, depth_multiplier, connection_type, relu)
    net1 = net1.to(device)
    
    if device == 'cuda':
        net1 = torch.nn.DataParallel(net1)
        cudnn.benchmark = True
    
    return net1
        


acc_dict = {'Width':[], 'Depth':[], 'No_connect':[], 'Add':[], 'Add_mul':[], 'Mul':[], 'Relu':[], 'Non_relu':[]}

multiplier = [1,2,3]

for i in range(3):  
    print('==> Building model..')
    print(multiplier[i])
    net1 = network_constructor(depth_multiplier=multiplier[i])
    
    # for writing to a log file
    import sys
    old_stdout = sys.stdout
    string = "network_details_depth_" + str(multiplier[i]) + ".log"
    log_file = open(string,"w")
    sys.stdout = log_file
    summary(net1, input_size=(3, 32, 32))
    sys.stdout = old_stdout
    log_file.close()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net1.parameters(), lr=0.1) #, momentum=0.9, weight_decay=5e-4
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)

    print (count_parameters(net1))
    
    for experment_num in range(0,3): # do two experiments
        experiment_best_acc = 0
        for epoch in range(10):    
            acc = train(epoch)
            test(epoch)
            if acc>experiment_best_acc: 
                experiment_best_acc = acc
        
        acc_dict['Depth'].append(acc)
        print(f"Depth: {acc_dict['Depth']}")
        
for i in range(3):  
    print('==> Building model..')
    net1 = network_constructor(width_multiplier=multiplier[i])
    
    # for writing to a log file
    import sys
    old_stdout = sys.stdout
    string = "network_details_width_" + str(multiplier[i]) + ".log"
    log_file = open(string,"w")
    sys.stdout = log_file
    summary(net1, input_size=(3, 32, 32))
    sys.stdout = old_stdout
    log_file.close()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net1.parameters(), lr=0.1) #, momentum=0.9, weight_decay=5e-4
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)

    print (count_parameters(net1))
    
    for experment_num in range(0,3): # do two experiments
        experiment_best_acc = 0
        for epoch in range(10):    
            acc = train(epoch)
            test(epoch)
            if acc>experiment_best_acc: 
                experiment_best_acc = acc
        
        acc_dict['Width'].append(acc)
        print(f"Width: {acc_dict['Width']}")


            
connection = [0,1,2,3]
for i in range(4):  
    print('==> Building model..')
    net1 = network_constructor(connection_type=connection[i])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net1.parameters(), lr=0.1) #, momentum=0.9, weight_decay=5e-4
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)

    print (count_parameters(net1))
    
    for experment_num in range(1,6): # do two experiments
        experiment_best_acc = 0
        for epoch in range(10):    
            acc = train(epoch)
            test(epoch)
            if acc>experiment_best_acc: 
                experiment_best_acc = acc
        if i==0:
            acc_dict['No_connect'].append(acc)
            print(f"No_connect: {acc_dict['No_connect']}")
        elif i==1:
            acc_dict['Add'].append(acc)
            print(f"Add: {acc_dict['Add']}")
        elif i==2:
            acc_dict['Mul'].append(acc)
            print(f"Mul: {acc_dict['Mul']}")        
        else:
            acc_dict['Add_mul'].append(acc)
            print(f"Add_mul: {acc_dict['Add_mul']}")
            
            
relu_type = [0,1]
for i in range(2):  
    print('==> Building model..')
    net1 = network_constructor(relu=relu_type[i])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net1.parameters(), lr=0.1) #, momentum=0.9, weight_decay=5e-4
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)

    print (count_parameters(net1))
    
    for experment_num in range(5): # do two experiments
        experiment_best_acc = 0
        for epoch in range(10):    
            acc = train(epoch)
            test(epoch)
            if acc>experiment_best_acc: 
                experiment_best_acc = acc
        if i==0:
            acc_dict['Non_relu'].append(acc)
            print(f"non_relu: {acc_dict['Non_relu']}")
        else:
            acc_dict['Relu'].append(acc)
            print(f"Relu: {acc_dict['Relu']}")
 
    
print(f"Width: {acc_dict['Width']}")     
print(f"Depth: {acc_dict['Depth']}")            
print(f"No_connect: {acc_dict['No_connect']}")     
print(f"Add: {acc_dict['Add']}")
print(f"Mul: {acc_dict['Mul']}")      
print(f"Add_mul: {acc_dict['Add_mul']}")
print(f"non_relu: {acc_dict['Non_relu']}")           
print(f"relu: {acc_dict['Relu']}")


#f=plt.figure(1)
#plt.plot(acc_dict['Non_relu'], 'r--') 
#plt.plot(acc_dict['Relu'], 'g--')
#f.show()
#
#g=plt.figure(2)
#plt.plot(acc_dict['Add'], 'gs')
#plt.plot(acc_dict['Add_mul'], 'b--')
#plt.plot(acc_dict['Mul'], 'r--')
#plt.plot(acc_dict['No_connect'], 'k--')
#g.show()
#
#h=plt.figure(2)
plt.plot(acc_dict['Depth'], 'r--')   
plt.plot(acc_dict['Width'], 'b--')  
#h.show()



# Future to-do
    # start with a lightweight resnet with arguments of width multiplier, depth multiplier, connection type, relu or not relu
    # after 50 epochs, loop for 5 epochs with each of the varying arguments, and see improvement in performance and increase in GFLOPS,
    # use the best model at 55 epochs, and continue the process until 75 epochs
    # dictionary for each argument vals' performance improvement and increase in GFLOPS as keys, e.g. Relu_perf, Relu_GFLOPS, non_Relu_perf, non_Relu_GFLOPS, etc.
    
#list1 = [x for x in acc_dict['Depth']]
#list2 = [x for x in acc_dict['Width']]
#list3 = list1[0:3]
#list4 = list2[0:3]
#plt.plot(list3, 'r--')   
#plt.plot(list4, 'b--')  
#
#list3 = list1[3:6]
#list4 = list2[3:6]
#plt.plot(list3, 'r--')   
#plt.plot(list4, 'b--')  
#
#list3 = list1[6:9]
#list4 = list2[6:9]
#plt.plot(list3, 'r--')   
#plt.plot(list4, 'b--')  