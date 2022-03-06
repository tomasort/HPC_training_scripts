import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from PIL import Image 
import numpy as np
from torch import sin, sqrt, abs
import time

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

P4_N = 100_000
BATCH_SIZE = 1_000
NUM_EPOCHS = 2_000
LR = 0.05
MOMENTUM = 0.2

root_path = 'gdrive/My Drive/DSL_training_data/'

def eggholder_function(x_1, x_2):
    return -(x_2 + 47) * sin(sqrt(abs((x_1/2) + (x_2 + 47)))) - x_1 * sin(sqrt(abs(x_1 - (x_2 + 47))))

# Create the dataset using eggholder_function
X = (-512 - 512) * torch.rand((P4_N, 2)) + 512 
# add noise to the output of the eggholder function
y = eggholder_function(X[:, 0], X[:, 1]).unsqueeze(1) + torch.normal(0, 0.3, size=(P4_N, 1))
# Concatenate x_1, x_2, and y into a single tensor
dataset = torch.concat([X, y], axis=1)
# Divide the dataset into train and test (80/20)
dataset_train, dataset_test = train_test_split(dataset, test_size=0.2)
eggholder_train_loader = DataLoader(TensorDataset(dataset_train[:, :2], dataset_train[:, 2].unsqueeze(1)), batch_size=BATCH_SIZE, shuffle=True)
eggholder_test_loader = DataLoader(TensorDataset(dataset_test[:, :2], dataset_test[:, 2].unsqueeze(1)), batch_size=BATCH_SIZE, shuffle=True)

def RMSE(output, target):
    loss = torch.sqrt(torch.mean((output - target)**2))
    return loss

def train_loop(dataloader, model, loss_fn, optimizer, v=True):
    size = len(dataloader.dataset)
    model.train()
    train_loss, correct = 0, 0
    num_batches = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss += loss.item()
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            if v is True:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= num_batches
    correct /= size
    return train_loss, correct

def test_loop(dataloader, model, loss_fn, v_loss=True, v_acc=True):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    str_output = "Test Error: \n"
    if v_acc is True:
        str_output += f" Accuracy: {(100*correct):>0.1f}%\n"
    if v_loss is True:
        str_output += f" Avg loss: {test_loss:>8f} \n"
    print(str_output) 
    return test_loss, correct 

def train(net, train_iter, test_iter, device, loss, optimizer=None, num_epochs=10, init_weights=None):
    history_df = pd.DataFrame(columns=['epoch', 'test_accuracy', 'test_loss', 'train_accuracy', 'train_loss', 'training_time'])
    if init_weights is not None:
        net.apply(init_weights)
    print('training on', device)
    net.to(device)
    if optimizer is None:
        optimizer = torch.optim.SGD(net.parameters(), lr=LR, nesterov=True, momentum=MOMENTUM)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        # Sum of training loss, sum of training accuracy, no. of examples
        start = time.time()
        train_loss, train_acc = train_loop(train_iter, net, loss, optimizer)
        stop = time.time()
        training_time = stop-start
        test_loss, test_acc = test_loop(test_iter, net, loss, v_acc=False)
        history_df.loc[history_df.shape[0]] = [epoch+1, test_acc, test_loss, train_acc, train_loss, training_time]
    print("Done!")
    return history_df

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_networks(networks, num_layers, plot_results=True):
    networks_df = pd.DataFrame(columns=['layers', 'loss', 'units', 'num_parameters', 'training_time'])
    for i, net in enumerate(networks): 
        print(f"\nTraining Network with {net.hidden_units} neurons in hidden layer\n\n")
        h = train(net, eggholder_train_loader, eggholder_test_loader, device, loss=RMSE, num_epochs=NUM_EPOCHS)
        if plot_results:
            plt.plot(h['epoch'], h['test_loss'], label=f'Training Loss {net.hidden_units} Units')
        networks_df.loc[networks_df.shape[0]] = [num_layers, h['test_loss'].iloc[-1], net.hidden_units, count_parameters(net), h['training_time'].sum()]
        h[['test_loss', 'train_loss']].to_csv(root_path+f"/{num_layers}_layer/{net.hidden_units}_network.csv", index=False)
    if plot_results:
        plt.title(f'Comparison Between {num_layers} Hidden Layer Networks')
        plt.xlabel('Epoch #')
        plt.ylabel('RMSE Loss')
        plt.legend(loc = 'upper right')
        plt.show()
        plt.savefig(f'{root_path}/{num_layers}_hidden_comparison.png')
    networks_df.to_csv(root_path+f"/{num_layers}_layer_networks.csv", index=False)
    return networks_df

class NetOne(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.hidden_units  = hidden_units
        self.fc1 = nn.Linear(2, self.hidden_units)
        self.norm = nn.BatchNorm1d(self.hidden_units) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_units, 1)
    
    def forward(self, x):
        return self.fc2(self.relu(self.norm(self.fc1(x))))

print("Start Training 1 Layer Networks\n")
one_layer_networks = [NetOne(32), NetOne(64), NetOne(128), NetOne(256), NetOne(512)]
l1_networks = train_networks(one_layer_networks, num_layers=1)

class NetTwo(nn.Module):
    def __init__(self, hidden_size_1, hidden_size_2):
        super().__init__()
        self.hidden_units = hidden_size_1 + hidden_size_2
        self.fc1 = nn.Linear(2, hidden_size_1)
        self.norm1 = nn.BatchNorm1d(hidden_size_1) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.norm2 = nn.BatchNorm1d(hidden_size_2) 
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_2, 1)
    
    def forward(self, x):
        x = self.relu1(self.norm1(self.fc1(x)))
        x = self.relu2(self.norm2(self.fc2(x)))
        return self.fc3(x)

print("Start Training 2 Layer Networks\n")
two_layer_networks = [NetTwo(16, 16), NetTwo(32, 64), NetTwo(64, 128), NetTwo(128, 256), NetTwo(256, 256)]
l2_networks = train_networks(two_layer_networks, num_layers=2)

class NetThree(nn.Module):
    def __init__(self, hidden_size_1, hidden_size_2, hidden_size_3):
        super().__init__()
        self.hidden_units = hidden_size_1 + hidden_size_2 + hidden_size_3
        self.fc1 = nn.Linear(2, hidden_size_1)
        self.norm1 = nn.BatchNorm1d(hidden_size_1) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.norm2 = nn.BatchNorm1d(hidden_size_2) 
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.norm3 = nn.BatchNorm1d(hidden_size_3) 
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size_3, 1)
    
    def forward(self, x):
        x = self.relu1(self.norm1(self.fc1(x)))
        x = self.relu2(self.norm2(self.fc2(x)))
        x = self.relu3(self.norm3(self.fc3(x)))
        return self.fc4(x)

print("Start Training 3 Layer Networks\n")
three_layer_networks = [NetThree(16, 16, 16), NetThree(16, 32, 32), NetThree(32, 64, 32), NetThree(64, 128, 32), NetThree(128, 256, 64)]
l3_networks = train_networks(three_layer_networks, num_layers=3)

plt.title(f'Loss vs Number of Hidden Units')
plt.plot(l1_networks['units'], l1_networks['loss'], label='1 Layer Networks')
plt.plot(l2_networks['units'], l2_networks['loss'], label='2 Layer Networks')
plt.plot(l3_networks['units'], l3_networks['loss'], label='3 Layer Networks')
plt.xscale("log")
plt.xlabel('Hidden Units')
plt.ylabel('RMSE Loss')
plt.legend(loc = 'upper right')
plt.savefig(f'{root_path}/loss_vs_units.png')
plt.show()

plt.title(f'Loss vs Number of Parameters')
plt.plot(l1_networks['num_parameters'], l1_networks['loss'], label='1 Layer Networks')
plt.plot(l2_networks['num_parameters'], l2_networks['loss'], label='2 Layer Networks')
plt.plot(l3_networks['num_parameters'], l3_networks['loss'], label='3 Layer Networks')
plt.xscale("log")
plt.xlabel('Number of Parameters')
plt.ylabel('RMSE Loss')
plt.legend(loc = 'upper right')
plt.savefig(f'{root_path}/loss_vs_parameters.png')
plt.show()

plt.title(f'Training Time vs Number of Parameters')
plt.plot(l1_networks['num_parameters'], l1_networks['training_time'], label='1 Layer Networks')
plt.plot(l2_networks['num_parameters'], l2_networks['training_time'], label='2 Layer Networks')
plt.plot(l3_networks['num_parameters'], l3_networks['training_time'], label='3 Layer Networks')
plt.xscale("log")
plt.xlabel('Number of Parameters')
plt.ylabel('Training Time')
plt.legend(loc = 'upper right')
plt.savefig(f'{root_path}/training_time_vs_parameters.png')
plt.show()