#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 14:39:48 2020

@author: jkern
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
import numpy as np



input_size_D = 784       # The image size = 28 x 28 = 784
hidden_size_D = [1024,512,256]      # The number of nodes at the hidden layer
output_size_D = 1       #
input_size_G = 100       # The image size = 28 x 28 = 784
hidden_size_G = [256,512,1024]      # The number of nodes at the hidden layer
output_size_G = input_size_D  #
num_epochs = 50         # The number of times entire dataset is trained
batch_size = 64       # The size of input data took for one iteration
learning_rate = 0.0002  # The speed of convergence
leaky_slope = 0.2


def get_fashion_mnist_dataloaders(val_percentage=0.3, batch_size=1):
  dataset = MNIST("./dataset", train=True,  download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
  dataset_test = MNIST("./dataset", train=False,  download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
  len_train = int(len(dataset) * (1-val_percentage))
  len_val = len(dataset) - len_train
  dataset_train, dataset_val = random_split(dataset, [len_train, len_val])
  data_loader_train = DataLoader(dataset_train, batch_size=batch_size,shuffle=True,num_workers=4)
  data_loader_val = DataLoader(dataset_val, batch_size=batch_size,shuffle=True,num_workers=4)
  data_loader_test = DataLoader(dataset_test, batch_size=batch_size,shuffle=True,num_workers=4)
  return data_loader_train, data_loader_val, data_loader_test


class Logger:
    
    def __init__(self):
        self.losses_G = []
        self.losses_D = []


    def log(self, loss_G, loss_D):
        self.losses_G.append(loss_G)
        self.losses_D.append(loss_D)

    
class Discriminator(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc4 = nn.Linear(hidden_size[2], output_size)
        
        self.dropout = nn.Dropout(0.3)

        
    def forward(self, x):
        out = F.leaky_relu(self.fc1(x),leaky_slope)
        out = self.dropout(out)
        out = F.leaky_relu(self.fc2(out),leaky_slope)
        out = self.dropout(out)
        out = F.leaky_relu(self.fc3(out),leaky_slope)
        out = self.dropout(out)
        out = F.sigmoid(self.fc4(out))
        
        return out
    
    
class Generator(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc4 = nn.Linear(hidden_size[2], output_size)
        
    def forward(self, x):
        out = F.leaky_relu(self.fc1(x),leaky_slope)
        out = F.leaky_relu(self.fc2(out),leaky_slope)
        out = F.leaky_relu(self.fc3(out),leaky_slope)
        out = F.tanh(self.fc4(out))
        
        return out
        
    
D = Discriminator(input_size_D, hidden_size_D, output_size_D)
G = Generator(input_size_G, hidden_size_G, output_size_G)    

# print(D)
# print(G)
    
    
data_loader_train, data_loader_val, data_loader_test = get_fashion_mnist_dataloaders(batch_size=batch_size)

criterion = nn.BCELoss()

latent_noise_vector = Variable(torch.randn(64, input_size_G))

optim_D = torch.optim.Adam(D.parameters(), lr=learning_rate)
optim_G = torch.optim.Adam(G.parameters(), lr=learning_rate)




logger = Logger()


for epoch in range(num_epochs):
    
    
    for i, (images, labels) in enumerate(data_loader_train):
        
        
        # Update Discriminator
        
        optim_D.zero_grad()        
        # Compute Discriminator loss on real images
        images = Variable(images.view(-1, 28*28)) 
        
        output = D(images)
        
        loss_real = criterion(output, torch.ones(images.size()[0]))
        loss_real.backward()
        # Compute Discriminator loss on fake images        
        
        noise = Variable(torch.randn(images.size()[0], input_size_G))
        
        fake_images = G(noise)
        
        output = D(fake_images)

        loss_fake = criterion(output, torch.zeros(images.size()[0]))
        loss_fake.backward()
        D_loss = loss_real+loss_fake
        # D_loss.backward()
        optim_D.step()
        
        
        # Update Generator
        
        optim_G.zero_grad()
        
        noise = Variable(torch.randn(images.size()[0], input_size_G))
        
        fake_images = G(noise)
        
        output = D(fake_images)
        
        G_loss = criterion(output, torch.ones(images.size()[0]))
        G_loss.backward()
        optim_G.step()
        
        logger.log(G_loss.item(), D_loss.item())        

        
        if (i+1) % 100 == 0:                              # Logging
            print('Epoch [%d/%d], Step [%d/%d], Loss G: %.4f, Loss D: %.4f'
                 %(epoch+1, num_epochs, i+1,  len(data_loader_train.dataset.indices)//batch_size, G_loss.item(), D_loss.item()))
        
        

    
    if epoch%5==0:

        fig, axes = plt.subplots(nrows=5, ncols=5)

        for i in range(5):
            for j in range(5):
                test = G(latent_noise_vector)
                a = test[i*5+j].detach().numpy()
                axes[i,j].imshow(a.reshape(28,28),cmap='gray')
        
        
        
    print(epoch)


plt.figure()
plt.plot(logger.losses_D, label="Loss Discriminator")
plt.plot(logger.losses_G, label="Loss Generator")
plt.legend()


fig, axes = plt.subplots(nrows=5, ncols=5)

for i in range(5):
    for j in range(5):
        test = G(latent_noise_vector)
        a = test[i*5+j].detach().numpy()
        axes[i,j].imshow(a.reshape(28,28),cmap='gray')

# dataiter = iter(data_loader_train)
# x, y = dataiter.next()
# x = x.numpy()

# f = plt.figure()
# plt.imshow(np.squeeze(x[0]),cmap='gray')



# fig, axes = plt.subplots(nrows=5, ncols=5)

# for i in range(5):
#     for j in range(5):
#         test = G(latent_noise_vector)
#         a = test[i*5+j].detach().numpy()
#         axes[i,j].imshow(a.reshape(28,28),cmap='gray')
