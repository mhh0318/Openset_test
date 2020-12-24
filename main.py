#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/10/12 22:50
@author: merci
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from vanilla_dpvae import vanillaVAE

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def get_mnist(batch_size=128):
    # Training dataset
    train_loader = torch.utils.data.DataLoader(
        MNIST(root='.', train=True, download=True,
            transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, pin_memory=True)
    # Test dataset
    test_loader = torch.utils.data.DataLoader(
        MNIST(root='.', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, pin_memory=True)
    return train_loader, test_loader

device = torch.device('cuda:0')

def train(model, train_loader, epoch, learning_rate=1e-3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criteria = nn.CrossEntropyLoss()
    train_loss = 0
    for i, (img,label) in enumerate(train_loader):
        optimizer.zero_grad()
        batch = img.view(-1, D).to(device)
        label = label.to(device)
        recon, pre, z, mu, logvar = model(batch)
        loss = loss_function(recon, batch, mu, logvar)
        cel = criteria(pre, label)
        total_loss = cel*30 + loss
        total_loss.backward()
        train_loss += total_loss.item()
        predictions = torch.argmax(F.softmax(pre,dim=1),dim=1)
        accuracy  = torch.sum(predictions==label).float()/img.size(0)
        optimizer.step()
        if (i + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClassification Acc:{:.6f}'.format(
                epoch, i * len(img), len(train_loader.dataset),
                100. * i / len(train_loader),
                loss.item() / len(img),accuracy))

def test(model, test_loader):
    model.eval()
    test_loss = 0
    figs = {'img': [], 'rec_dp': []}

    for i, (batch, labels) in enumerate(test_loader):
        model.eval()
        img = batch.cpu().numpy()
        img = np.concatenate(img)
        figs['img'].append(img)
        batch = batch.view(-1, D).to(device)
        recon, pre, z, mu, logvar = model(batch)
        loss = loss_function(recon, batch, mu, logvar)
        rec_fig = model.sample(batch)
        figs['rec_dp'].append(rec_fig.detach().cpu().numpy())
        test_loss += loss.item()

    figs['img'] = np.concatenate(figs['img'])
    figs['rec_dp'] = np.concatenate(figs['rec_dp'])
    fig, ax = plt.subplots(4, 5)
    for i, ax in enumerate(ax.flatten()):
        if i < 10:
            plottable_image = np.reshape(figs['img'][i], (28, 28))
            ax.imshow(plottable_image, cmap='gray_r')
        elif i < 20:
            plottable_image = np.reshape(figs['rec_dp'][i - 10], (28, 28))
            ax.imshow(plottable_image, cmap='gray_r')
    plt.show()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))



torch.manual_seed(18)
D = 28 * 28
model = vanillaVAE(D, 400, 20)
model.to(device)
train_loader, test_loader = get_mnist()

for ep in range(100):
    train(model, train_loader, ep)
    test(model, test_loader)