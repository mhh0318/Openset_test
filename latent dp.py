#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/10/13 17:41
@author: merci
"""

import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
from vanilla_dpvae import vanillaVAE
from sklearn.mixture import BayesianGaussianMixture



D = 28*28
device = torch.device('cuda:0')

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

def kld(IGMM, mu, logvar, z):
    type = IGMM.predict(z.cpu().detach().numpy())
    mu_ = torch.tensor(IGMM.means_[type]).cuda()
    logvar_ = torch.tensor(np.log(IGMM.covariances_[type])).cuda()
    kl_ = 0.5 * torch.sum( (mu_-mu).pow(2) / logvar_.exp() + logvar_ - logvar + logvar.exp()/logvar_.exp() )
    # kl_ = torch.sum(mu_.sub_(mu).pow(2).div(logvar_.exp()).add_(logvar_).sub_(logvar).add_(logvar.exp().div(logvar_.exp()))).mul_(0.5)
    return kl_


def ELBO(recon, x, mu, logvar, IGMM, z):
    BCE = F.binary_cross_entropy(recon, x.view(-1, 784))
    KLD = kld(IGMM, mu, logvar, z)
    return BCE + KLD

def latent_code(model, train_loader):
    model.eval()
    latents = []
    for i, (img,label) in enumerate(train_loader):
        batch = img.view(-1, D).to(device)
        _, _, latent, _, _ = model(batch)
        latents.append(latent.cpu().detach().numpy())
    latents = np.concatenate(latents)
    return latents

def train(model, IGMM, train_loader, epoch, optimizer):
    model.train()
    criteria = nn.CrossEntropyLoss()
    train_loss = 0
    latent = latent_code(model, train_loader)
    print('Train IGMM')
    IGMM.fit(latent)
    print('IGMM: ' + str(IGMM.weights_))
    for i, (img,label) in enumerate(train_loader):
        batch = img.view(-1, D).to(device)
        label = label.to(device)
        optimizer.zero_grad()
        recon, pre, z, mu, logvar = model(batch)
        elbo_loss = ELBO(recon, batch, mu, logvar, IGMM, z)
        c_loss = criteria(pre, label)
        loss = elbo_loss + c_loss
        # elbo_loss.backward()
        loss.backward()
        train_loss += loss
        optimizer.step()
        predictions = torch.argmax(F.softmax(pre,dim=1),dim=1)
        accuracy = torch.sum(predictions == label).float() / img.size(0)
        if (i + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClassification Acc:{:.6f}'.format(
                epoch, i * len(img), len(train_loader.dataset),
                       100. * i / len(train_loader),
                       loss.item() / len(img), accuracy))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return IGMM

def test(model, IGMM, test_loader):
    model.eval()
    test_loss = 0
    figs = {'img': [], 'rec_dp': []}
    for img, label in test_loader:
        batch = img.view(-1, D).to(device)
        figs['img'].append(img)
        recon, pre, z, mu, logvar = model(batch)
        test_loss += ELBO(recon, batch, mu, logvar, IGMM, z)
        rec_fig = model.dpsample(batch, IGMM)
        figs['rec_dp'].append(rec_fig.detach().cpu().numpy())

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


torch.manual_seed(18)
D = 28 * 28
model = vanillaVAE(D, 400, 50)
model.to(device)
train_loader, test_loader = get_mnist()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
prior = BayesianGaussianMixture(n_components=100, covariance_type='diag')
np.set_printoptions(suppress=True)
for ep in range(100):
    prior = train(model, prior, train_loader, ep, optimizer)
    test(model, prior, test_loader)
