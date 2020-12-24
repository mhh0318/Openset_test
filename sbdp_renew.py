#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/10/25 1:16
@author: merci
"""
from tqdm import tqdm
import torch
import math
from torch import nn
from torch.distributions import Beta, Bernoulli, Independent, kl_divergence, Normal
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
import os

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

def _clipped_sigmoid(x):
    finfo = torch.finfo(x.dtype)
    return torch.clamp(torch.sigmoid(x), min=finfo.tiny, max=1. - finfo.eps)

def mix_weights(x):
    offset = x.shape[-1] + 1 - x.new_ones(x.shape[-1]).cumsum(-1)
    z = _clipped_sigmoid(x - offset.log())
    z_cumprod = (1 - z).cumprod(-1)
    y = F.pad(z, (0, 1), value=1) * F.pad(z_cumprod, (1, 0), value=1)
    return y

def generate_pi(vi):
    remaining_stick_lengths = torch.cat((torch.ones(vi.size(0),1).to(device), torch.cumprod((1-vi),-1)[:,:-1]),1)
    pi = vi*remaining_stick_lengths
    return pi

def estimate_log_weights(a,b):
    digamma_sum = torch.digamma(b + a)
    digamma_a = torch.digamma(b)
    digamma_b = torch.digamma(a)
    value = (digamma_a - digamma_sum + torch.cat((torch.zeros(a.size(0),1).to(device), torch.cumsum((digamma_b - digamma_sum),1)[:,:-1]),1))
    # npv = (digamma_a - digamma_sum +np.hstack((0, np.cumsum(digamma_b - digamma_sum)[:-1])))
    return value



class ibop(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dims, prior_alpha, clusters=50):
        super(ibop, self).__init__()
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.prior_alpha = prior_alpha
        self.hidden_dims = hidden_dims
        self.clusters = clusters
        self.encoder_sb = nn.Sequential(*(
                [nn.Linear(self.input_dims, self.hidden_dims),
                 nn.ReLU()] +
                [nn.Linear(self.hidden_dims, self.hidden_dims),
                 nn.ReLU()] +
                [nn.Linear(self.hidden_dims, 2 * self.clusters),
                 nn.Softplus()]
        ))

        self.base = Independent(Normal(torch.zeros((1, self.latent_dims)).to(device), torch.ones((1, self.latent_dims)).to(device)), 1)
        # self.mu = nn.Parameter(self.base.rsample([clusters]).permute(1, 0, 2), requires_grad=True)
        self.mu = nn.Parameter(torch.randn(1,10,30), requires_grad=False)
        # self.log_var = nn.Parameter(base.rsample([clusters]).permute(1, 0, 2), requires_grad=True)
        self.log_var = nn.Parameter(torch.randn(1,10,30), requires_grad=True)

        self.encoder_Gaussian = nn.Sequential(
            nn.Linear(self.input_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_dims, 2 * self.latent_dims),
        )

        self.decoder = nn.Sequential(*(
                [nn.Linear(self.latent_dims, self.hidden_dims),
                 nn.ReLU()] +
                [nn.Linear(self.hidden_dims, self.hidden_dims),
                 nn.ReLU()] +
                [nn.Linear(self.hidden_dims, self.input_dims)]
        ))

    def encode_sb(self, X):
        encoder_out = self.encoder_sb(X)
        alpha, beta = encoder_out[:, :self.clusters], encoder_out[:, self.clusters:]
        return (Independent(Beta(alpha, beta), 1),
                alpha, beta)

    def encode_Gaussian(self, x):
        inf_para = self.encoder_Gaussian(x)
        mu, logvar = inf_para[:, :self.latent_dims], inf_para[:, self.latent_dims:]
        return Independent(Normal(mu, torch.exp(0.5 * logvar)), 1), mu, logvar

    def decode(self, pi):
        logits = self.decoder(pi)
        recon = torch.sigmoid(logits)
        return recon

    def Try1(self, x):
        batch_n = x.size(0)

        qz_x, mu_, logvar_ = self.encode_Gaussian(x)
        z = qz_x.rsample()

        q_pi_x, alpha, beta = self.encode_sb(x)
        vi = q_pi_x.rsample()
        pi = generate_pi(vi)
        log_pi = torch.log(pi)

        x_rec = self.decode(z)


        # compute the log_p of each dist
        log_p = []
        log_p_base = []
        mu_prior = self.mu.squeeze()
        log_var_prior = self.log_var.squeeze()
        base_model = self.base
        for i in range(self.clusters):
            mu_i = mu_prior[i,:]
            log_var_i = log_var_prior[i,:]
            sigma_i = torch.exp(0.5 * log_var_i)
            G_model = Independent(Normal(mu_i, sigma_i),1)
            log_p_i = G_model.log_prob(z)
            log_p.append(log_p_i)
            log_p_b = base_model.log_prob(mu_i)
            log_p_base.append(log_p_b)

        log_p = torch.stack(log_p,1)
        log_p_base = torch.tensor(log_p_base).to(device)

        # compute log_p of beta
        B_model = Beta(torch.ones_like(alpha),torch.ones_like(alpha)*self.prior_alpha)
        kl2 = kl_divergence(q_pi_x,Independent(B_model,1)).mean()
        log_p_vi = B_model.log_prob(vi)


        l1 = (log_p+log_pi+log_p_base.unsqueeze(0).repeat(batch_n,1)).sum(-1).mean()
        l2 = F.binary_cross_entropy(x_rec, x.view(-1, 784), reduction='sum')

        ELBO = l2-l1+kl2

        return ELBO

    def Try2(self, x, y):
        batch_n = x.size(0)

        qz_x, mu_, logvar_ = self.encode_Gaussian(x)
        z = qz_x.rsample()

        q_pi_x, alpha, beta = self.encode_sb(x)
        vi = q_pi_x.rsample()
        pi = mix_weights(vi)[:, :-1]

        log_p = []
        mu_prior = self.mu.squeeze()
        base_model = self.base
        for i in range(self.clusters):
            mu_i = mu_prior[i, :]
            sigma_i = torch.ones_like(mu_i)
            G_model = Independent(Normal(mu_i, sigma_i), 1)
            log_p_i = G_model.log_prob(z)
            log_p.append(log_p_i)
        log_p = torch.stack(log_p, 1)
        log_p_choose = log_p[range(batch_n),y]




    def sample(self, x, y):
        batch_n = x.size(0)

        qz_x, mu_, logvar_ = self.encode_Gaussian(x)
        z = qz_x.rsample()

        q_pi_x, alpha, beta = self.encode_sb(x)
        vi = q_pi_x.rsample()
        pi = generate_pi(vi)
        log_pi = torch.log(pi)

        x_rec = self.decode(z)

        log_p = []
        log_p_base = []
        mu_prior = self.mu.squeeze()
        base_model = self.base
        for i in range(self.clusters):
            mu_i = mu_prior[i,:]
            sigma_i = torch.ones_like(mu_i)
            G_model = Independent(Normal(mu_i, sigma_i),1)
            log_p_i = G_model.log_prob(z)
            log_p.append(log_p_i)
            log_p_b = base_model.log_prob(mu_i)
            log_p_base.append(log_p_b)

        log_p = torch.stack(log_p,1)
        log_p_base = torch.tensor(log_p_base).to(device)

        # B_model = Beta(torch.ones_like(alpha),torch.ones_like(alpha)*self.prior_alpha)
        # log_p_vi = B_model.log_prob(vi)

        log_p_total = (log_p+log_pi+log_p_base.unsqueeze(0).repeat(batch_n,1))
        predict = torch.argmax(log_p_total, -1).squeeze()
        predict1 = torch.argmax(log_p,-1)
        predict3 = torch.argmax(log_pi,-1)
        return x_rec, predict, predict1, predict3




def train(model, train_loader, epoch, optimizer):
    model.train()
    train_loss = 0
    for i, (img, label) in enumerate(train_loader):
        optimizer.zero_grad()
        batch = img.view(-1, D).to(device)
        loss = model.Try1(batch)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if (i + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                epoch, i * len(img), len(train_loader.dataset),
                       100. * i / len(train_loader),
                       loss.item() / len(img)))


def test(model, test_loader):
    model.eval()
    predict = []
    predict1 = []
    predict2 = []
    labels = []
    for i, (img, label) in enumerate(test_loader):
        batch = img.view(-1, D).to(device)
        labels.append(label.numpy())
        label = label.to(device)
        rec, pre, pre1, pre2 = model.sample(batch,label)
        predict.append(pre.detach().cpu().numpy())
        predict1.append(pre1.detach().cpu().numpy())
        predict2.append(pre2.detach().cpu().numpy())


    predict = np.concatenate(predict)
    predict1 = np.concatenate(predict1)
    predict2 = np.concatenate(predict2)
    labels = np.concatenate(labels)

    ARI1 = adjusted_rand_score(labels,predict)
    ARI2 = adjusted_rand_score(labels,predict1)
    ARI3 = adjusted_rand_score(labels,predict2)
    '''
    rec = rec.detach().cpu().numpy()
    fig, ax = plt.subplots(4, 5)
    for i, ax in enumerate(ax.flatten()):
        if i < 10:
            plottable_image = np.reshape(img[i], (28, 28))
            ax.imshow(plottable_image)
        elif i < 20:
            plottable_image = np.reshape(rec[i - 10], (28, 28))
            ax.imshow(plottable_image)
    plt.show()
    '''
    print('\nARI1:{:.3f}\nARI2:{:.3f}\nARI3:{:.3f}\n'.format(ARI1,ARI2,ARI3))

def test_network():
    model = ibop(28 * 28, 500, 20, 0.5)
    img = torch.randn((128, 28 * 28))
    model(img)

if __name__ == '__main__':
    device = torch.device('cuda:0')
    torch.manual_seed(18)
    D = 28*28
    flag = False
    if flag:
        test_network()
    else:
        torch.autograd.set_detect_anomaly(True)
        D = 28 * 28
        train_loader, test_loader = get_mnist()
        model = ibop(D, hidden_dims=400, latent_dims=30, prior_alpha=10, clusters=10)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for i in range(1000):
            train(model, train_loader, i, optimizer)
            test(model, test_loader)