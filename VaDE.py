#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/10/25 15:26
@author: merci
"""
import math

import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import os

import numpy as np

import matplotlib.pyplot as plt
from munkres import Munkres
from sklearn.manifold import TSNE
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from sklearn.mixture import GaussianMixture


def _reparameterize(mu, logvar):
    """Reparameterization trick.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    return z


class VaDE(torch.nn.Module):
    """Variational Deep Embedding(VaDE).

    Args:
        n_classes (int): Number of clusters.
        data_dim (int): Dimension of observed data.
        latent_dim (int): Dimension of latent space.
    """
    def __init__(self, n_classes, data_dim, latent_dim):
        super(VaDE, self).__init__()

        self._pi = Parameter(torch.zeros(n_classes))
        self.mu = Parameter(torch.randn(n_classes, latent_dim))
        self.logvar = Parameter(torch.randn(n_classes, latent_dim))

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(data_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2048),
            torch.nn.ReLU(),
        )
        self.encoder_mu = torch.nn.Linear(2048, latent_dim)
        self.encoder_logvar = torch.nn.Linear(2048, latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, data_dim),
            torch.nn.Sigmoid(),
        )

    @property
    def weights(self):
        return torch.softmax(self._pi, dim=0)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = _reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def classify(self, x, n_samples=8):
        with torch.no_grad():
            mu, logvar = self.encode(x)
            z = torch.stack(
                [_reparameterize(mu, logvar) for _ in range(n_samples)], dim=1)
            z = z.unsqueeze(2)
            h = z - self.mu
            h = torch.exp(-0.5 * torch.sum(h * h / self.logvar.exp(), dim=3))
            # Same as `torch.sqrt(torch.prod(self.logvar.exp(), dim=1))`
            h = h / torch.sum(0.5 * self.logvar, dim=1).exp()
            p_z_given_c = h / (2 * math.pi)
            p_z_c = p_z_given_c * self.weights
            y = p_z_c / torch.sum(p_z_c, dim=2, keepdim=True)
            y = torch.sum(y, dim=1)
            pred = torch.argmax(y, dim=1)
        return pred


def lossfun(model, x, recon_x, mu, logvar):
    batch_size = x.size(0)

    # Compute gamma ( q(c|x) )
    z = _reparameterize(mu, logvar).unsqueeze(1)
    h = z - model.mu
    h = torch.exp(-0.5 * torch.sum((h * h / model.logvar.exp()), dim=2))
    # Same as `torch.sqrt(torch.prod(model.logvar.exp(), dim=1))`
    h = h / torch.sum(0.5 * model.logvar, dim=1).exp()
    p_z_given_c = h / (2 * math.pi)
    p_z_c = p_z_given_c * model.weights
    gamma = p_z_c / torch.sum(p_z_c, dim=1, keepdim=True)

    h = logvar.exp().unsqueeze(1) + (mu.unsqueeze(1) - model.mu).pow(2)
    h = torch.sum(model.logvar + h / model.logvar.exp(), dim=2)
    loss = F.binary_cross_entropy(recon_x, x, reduction='sum') \
        + 0.5 * torch.sum(gamma * h) \
        - torch.sum(gamma * torch.log(model.weights + 1e-9)) \
        + torch.sum(gamma * torch.log(gamma + 1e-9)) \
        - 0.5 * torch.sum(1 + logvar)
    loss = loss / batch_size
    return loss


class AutoEncoderForPretrain(torch.nn.Module):
    """Auto-Encoder for pretraining VaDE.

    Args:
        data_dim (int): Dimension of observed data.
        latent_dim (int): Dimension of latent space.
    """
    def __init__(self, data_dim, latent_dim):
        super(AutoEncoderForPretrain, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(data_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2048),
            torch.nn.ReLU(),
        )
        self.encoder_mu = torch.nn.Linear(2048, latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, data_dim),
            torch.nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder_mu(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x


N_CLASSES = 10
PLOT_NUM_PER_CLASS = 128


def train(model, data_loader, optimizer, device, epoch, writer):
    model.train()

    total_loss = 0
    for x, _ in data_loader:
        x = x.to(device).view(-1, 784)
        recon_x, mu, logvar = model(x)
        loss = lossfun(model, x, recon_x, mu, logvar)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    writer.add_scalar('Loss/train', total_loss / len(data_loader), epoch)
    print('Train: Epoch {:>3}: Train Loss = {:.4f}'.format(
        epoch, total_loss / len(data_loader)))


def test(model, data_loader, device, epoch, writer, plot_points):
    model.eval()

    gain = torch.zeros((N_CLASSES, N_CLASSES), dtype=torch.int, device=device)
    with torch.no_grad():
        for xs, ts in data_loader:
            xs, ts = xs.to(device).view(-1, 784), ts.to(device)
            ys = model.classify(xs)
            for t, y in zip(ts, ys):
                gain[t, y] += 1
        cost = (torch.max(gain) - gain).cpu().numpy()
        assign = Munkres().compute(cost)
        acc = torch.sum(gain[tuple(zip(*assign))]).float() / torch.sum(gain)

        # Plot latent space
        xs, ts = plot_points[0].to(device), plot_points[1].numpy()
        zs = model.encode(xs)[0].cpu().numpy()
        tsne = TSNE(n_components=2, init='pca')
        zs_tsne = tsne.fit_transform(zs)

        fig, ax = plt.subplots()
        cmap = plt.get_cmap("tab10")
        for t in range(10):
            points = zs_tsne[ts == t]
            ax.scatter(points[:, 0], points[:, 1], color=cmap(t), label=str(t))
        ax.legend()


    writer.add_scalar('Acc/test', acc.item(), epoch)
    writer.add_figure('LatentSpace', fig, epoch)


def pretrain_train(model, data_loader, optimizer, device, epoch):
    model.train()

    total_loss = 0
    for x, _ in data_loader:
        batch_size = x.size(0)
        x = x.to(device).view(-1, 784)
        recon_x = model(x)
        loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch {:>3}: Train Loss = {:.4f}'.format(
        epoch, total_loss / len(data_loader)))


def pretrain():
    epochs=20
    gpu = 2
    learning_rate = 0.001
    batch_size = 128
    path = './pretrain_model_gpu2.ckpt'



    if_use_cuda = torch.cuda.is_available() and gpu >= 0
    device = torch.device('cuda:{}'.format(gpu) if if_use_cuda else 'cpu')

    dataset = datasets.MNIST('./', train=True, download=True,
                             transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=if_use_cuda)

    pretrain_model = AutoEncoderForPretrain(784, 10).to(device)

    optimizer = torch.optim.Adam(pretrain_model.parameters(),
                                 lr=learning_rate)

    for epoch in range(1, epochs + 1):
        pretrain_train(pretrain_model, data_loader, optimizer, device, epoch)

    with torch.no_grad():
        x = torch.cat([data[0] for data in dataset]).view(-1, 784).to(device)
        z = pretrain_model.encode(x).cpu()

    pretrain_model = pretrain_model.cpu()
    state_dict = pretrain_model.state_dict()

    gmm = GaussianMixture(n_components=10, covariance_type='diag')
    gmm.fit(z)

    model = VaDE(N_CLASSES, 784, 10)
    model.load_state_dict(state_dict, strict=False)
    model._pi.data = torch.log(torch.from_numpy(gmm.weights_)).float()
    model.mu.data = torch.from_numpy(gmm.means_).float()
    model.logvar.data = torch.log(torch.from_numpy(gmm.covariances_)).float()

    torch.save(model.state_dict(), path)


def main():
    epochs = 100
    gpu = 2
    learning_rate = 0.002
    batch_size = 128
    root = './pretrain_model_gpu2.ckpt'


    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')

    dataset = datasets.MNIST('./', train=True, download=True,
                             transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    # For plotting

    plot_points = {}
    for t in range(10):
        points = torch.cat([data for data, label in dataset if label == t])
        points = points.view(-1, 784)[:PLOT_NUM_PER_CLASS].to(device)
        plot_points[t] = points
    xs = []
    ts = []
    for t, x in plot_points.items():
        xs.append(x)
        t = torch.full((x.size(0),), t, dtype=torch.long)
        ts.append(t)
    plot_points = (torch.cat(xs, dim=0), torch.cat(ts, dim=0))


    model = VaDE(N_CLASSES, 784, 10)
    if not os.path.exists('./pretrain_model_gpu2.ckpt'):
        pretrain()
    else:
        model.load_state_dict(torch.load(root))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # LR decreases every 10 epochs with a decay rate of 0.9
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.9)

    # TensorBoard
    writer = SummaryWriter()

    for epoch in range(1, epochs + 1):
        train(model, data_loader, optimizer, device, epoch, writer)
        test(model, data_loader, device, epoch, writer, plot_points)
        lr_scheduler.step()

    writer.close()


if __name__ == '__main__':
    torch.manual_seed(18)
    np.random.seed(18)
    main()
