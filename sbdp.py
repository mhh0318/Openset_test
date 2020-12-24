#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/10/16 3:29
@author: merci
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/7/21 4:44
@author: merci
"""
import torch
from torch import nn
from torch.distributions import Beta, Bernoulli, Independent, kl_divergence, Normal
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

################################Utils################################################
def log_mean_exp(x, dim):
    """
    Compute the log(mean(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which mean is computed

    Return:
        _: tensor: (...): log(mean(exp(x), dim))
    """
    return log_sum_exp(x, dim) - np.log(x.size(dim))


def log_sum_exp(x, dim=0):
    """
    Compute the log(sum(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which sum is computed

    Return:
        _: tensor: (...): log(sum(exp(x), dim))
    """
    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)
    return max_x + (new_x.exp().sum(dim)).log()


def log_normal(x, m, v):
    """
    Computes the elem-wise log probability of a Gaussian and then sum over the
    last dim. Basically we're assuming all dims are batch dims except for the
    last dim.

    Args:
        x: tensor: (batch, ..., dim): Observation
        m: tensor: (batch, ..., dim): Mean
        v: tensor: (batch, ..., dim): Variance

    Return:
        kl: tensor: (batch1, batch2, ...): log probability of each sample. Note
            that the summation dimension (dim=-1) is not kept
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Compute element-wise log probability of normal and remember to sum over
    # the last dimension
    ################################################################################
    # print("q_m", m.size())
    # print("q_v", v.size())
    const = -0.5 * x.size(-1) * torch.log(2 * torch.tensor(np.pi))
    # print(const.size())
    log_det = -0.5 * torch.sum(torch.log(v), dim=-1)
    # print("log_det", log_det.size())
    log_exp = -0.5 * torch.sum((x - m) ** 2 / v, dim=-1)

    log_prob = const + log_det + log_exp

    ################################################################################
    # End of code modification
    ################################################################################
    return log_prob


def log_normal_mixture(z, m, v):
    """
    Computes log probability of a uniformly-weighted Gaussian mixture.

    Args:
        z: tensor: (batch, dim): Observations
        m: tensor: (batch, mix, dim): Mixture means
        v: tensor: (batch, mix, dim): Mixture variances

    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Compute the uniformly-weighted mixture of Gaussians density for each sample
    # in the batch
    ################################################################################
    z = z.unsqueeze(1)
    log_probs = log_normal(z, m, v)
    # print("log_probs_mix", log_probs.shape)

    log_prob = log_mean_exp(log_probs, 1)
    # print("log_prob_mix", log_prob.size())

    ################################################################################
    # End of code modification
    ################################################################################

    return log_prob
    ################################################################################
    ################################################################################
    ################################################################################
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
    y = F.pad(z, (0, 1), value=1) * \
        F.pad(z_cumprod, (1, 0), value=1)
    return y

####################################################################################
####################################################################################

class sb_vae(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dims, prior_alpha, clusters=50):
        super(sb_vae, self).__init__()
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

        base = Independent(Normal(torch.zeros((1, self.latent_dims)), torch.ones((1, self.latent_dims))), 1)
        self.should_fixed = base.rsample([clusters]).permute(1, 0, 2).cuda()
        self.theta = nn.Parameter(base.rsample([clusters]).permute(1, 0, 2), requires_grad=False)

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
        # return Independent(Bernoulli(logits=logits), 1)

    def forward(self, X):
        batch_size = X.size(0)
        theta = self.should_fixed.repeat(batch_size, 1, 1)

        qz_x, mu_, logvar_ = self.encode_Gaussian(X)
        z = qz_x.rsample()
        px_z = self.decode(z)

        q_pi_x, alpha, _ = self.encode_sb(X)
        ###base = Independent(Normal(torch.zeros_like(mu_), torch.ones_like(logvar_)), 1)
        pi = mix_weights(q_pi_x.rsample())[:, :-1]
        ###theta = base.rsample([self.clusters]).permute(1,0,2)
        #pi_for_mul = pi.unsqueeze(-1).repeat(1,1,self.latent_dims)
        #pi_for_mul = pi[range(pi.size(0)), torch.argmax(pi,1)]
        theta_for_mul = theta[range(pi.size(0)), torch.argmax(pi,1),:]
        G_mean = theta_for_mul
        p_mu = Independent(Normal(G_mean, torch.ones_like(mu_)), 1)
        mu = p_mu.rsample()
        p_pi = Independent(
            Beta(torch.ones_like(alpha), torch.ones_like(alpha) * self.prior_alpha), 1)
        p_z = Independent(Normal(mu, torch.ones_like(logvar_)), 1)

        # nll = -px_z.log_prob(X).mean()
        BCE = F.binary_cross_entropy(px_z, X.view(-1, 784), reduction='sum')

        #log_p_theta = log_normal_mixture(z, G_mean, torch.ones_like(G_mean))
        #log_q_phi = log_normal(z, mu_, torch.exp(logvar_))
        #kl = log_q_phi - log_p_theta

        kl = kl_divergence(qz_x, p_z).mean()
        kl2 = kl_divergence(q_pi_x, p_pi).mean()
        return BCE+kl+kl2

    def sample(self, X, label):
        batch_size = X.size(0)
        theta = self.should_fixed.repeat(batch_size, 1, 1)

        qz_x, mu_, logvar_ = self.encode_Gaussian(X)
        z = qz_x.rsample()
        px_z = self.decode(z)
        rec = px_z
        print('Prior:{}'.format(self.theta[0][0]))
        q_pi_x, alpha, _ = self.encode_sb(X)
        pi = mix_weights(q_pi_x.rsample())[:, :-1]

        means = self.theta.permute(1,0,2)
        estimate_log_gaussian_probs = []
        for i in means:
            p_mu = Independent(Normal(i, torch.ones_like(i)), 1)
            mu = p_mu.sample()
            p_z = Independent(Normal(mu, torch.ones_like(mu)), 1)
            estimate_log_gaussian_probs.append(p_z.log_prob(z))
        estimate_log_gaussian_probs = torch.stack(estimate_log_gaussian_probs)
        predict = torch.argmin(estimate_log_gaussian_probs,0)
        prob = torch.min(estimate_log_gaussian_probs,0)[0]
        sort_pi = torch.argmax(pi,-1)

        print('Prob:{}'.format(prob[:10]))
        print('Pred:{}'.format(predict[:10]))
        print('Label:{}'.format(label[:10]))
        print('Sort Pi:{}'.format(sort_pi[:10]))
        return rec


def test_network():
    model = sb_vae(28 * 28, 500, 20, 0.5)
    img = torch.randn((128, 28 * 28))
    model(img)

def train(model, train_loader, epoch, optimizer):
    model.train()
    train_loss = 0
    for i, (img,label) in enumerate(train_loader):
        optimizer.zero_grad()
        batch = img.view(-1, D).to(device)
        label = label.to(device)
        loss = model(batch)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if (i + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                epoch, i * len(img), len(train_loader.dataset),
                100. * i / len(train_loader),
                loss.item() / len(img)))
        '''
        if (i + 1) % 100 == 0:
            rec = model.sample(batch, label).detach().cpu().numpy()
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


def test(model, train_loader, epoch):
    model.eval()
    test_loss = 0
    for i, (img,label) in enumerate(train_loader):
        batch = img.view(-1, D).to(device)
        label = label.to(device)
        loss = model(batch)
        test_loss += loss.item()

    rec = model.sample(batch, label).detach().cpu().numpy()
    fig, ax = plt.subplots(4, 5)
    for i, ax in enumerate(ax.flatten()):
        if i < 10:
            plottable_image = np.reshape(img[i], (28, 28))
            ax.imshow(plottable_image)
        elif i < 20:
            plottable_image = np.reshape(rec[i - 10], (28, 28))
            ax.imshow(plottable_image)
    plt.show()
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == '__main__':
    device = torch.device('cuda:0')
    torch.manual_seed(18)
    flag = False
    if flag:
        test_network()
    else:
        D = 28*28
        train_loader, test_loader = get_mnist()
        model = sb_vae(D, hidden_dims=400, latent_dims=30, prior_alpha=10, clusters=10)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for i in range(1000):
            train(model, train_loader, i, optimizer)
            test(model, test_loader, i)

