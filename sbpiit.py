#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/10/24 23:03
@author: merci
"""

import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from scipy.stats import mode

device = "cpu"

torch.autograd.set_detect_anomaly(True)

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


class VAE_NP(nn.Module):
    def __init__(self, latent_variable_dim, alpha=1.0, rholr=10e-12):
        super(VAE_NP, self).__init__()

        ### Global Params
        self.eps1 = torch.tensor(10e-6).float()
        self.eps2 = torch.tensor(10e-4).float()
        self.inter = 400

        # V : Stick breaking : Beta {kumaraswamy}
        self.aeys = nn.Parameter(torch.rand(1, latent_variable_dim) + 1)
        self.bees = nn.Parameter(torch.rand(1, latent_variable_dim) + 1)

        self.unif_sampler = torch.distributions.uniform.Uniform(0, 1)

        # IBP prior
        self.alpha = alpha
        self.euler_constant = np.e

        ### Encoder part
        self.fc1 = nn.Linear(784, self.inter)

        # A : Gaussian
        self.weight_enc_mean = nn.Parameter(torch.randn(latent_variable_dim, self.inter) * 0.01)
        self.weight_enc_std = nn.Parameter(torch.randn(latent_variable_dim, self.inter) * 0.1)

        # Z : Bernoulli
        self.phi = nn.Parameter(torch.randn(self.inter, latent_variable_dim) * 0.001)

        # Gumbel Softmax params
        self.temperature = 10
        self.t_prior = 0.5  # prior lambda
        self.gumbel_sampler = torch.distributions.gumbel.Gumbel(0, 1)

        ### Decoder part
        self.weight_dec = nn.Parameter(torch.randn(self.inter, latent_variable_dim, ) * 0.01)

        self.fc4 = nn.Linear(self.inter, 784)

        ### Russian Roulette part
        self.rhos = torch.zeros(latent_variable_dim + 1, 1) + 0.5
        self.rholr = rholr

        ## Optimizer
        self.optimizer = None
        self.K = latent_variable_dim
        self.max_K = latent_variable_dim

        self.grads = {
            'aeys': 0,
            'bees': 0,
            'weight_enc_mean': 0,
            'weight_enc_std': 0,
            'phi': 0,
            'weight_dec': 0,
            'fc1_weight': 0,
            'fc4_weight': 0,
            'fc1_bias': 0,
            'fc4_bias': 0,
        }

        self.coordinate = 1

    def reparameterize_gaussian(self, log_var, mu):
        s = torch.exp(0.5 * log_var) + self.eps2
        eps = torch.rand_like(s) - 0.5  # generate a iid standard normal same shape as s
        return eps.mul(s).add_(mu)

    def reparameterize_gumbel_kumaraswamy(self, inter_z):

        N, K = inter_z.shape
        sample_size = 10

        U = self.unif_sampler.sample([N, K, sample_size])
        G1 = self.unif_sampler.sample([N, K, sample_size])
        logit_G1 = G1.log() - (1 - G1).log()

        V = (1 - U.pow(1 / self.aeys.exp()[:, :K].view(-1, K, 1))).pow(1 / self.bees.exp()[:, :K].view(-1, K, 1))

        pi = torch.zeros_like(V) + 1
        for i in range(K):
            for j in range(i + 1):
                pi[:, i, :] *= V[:, j, :]

        rand_num = torch.rand_like(pi)
        rand_logit = (rand_num / (1 - rand_num)).log()

        logit_pi = ((pi + self.eps1) / (1 - pi + self.eps1)).log()
        alpha = (logit_pi + inter_z.view(N, K, 1)).sigmoid().pow(1)
        logit_alpha = (alpha + self.eps1) / (1 - alpha + self.eps1)

        z1 = (logit_alpha.log() + logit_G1) / self.temperature

        y = z1.sigmoid()

        return y, alpha, pi

    def forward(self, input, k):
        x = input.view(-1, 784)
        N, D = x.shape
        x = torch.sigmoid(self.fc1(x))

        if (k == 0):
            k = self.get_current_K()

        log_s = F.linear(x, self.weight_enc_std[:k, :])
        #         log_s[log_s>10]=10.0
        #         log_s[log_s<-10]=-10.0
        m = F.linear(x, self.weight_enc_mean[:k, :])

        inter_z = self.phi[:, :k].transpose(0, 1)  # K x self.inter0
        inter_z = F.linear(x, inter_z)  # N x K

        z, gi, pi = self.reparameterize_gumbel_kumaraswamy(inter_z)  # N x K

        a = self.reparameterize_gaussian(log_s, m)  # N x K

        az = a * (z.mean(dim=-1).view(N, k))

        x = self.decode(a, k)

        return x, m, log_s, z, pi, gi

    def decode(self, z, k):
        x = torch.sigmoid(F.linear(z, self.weight_dec[:, :k]))
        x = self.fc4(x)
        x = torch.sigmoid(x)

        return x

    def add_k_node(self, k):
        # Add k latent features ...
        if (k == 0):
            return
        with torch.no_grad():
            self.aeys = nn.Parameter(torch.cat((self.aeys, torch.rand(1, k) + 1), 1))
            self.bees = nn.Parameter(torch.cat((self.bees, torch.rand(1, k) + 1), 1))

            self.phi = nn.Parameter(torch.cat((self.phi, torch.randn((self.inter), k)), 1) * 0.001)

            self.weight_enc_mean = nn.Parameter(
                torch.cat((self.weight_enc_mean, torch.randn(k, self.inter)), 0) * 0.001)
            self.weight_enc_std = nn.Parameter(torch.cat((self.weight_enc_std, torch.randn(k, self.inter)), 0) * 0.001)
            self.weight_dec = nn.Parameter(torch.cat((self.weight_dec, torch.randn(self.inter, k)), 1) * 0.001)

            self.rhos = torch.cat((self.rhos, torch.zeros(k, 1) + 0.5), 0)

    def del_k_node(self, k):
        # Retain k Latent Features ...
        if (k == 0 or k == self.weight_dec.shape[1]):
            return
        with torch.no_grad():
            c_K = self.weight_dec.shape[1]

            self.aeys = nn.Parameter(list(torch.split(self.aeys, c_K - k, 1))[0])
            self.bees = nn.Parameter(list(torch.split(self.bees, c_K - k, 1))[0])

            self.phi = nn.Parameter(list(torch.split(self.phi, c_K - k, 1))[0])

            self.weight_enc_mean = nn.Parameter(list(torch.split(self.weight_enc_mean, c_K - k, 0))[0])
            self.weight_enc_std = nn.Parameter(list(torch.split(self.weight_enc_std, c_K - k, 0))[0])
            self.weight_dec = nn.Parameter(list(torch.split(self.weight_dec, c_K - k, 1))[0])

            self.rhos = list(torch.split(self.rhos, c_K - k + 1, 0))[0]

    def get_current_K(self):
        return self.K

    def constraint_proj(self):
        with torch.no_grad():
            #             self.aeys[self.aeys < 0.001] = 0.001
            #             self.bees[self.bees < 0.001] = 0.001
            self.rhos[self.rhos < self.eps2] = self.eps2
            self.rhos[self.rhos > 1 - self.eps2] = 1 - self.eps2

    def store_grads(self, l, omr, k):
        self.optimizer.zero_grad()
        l.backward()

        self.grads['aeys'] = self.grads['aeys'] + self.aeys.grad * omr
        self.grads['bees'] = self.grads['bees'] + self.bees.grad * omr
        self.grads['weight_enc_mean'] = self.grads['weight_enc_mean'] + self.weight_enc_mean.grad * omr
        self.grads['weight_enc_std'] = self.grads['weight_enc_std'] + self.weight_enc_std.grad * omr
        self.grads['phi'] = self.grads['phi'] + self.phi.grad * omr
        self.grads['weight_dec'] = self.grads['weight_dec'] + self.weight_dec.grad * omr

        if (k == self.K):
            self.grads['fc1_weight'] = self.grads['fc1_weight'] + self.fc1.weight.grad * omr
            self.grads['fc4_weight'] = self.grads['fc4_weight'] + self.fc4.weight.grad * omr
            self.grads['fc1_bias'] = self.grads['fc1_bias'] + self.fc1.bias.grad * omr
            self.grads['fc4_bias'] = self.grads['fc4_bias'] + self.fc4.bias.grad * omr

    def use_grads(self, lr=0.1):

        k = self.coordinate

        if (k == 0):
            k = self.get_current_K()

        cK = self.get_current_K()
        mask = (torch.arange(cK) == k - 1).float()

        for key in self.grads.keys():
            if (self.grads[key] is None):
                self.grads[key] = 0

        self.aeys.data[:, :cK] -= lr * self.grads['aeys']  # *mask.view(-1,cK)
        self.bees.data[:, :cK] -= lr * self.grads['bees']  # *mask.view(-1,cK)
        self.weight_enc_mean.data[:cK, :] -= lr * self.grads['weight_enc_mean']  # *mask.view(cK,-1)
        self.weight_enc_std.data[:cK, :] -= lr * self.grads['weight_enc_std']  # *mask.view(cK,-1)
        self.phi.data[:, :cK] -= lr * self.grads['phi']  # *mask.view(-1,cK)
        self.weight_dec.data[:, :cK] -= lr * self.grads['weight_dec']  # *mask.view(1,cK)
        self.fc1.weight.data -= lr * self.grads['fc1_weight']
        self.fc4.weight.data -= lr * self.grads['fc4_weight']
        self.fc1.bias.data -= lr * self.grads['fc1_bias']
        self.fc4.bias.data -= lr * self.grads['fc4_bias']

        for key in self.grads.keys():
            self.grads[key] = 0

        self.coordinate += 1
        if (self.coordinate == cK + 1):
            self.coordinate = 1

def loss(input_image, recon_image):
    CE = F.binary_cross_entropy(recon_image, input_image.view(-1, 784), reduction='sum')
    return CE

def retain_k_nodes(model, new_K=0):

    current_K = model.weight_dec.shape[1]
    if (current_K < new_K):
        model.add_k_node(new_K - current_K)
    elif (current_K > new_K):
        model.del_k_node(current_K - new_K)
    else:
        pass

    optimizer = torch.optim.SGD(model.parameters(), 0.1)

    return optimizer

def get_kth_trunc_loss(model, images, K=0):
    N = images.shape[0]

    recon_image, log_var, mu, z, pi, gi = model(images, K)
    #     print((recon_image != recon_image).sum())
    softplus = nn.Softplus()
    eps = model.eps1

    if (K == 0):
        K = model.aeys.shape[1]

    log_var = log_var[:, :K]
    mu = mu[:, :K]
    z = z[:, :K]
    pi = pi[:, :K]
    gi = gi[:, :K]

    KL_gauss = -0.5 * (1 + log_var.sum() - mu.pow(2).sum() - log_var.exp().sum())
    KL_gauss /= N

    KL_kuma = ((model.aeys.exp() - model.alpha) / (model.aeys.exp())) * (
                -model.euler_constant - torch.digamma(model.bees.exp()) - 1 / model.bees.exp())
    KL_kuma += (model.aeys.exp().log() + model.bees.exp().log())
    KL_kuma += -(model.bees.exp() - 1) / (model.bees.exp())

    KL_kuma = torch.sum(KL_kuma[:, :K])

    logit_pi = (pi + eps).log() - (1 - pi + eps).log()
    logit_x = (z + eps).log() - (1 - z + eps).log()
    logit_gi = (gi + eps).log() - (1 - gi + eps).log()

    tau = model.temperature
    tau_prior = model.t_prior

    exp_term_p = logit_pi - logit_x * (tau_prior)
    exp_term_q = logit_gi - logit_x * (tau)

    log_tau = torch.log(torch.tensor(model.temperature, requires_grad=False))

    log_pz = log_tau + exp_term_p - 2.0 * softplus(exp_term_p)
    log_qz = log_tau + exp_term_q - 2.0 * softplus(exp_term_q)

    KL_gumb = (log_qz - log_pz)

    KL_gumb[KL_gumb != KL_gumb] = 0
    KL_gumb[KL_gumb < 0] = 0
    #     print(KL_gumb.shape)
    KL_gumb = torch.sum(KL_gumb.mean(dim=-1))  # .abs()
    KL_gumb /= N

    l = loss(images, recon_image) / N
    KL_l = KL_gauss + KL_kuma + KL_gumb

    print(l, KL_gauss, KL_kuma, KL_gumb)
    return l + KL_l

def rrs_loss(model, images, curr_K):

    l = torch.zeros(curr_K + 1, 1)
    for i in range(1, curr_K + 1):
        loss = get_kth_trunc_loss(model, images, K=i)
        model.store_grads(loss, 1 - model.rhos[i], i)
        #         model.use_grads(0.01)
        l[i, :] = loss

    one_minus_rho = (1 - model.rhos[0:curr_K + 1]).view(curr_K + 1, 1)

    return l, one_minus_rho


def train_step(model, images, sample_max=5, sample=False, keep_graph=False):
    """ sample a trucation level and then do the same"""

    curr_K = model.get_current_K()
    model.rhos[0] = 1.0
    model.optimizer = retain_k_nodes(model, new_K=model.max_K + 1)
    model.optimizer = retain_k_nodes(model, new_K=model.max_K)

    if (sample):

        curr_K = model.get_current_K()
        rhos = list(model.rhos)

        L = len(rhos)

        samples = []
        for i in range(sample_max):

            k = 1
            while (True):
                u = np.random.uniform()
                if (u > rhos[k]):
                    samples.append(k)
                    break
                k += 1

                if (k > L - 1):
                    rhos.append(0.5)

        samples.sort()
        new_value = int(np.mean(samples[-5:]))

        if (new_value > model.max_K):
            model.optimizer = retain_k_nodes(model, new_K=new_value)
            model.K = new_value
            model.max_K = new_value
        else:
            model.optimizer = retain_k_nodes(model, new_K=model.max_K)
            model.K = new_value


    else:
        new_value = curr_K

    print("Current Truncated Level :", new_value, 'current_coordinate :', model.coordinate)
    print(model.rhos)

    model.optimizer.zero_grad()
    curr_K = model.get_current_K()

    l, one_minus_rho = rrs_loss(model, images, curr_K)
    l[l != l] = 0
    l_final_params = (l * one_minus_rho).sum()
    #     l_final_params.backward(retain_graph = True)

    #     model.optimizer.step()

    #     model.use_grads()

    try:
        model.use_grads(0.01)
    except:
        print('Failed :(')

    ws = torch.zeros(curr_K + 1, curr_K)

    for k in range(1, curr_K + 1):
        for i in range(k - 1, curr_K):
            if (i < k - 1):
                ws[k, i] = 0
            elif (i == k - 1):
                ws[k, i] = 1 / (model.rhos[k] - 1)
            else:
                ws[k, i] = 1 / model.rhos[k]

    rho_grads = torch.mm(ws, (-l * one_minus_rho)[1:].view(curr_K, 1))
    rho_grads[rho_grads != rho_grads] = 0.0
    rho_logit = ((model.rhos + model.eps1).log() - (1 - model.rhos + model.eps1).log())[:curr_K + 1]
    sig_rho = rho_logit.sigmoid()

    rho_logit[:curr_K + 1, :] = rho_logit[:curr_K + 1, :] - model.rholr * (
                sig_rho * (1 - sig_rho) * rho_grads.view(-1, 1))
    model.rhos[:curr_K + 1, :] = rho_logit.sigmoid()[:curr_K + 1, :]

    #     model.constraint_proj()
    #     model.optimizer.step()

    return l_final_params

vae = VAE_NP(5,10,10e-4)
train_loss = []
vae.temperature = 1.0

trainloader, testloader = get_mnist()

# vae.optimizer = torch.optim.Adam(vae.parameters(), 0.01)
# optimizer = torch.optim.Adam(vae.parameters(), 0.01)
for epoch in range(50):

    for i, data in enumerate(trainloader, 0):
        images, labels = data
        images = images.to(device)

        try:
            if (i % 100 == 0 and False):
                l = train_step(vae, images, 50, True)
            else:
                l = train_step(vae, images, 50, False)

            train_loss.append(l.item())
        except:
            print("Failed")

        vae.temperature /= 1.1
        if (vae.temperature < .005):
            vae.temperature = .005

        #         optimizer.zero_grad()

        #         l = get_kth_trunc_loss(vae, images, K = 0)
        #         l.backward()

        #         train_loss.append(l.item() / len(images))
        #         optimizer.step()

        if (i % 1 == 0):
            print("Epoch no :", epoch + 1, "batch_no :", i, "curr_loss :", train_loss[-1])

plt.plot(train_loss)
plt.show()

def show_images(images):
    images = torchvision.utils.make_grid(images)
    show_image(images[0])

def show_image(img):
    plt.imshow(img, cmap='gray')
    plt.show()
with torch.no_grad():
    for i, data in enumerate(testloader, 7):
        images, labels = data
        images = images.to(device)
        recon_image, s, mu, z, pi, gi = vae(images, 0)
        recon_image_ = recon_image.view(128, 1, 28, 28)
        if i % 100 == 0:
            show_images(recon_image_)

