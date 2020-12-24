#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/10/12 21:44
@author: merci
"""
import torch
import torch.nn as nn
import torch.distributions as D
from torch.nn import functional as F

class vanillaVAE(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dims):
        super(vanillaVAE, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.latent_dims = latent_dims
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_dims, 2 * self.latent_dims),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_dims, self.input_dims),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dims, 10)
        )

    def encode(self, x):
        inf_para = self.encoder(x)
        mu, logvar = inf_para[:, :self.latent_dims], inf_para[:, self.latent_dims:]
        return D.Independent(D.Normal(mu,torch.exp(0.5*logvar)),1), mu, logvar

    def decode(self, z):
        logit = self.decoder(z)
        return logit

    def forward(self, x):
        qz_x, mu, logvar = self.encode(x)
        z = qz_x.rsample()
        recon =self.decode(z)
        predictions = self.classifier(z)
        return recon, predictions, z, mu, logvar

    def sample(self,x):
        qz_x, mu, var = self.encode(x)
        z = qz_x.rsample()
        px_z =self.decode(z)
        rec = px_z.reshape(-1,28,28)
        return rec

    def dpsample(self, x, IGMM):
        z = torch.from_numpy(IGMM.sample(10)[0]).float().cuda()
        return self.decode(z)



if __name__ == '__main__':
    test_in = torch.randn(16,784).cuda()
    model = vanillaVAE(28*28,400,20).cuda()
    out = model(test_in)