#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/10/20 3:34
@author: merci
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/10/19 11:38
@author: merci
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/10/17 5:40
@author: merci
"""
# !/usr/bin/env python
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
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
import os


################################Utils################################################


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

####################################################################################
####################################################################################


class sb_vae(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dims, clusters=50):
        super(sb_vae, self).__init__()
        self.input_dims = input_dims
        self.latent_dims = latent_dims

        self.hidden_dims = hidden_dims
        self.clusters = clusters


        self.theta=nn.Parameter(torch.FloatTensor(self.clusters,self.latent_dims).fill_(0),requires_grad=True)
        self.log_var=nn.Parameter(torch.FloatTensor(self.clusters,self.latent_dims).fill_(0),requires_grad=True)
        self.prior=nn.Parameter(torch.FloatTensor(self.clusters).fill_(0),requires_grad=False)

        #self.alpha = nn.Parameter(torch.FloatTensor(self.clusters).fill_(0),requires_grad=True)
        #self.beta = nn.Parameter(torch.FloatTensor(self.clusters).fill_(0),requires_grad=True)

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

        self.encoder_sb = nn.Sequential(*(
                [nn.Linear(self.input_dims, self.hidden_dims),
                 nn.ReLU()] +
                [nn.Linear(self.hidden_dims, self.hidden_dims),
                 nn.ReLU()] +
                [nn.Linear(self.hidden_dims, 2 * self.clusters),
                 nn.Softplus()]
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

    def lbeta(self,a,b):
        value = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        return value

    def sb_process(self,a,b):
        weight_dirichlet_sum = (a + b)
        tmp = b / weight_dirichlet_sum
        weights_ = (a / weight_dirichlet_sum * torch.cat((torch.ones(a.size(0),1).cuda(1), tmp[:,:-1].cumprod(1)),1))
        weights_ /= torch.sum(weights_,-1,keepdim=True)
        return weights_

    def estimate_log_weights(self,a,b):
        digamma_sum = torch.digamma(a + b)
        digamma_a = torch.digamma(a)
        digamma_b = torch.digamma(b)
        value = (digamma_a - digamma_sum + torch.cat((torch.zeros(a.size(0),1).cuda(1), torch.cumsum((digamma_b - digamma_sum),1)[:,:-1]),1))
        return value

    def forward(self, X):
        qz_x, mu_, logvar_ = self.encode_Gaussian(X)
        z = qz_x.rsample()
        px_z = self.decode(z)
        log_prob_p = qz_x.log_prob(z)


        pi = self.sb_process(self.alpha, self.beta)
        esti_log_pi = self.estimate_log_weights(self.alpha,self.beta)
        log_norm_weight = -torch.sum(self.lbeta(self.alpha,self.beta))

        z0 = z.unsqueeze(1).repeat(1,self.clusters,1)

        log_scale = self.log_var/2
        const = math.log(math.sqrt(2 * math.pi))
        log_probs = -((z0 - self.theta) ** 2) / (2*torch.exp(self.log_var)) - log_scale - const
        log_prob_mm = torch.logsumexp(log_probs.mean(-1) + esti_log_pi, dim=-1)


        BCE = F.binary_cross_entropy(px_z, X.view(-1, 784), reduction='sum')
        kl = log_prob_p - log_prob_mm.mean(-1)

        elbo = BCE+ kl

        return elbo.mean()

    def ELBO_Loss(self,x):
        det=1e-10

        L_rec=0

        qz_x, z_mu, z_sigma2_log = self.encode_Gaussian(x)

        z = qz_x.rsample()

        x_pro=self.decode(z)

        L_rec+=F.binary_cross_entropy(x_pro,x)

        Loss=L_rec*x.size(1)

        q_pi_x, alpha, beta = self.encode_sb(x)
        pi = self.sb_process(alpha, beta)
        esti_log_pi = self.estimate_log_weights(alpha,beta)
        mu_c=self.theta.squeeze()
        log_sigma2_c=self.log_var.squeeze()

        p_pi = Independent(
            Beta(torch.ones_like(alpha), torch.ones_like(alpha) * 100), 1)
        kl2 = kl_divergence(q_pi_x, p_pi).mean()

        # yita_c=torch.exp(esti_log_pi+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det
        yita_c=torch.exp(torch.log(pi)+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det

        yita_c=yita_c/(yita_c.sum(1).view(-1,1))#batch_size*Clusters

        Loss+=0.5*torch.mean(torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1))

        # Loss-=torch.mean(torch.sum(yita_c*(esti_log_pi-torch.log(yita_c)),1))+0.5*torch.mean(torch.sum(1+z_sigma2_log,1))
        Loss-=torch.mean(torch.sum(yita_c*(torch.log(pi/yita_c)),1))+0.5*torch.mean(torch.sum(1+z_sigma2_log,1))


        return (Loss+kl2)


    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.clusters):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)

    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))

    def predict(self,x,y):
        qz_x, mu_, logvar_ = self.encode_Gaussian(x)
        z = qz_x.rsample()
        q_pi_x, alpha, beta = self.encode_sb(x)
        pi = self.sb_process(alpha, beta)
        mu_c=self.theta.squeeze()
        log_sigma2_c=torch.ones_like(mu_c).squeeze()
        esti_log_pi = self.estimate_log_weights(alpha, beta)

        yita_c = pi.unsqueeze(0)+torch.exp(self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))
        # yita_c2 = torch.exp(esti_log_pi)+torch.exp(self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))
        yita_c2 = torch.exp(self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))

        predict = torch.argmax(yita_c,-1).squeeze()
        predict2 = torch.argmax(yita_c2,-1).squeeze()
        return predict,predict2

    def pre_train(self,dataloader,pre_epoch=10):
        if not os.path.exists('./pretrain_model_gpu1.pk'):
            Loss=nn.MSELoss()
            opti = torch.optim.Adam(model.parameters(), lr=1e-3)

            print('Pretraining......')
            epoch_bar= tqdm(range(pre_epoch))
            for _ in epoch_bar:
                L=0
                for x,y in dataloader:
                    x=x.view(-1,784).cuda(1)

                    _,mu,_=self.encode_Gaussian(x)
                    x_=self.decoder(mu)
                    loss=Loss(x,x_)

                    L+=loss.detach().cpu().numpy()

                    opti.zero_grad()
                    loss.backward()
                    opti.step()

                epoch_bar.write('L2={:.4f}'.format(L/len(dataloader)))
            Z = []
            Y = []
            with torch.no_grad():
                for x, y in dataloader:

                    x = x.view(-1,784).cuda(1)

                    _, z, _=self.encode_Gaussian(x)
                    Z.append(z)
                    Y.append(y)

            Z = torch.cat(Z, 0).detach().cpu().numpy()
            Y = torch.cat(Y, 0).detach().numpy()

            dpgmm = GaussianMixture(n_components=self.clusters, covariance_type='diag')

            pre = dpgmm.fit_predict(Z)
            print('ARI={:.4f}%'.format(adjusted_rand_score(pre, Y)))

            # wei_c = dpgmm.weight_concentration_

            self.prior.data= torch.from_numpy(dpgmm.weights_).cuda(1).float()
            # self.alpha.data = torch.from_numpy(wei_c[0]).cuda().float()
            # self.beta.data = torch.from_numpy(wei_c[1]).cuda().float()
            self.theta.data = torch.from_numpy(dpgmm.means_).cuda().float()
            self.log_var.data = torch.log(torch.from_numpy(dpgmm.covariances_).cuda(1).float())



            torch.save(self.state_dict(), './pretrain_model_gpu1.pk')
        else:
            self.load_state_dict(torch.load('./pretrain_model_gpu1.pk'))





def test_network():
    model = sb_vae(28 * 28, 500, 20, 0.5)
    img = torch.randn((128, 28 * 28))
    model(img)


def train(model, train_loader, epoch, optimizer):
    model.train()
    train_loss = 0
    for i, (img, label) in enumerate(train_loader):
        optimizer.zero_grad()
        batch = img.view(-1, D).to(device)
        label = label.to(device)
        loss = model.ELBO_Loss(batch)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if (i + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                epoch, i * len(img), len(train_loader.dataset),
                       100. * i / len(train_loader),
                       loss.item() / len(img)))



def test(model, test_loader, epoch):
    model.eval()
    test_loss = 0
    predict = []
    predict2 = []
    labels = []
    for i, (img, label) in enumerate(test_loader):
        batch = img.view(-1, D).to(device)
        labels.append(label.numpy())
        label = label.to(device)
        #loss = model(batch)
        #test_loss += loss.item()
        pre1,pre2 = model.predict(batch,label)
        predict.append(pre1.detach().cpu().numpy())
        predict2.append(pre2.detach().cpu().numpy())


    predict = np.concatenate(predict)
    predict2 = np.concatenate(predict2)
    labels = np.concatenate(labels)

    ARI1 = adjusted_rand_score(labels,predict)
    ARI2 = adjusted_rand_score(labels,predict2)
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
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('ARI1:{:.3f}'.format(ARI1))
    print('ARI2:{:.3f}'.format(ARI2))
    return ARI1


if __name__ == '__main__':
    device = torch.device('cuda:1')
    torch.manual_seed(18)
    flag = 2
    if flag == 0:
        test_network()
    elif flag == 1:
        D = 28 * 28
        train_loader, test_loader = get_mnist()
        model = sb_vae(D, hidden_dims=500, latent_dims=20, clusters=10)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.pre_train(train_loader,50)
        ARI=0
        for i in range(500):
            train(model, train_loader, i, optimizer)
            ARIT = test(model, test_loader, i)
            if ARIT > ARI:
                torch.save(model.state_dict(), './train_model_dp.pk')
                ARI = ARIT
                print('Model Saved')
    elif flag==2:
        D=28*28
        model = sb_vae(28*28, hidden_dims=500, latent_dims=20, clusters=10)
        model.to(device)
        train_loader, test_loader = get_mnist()
        tmp_model = torch.load('./train_model_dp.pk')
        model.load_state_dict(tmp_model)
        ARIT = test(model, test_loader, 0)