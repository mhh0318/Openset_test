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
def log_normal(x, m, logv):
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
    log_det = -0.5 * torch.sum(logv, dim=-1)
    # print("log_det", log_det.size())
    log_exp = -0.5 * torch.sum((x - m) ** 2 / torch.exp(logv), dim=-1)

    log_prob = const + log_det + log_exp

    ################################################################################
    # End of code modification
    ################################################################################
    return log_prob

def log_mean_exp(x):
    max_, _ = torch.max(x, 1, keepdim=True)
    return torch.log(torch.mean(torch.exp(x - max_), 1)) + torch.squeeze(max_)

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

def cluster_acc(Y_pred, Y):
    from scipy.optimize import linear_sum_assignment as linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i[0],i[1]] for i in ind])*1.0/Y_pred.size, w
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

        # base = Independent(Normal(torch.zeros((1, self.latent_dims)), torch.ones((1, self.latent_dims))), 1)
        # self.theta = nn.Parameter(base.rsample([clusters]).permute(1, 0, 2), requires_grad=True)

        self.theta=nn.Parameter(torch.FloatTensor(self.clusters,self.latent_dims).fill_(0),requires_grad=True)
        self.var=nn.Parameter(torch.FloatTensor(self.clusters,self.latent_dims).fill_(0),requires_grad=True)

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

        qz_x, mu_, logvar_ = self.encode_Gaussian(X)
        z = qz_x.rsample()
        px_z = self.decode(z)
        log_q_z_x= qz_x.log_prob(z)  #q(z|x)

        q_pi_x, alpha, beta = self.encode_sb(X)
        pi = mix_weights(q_pi_x.rsample())[:, :-1]
        log_weight = torch.log(pi)   #q(c|x)

        z0 = z.unsqueeze(1).repeat(1,self.clusters,1)

        log_scale = torch.ones_like(self.theta)
        const = math.log(math.sqrt(2 * math.pi))
        log_probs = -((z0 - self.theta) ** 2) / 2 - log_scale - const
        log_prob_mm = torch.logsumexp(log_probs, dim=-1)


        kl = log_q_z_x.unsqueeze(1).repeat(1,self.clusters)+log_weight-log_prob_mm

        BCE = F.binary_cross_entropy(px_z, X.view(-1, 784), reduction='sum')

        p_pi = Independent(
            Beta(torch.ones_like(alpha), torch.ones_like(alpha) * self.prior_alpha), 1)
        kl2 = kl_divergence(q_pi_x, p_pi).mean()
        nelbo = torch.mean(BCE+kl+kl2)

        return nelbo

    def ELBO_Loss(self,x):
        det=1e-10

        L_rec=0

        qz_x, z_mu, z_sigma2_log = self.encode_Gaussian(x)

        z = qz_x.rsample()

        x_pro=self.decode(z)

        L_rec+=F.binary_cross_entropy(x_pro,x)

        Loss=L_rec*x.size(1)


        q_pi_x, alpha, beta = self.encode_sb(x)
        pi = mix_weights(q_pi_x.rsample())[:, :-1]
        mu_c=self.theta.squeeze()
        log_sigma2_c=self.var.squeeze()
        p_pi = Independent(
            Beta(torch.ones_like(alpha), torch.ones_like(alpha) * self.prior_alpha), 1)
        kl2 = kl_divergence(q_pi_x, p_pi).mean()

        yita_c=torch.exp(torch.log(pi)+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det

        yita_c=yita_c/(yita_c.sum(1).view(-1,1))#batch_size*Clusters

        Loss+=0.5*torch.mean(torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1))

        Loss-=torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1))+0.5*torch.mean(torch.sum(1+z_sigma2_log,1))


        return (Loss+kl2)


    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.clusters):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)

    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))

    def predict(self,x):
        pz_x,z_mu, z_sigma2_log = self.encode_Gaussian(x)
        #z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        z = pz_x.rsample()
        q_pi_x, alpha, beta = self.encode_sb(x)
        pi = mix_weights(q_pi_x.rsample())[:, :-1]
        mu_c=self.theta.squeeze()
        log_sigma2_c=torch.ones_like(mu_c).squeeze()

        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))

        yita=yita_c.detach().cpu().numpy()
        return np.argmax(yita,axis=-1)

    def pre_train(self,dataloader,pre_epoch=10):
        if not os.path.exists('./pretrain_model_dp.pk'):
            Loss=nn.MSELoss()
            opti = torch.optim.Adam(model.parameters(), lr=1e-3)

            print('Pretraining......')
            epoch_bar= tqdm(range(pre_epoch))
            for _ in epoch_bar:
                L=0
                for x,y in dataloader:
                    x=x.view(-1,784).cuda()

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

                    x = x.view(-1,784).cuda()

                    _, z, _=self.encode_Gaussian(x)
                    Z.append(z)
                    Y.append(y)

            Z = torch.cat(Z, 0).detach().cpu().numpy()
            Y = torch.cat(Y, 0).detach().numpy()

            dpgmm = BayesianGaussianMixture(n_components=self.clusters, covariance_type='diag')

            pre = dpgmm.fit_predict(Z)
            print('ARI={:.4f}%'.format(adjusted_rand_score(pre, Y)))
            print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0]))

            # self.pi_.data = torch.from_numpy(gmm.weights_).cuda().float()
            self.theta.data = torch.from_numpy(dpgmm.means_).cuda().float()
            self.var.data = torch.log(torch.from_numpy(dpgmm.covariances_).cuda().float())
            '''
            gmm = GaussianMixture(n_components=self.clusters, covariance_type='diag')
            pre1 = gmm.fit_predict(Z)
            print('ARI={:.4f}%'.format(adjusted_rand_score(pre1, Y)))
            print('Acc={:.4f}%'.format(cluster_acc(pre1, Y)[0]))
            '''
            torch.save(self.state_dict(), './pretrain_model_dp.pk')
        else:


            self.load_state_dict(torch.load('./pretrain_model_dp.pk'))




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


def test(model, test_loader, epoch):
    model.eval()
    test_loss = 0
    predict = []
    labels = []
    for i, (img, label) in enumerate(test_loader):
        batch = img.view(-1, D).to(device)
        labels.append(label.numpy())
        label = label.to(device)
        #loss = model(batch)
        #test_loss += loss.item()
        pre = model.predict(batch)
        predict.append(pre.squeeze())


    predict = np.concatenate(predict)
    labels = np.concatenate(labels)

    ARI = adjusted_rand_score(labels,predict)
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
    print('ARI:{:.3f}'.format(ARI))
    return ARI


if __name__ == '__main__':
    device = torch.device('cuda:0')
    torch.manual_seed(18)
    flag = False
    if flag:
        test_network()
    else:
        D = 28 * 28
        train_loader, test_loader = get_mnist()
        model = sb_vae(D, hidden_dims=500, latent_dims=50, prior_alpha=0.5, clusters=50)
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