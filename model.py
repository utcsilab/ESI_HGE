import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Bernoulli
from distributions import RiemannianNormal, WrappedNormal
from utils import Constants

from layers import GraphConvolution, GraphConvolutionK
import manifolds


class GCNModelVAE(nn.Module):
    def __init__(self, psi_input_dim, logv_input_dim, hidden_dim1, hidden_dim2, dropout, K, J, noise_dim=32, device='cpu', c=1):
        c = nn.Parameter(c * torch.ones(1), requires_grad=False)
        self.latent_dim = hidden_dim2
        self.device = device
        manifold = getattr(manifolds, 'PoincareBall')(self.latent_dim, c)
        super(GCNModelVAE, self).__init__()

        self.gc1_logv = GraphConvolution(logv_input_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2_logv = GraphConvolution(hidden_dim1, self.latent_dim, dropout, act=lambda x: x)

        self.gc1_psi = GraphConvolutionK(psi_input_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2_psi = GraphConvolutionK(hidden_dim1, self.latent_dim, dropout, act=lambda x: x)
        self.manifold = manifold
        self._pz_mu = nn.Parameter(torch.zeros(1, self.latent_dim), requires_grad=False).to(device)
        self._pz_logvar = nn.Parameter(torch.zeros(1, 1), requires_grad=False).to(device)
        self.pz = eval('WrappedNormal')
        self.qz_x = eval('WrappedNormal')
        self.qz_x_avg = eval('WrappedNormal')
        self.K = K
        self.J = J
        self.dc = InnerProductDecoder(rep=self.K, dropout=dropout, act=lambda x: x)
        self.noise_dim = noise_dim

    def sample_logv(self, x, adj):
        h_logv = self.gc1_logv(x, adj)
        logv = self.gc2_logv(h_logv, adj)
        return logv

    def sample_psi(self, rep, x, adj):
        input = x.unsqueeze(1)
        input = input.repeat(1, rep, 1)
        B = Bernoulli(0.5)
        e = B.sample(sample_shape=[input.shape[0], input.shape[1], self.noise_dim]).to(self.device)
        input_= torch.cat((input, e), dim=2)
        h_mu = self.gc1_psi(input_, adj)
        mu = self.gc2_psi(h_mu, adj)
        return mu

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        z_logvar = self.sample_logv(x, adj)
        z_log_iw = (F.softplus(z_logvar) + Constants.eta).unsqueeze(1).repeat(1, self.K, 1)
        sigma_iw1 = torch.exp(z_log_iw / 2)
        sigma_iw2 = sigma_iw1.unsqueeze(2).repeat(1, 1, self.J+1, 1)


        mu = self.sample_psi(self.K, x, adj)
        psi_iw = self.manifold.expmap0(mu)
        psi_iw_vec = psi_iw.mean(1)

        qz_x = self.qz_x(psi_iw, sigma_iw1, self.manifold)
        #pz = self.pz(self._pz_mu.mul(1.), F.softplus(self._pz_logvar).div(math.log(2)).mul(1.0), self.manifold)
        zs_sample_iw = qz_x.rsample(torch.Size([1])).squeeze()
        zs_sample_iw1 = zs_sample_iw.unsqueeze(2)
        zs_sample_iw2 = zs_sample_iw1.repeat(1, 1, self.J+1, 1)

        psi_iw_star = self.manifold.expmap0(self.sample_psi(self.J, x, adj))
        psi_iw_star0 = psi_iw_star.unsqueeze(1)
        psi_iw_star1 = psi_iw_star0.repeat(1, self.K, 1, 1)
        psi_iw_star2 = torch.cat((psi_iw_star1, psi_iw.unsqueeze(2)), dim=2)


        qz_x_avg = self.qz_x_avg(psi_iw_star2, sigma_iw2, self.manifold)

        #ker = torch.exp(-0.5 * ((zs_sample_iw2 - psi_iw_star2).pow(2)/(sigma_iw2 + 1e-10).pow(2)).sum(3))
        ker = torch.exp(qz_x_avg.log_prob(zs_sample_iw2).sum(3))

        log_H_iw_vec = torch.log(ker.mean(2) + 1e-10) - 0.5 * z_log_iw.sum(2)
        log_H_iw = log_H_iw_vec.mean(0)

        pz = self.pz(self._pz_mu.mul(1.), F.softplus(self._pz_logvar).div(math.log(2)).mul(1.0), self.manifold)
        log_prior_iw_vec = pz.log_prob(zs_sample_iw).sum(2)
        log_prior_iw = log_prior_iw_vec.mean(0)

        z_sample_iw = self.manifold.logmap0(zs_sample_iw)
        logits_x_iw = self.dc(z_sample_iw)
        reconstruct_iw = logits_x_iw

        return reconstruct_iw, log_prior_iw, log_H_iw, psi_iw_vec, psi_iw


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, rep, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act
        self.K = rep

    def forward(self,  z):

        for i in range(self.K):
            input_ = z[:, i, :].squeeze()
            input_ = F.dropout(input_, self.dropout, training=self.training)
            adj = self.act(torch.mm(input_, input_.t())).unsqueeze(2)
            if i == 0:
                output = adj
            else:
                output = torch.cat((output, adj), dim=2)
        return output
