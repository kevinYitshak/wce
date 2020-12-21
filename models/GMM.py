import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent

import numpy as np
import math

class GMM(nn.Module):

    def __init__(self, device):
        super(GMM, self).__init__()
        '''
        No of data samples = height * weight [h * w]
        No of featues = channels [c]
        No of GMMs = k
        Batch size = b
        '''
        self.device = device
        # self.z = z # [b, c, h*w]
        # self.g = g # [b, k, h*w]

        # self.softmax = nn.Softmax(dim=1)

    def cal_phi(self, g):
        '''
        phi => [b, k]
        -- torch.sum(phi, dim=1) == 1 --
        '''
        # g = self.softmax(g)
        b, k, h, w = g.size()
        g = g.view(b, k, h * w)

        phi =  torch.sum(g, dim=2) / g.size(2) # [b, k]

        # print('sum phi: ', torch.sum(phi, dim=1))
        return phi
    
    def cal_mean(self, z, g):
        '''
        sum (g_i^k * z^k) / sum g_i^k
        z = [b, c, h*w]
        g = [b, k, h*w]
        
        mu => [b, k, c]
        '''
        b, c, h, w = z.size()
        _, k, _, _ = g.size()

        z = z.view(b, c, h * w)
        g = g.view(b, k, h * w)

        mu_ = torch.zeros(b, k, c).to(self.device)
        for i in range(k):
            g_cl = g[:, i, :].clone()
            gz = g_cl.unsqueeze(1) * z
            # print(gz.shape)
            gz = torch.sum(gz, dim=-1)
            mu_[:, i, :] = gz
        
        mu = mu_ / (torch.sum(g, dim=2, keepdim=True) + 1e-6)

        # mu = torch.bmm(g, z.permute(0, 2, 1).contiguous()) / (torch.sum(g, dim=2, keepdim=True) + 1e-6) # [b, k, c]
        mu = self._l2norm(mu, dim=-1)
        return mu

    def cal_cov(self, z, g, mu):
        '''
        sum (g_i^k * [(z^k - mu_i) * (z^k - mu_i).T]) / sum g_i^k
        z = [b, c, h*w]
        g = [b, k, h*w]
        mu = [b, k, c]

        z_mu = [c, h*w]
        
        cov => [b, k, c, c]
        '''
        b, c, h, w = z.size()
        _, k, _, _ = g.size()

        z = z.view(b, c, h * w)
        g = g.view(b, k, h * w)

        cov = torch.zeros(b, k, c, c)

        for i in range(b):
            for j in range(k):
                z_mu = z[i, :, :] - mu[i, j, :].unsqueeze(-1) # [c, h*w]
                z_muT = z_mu.t()
                g_c = g[i, j, :].clone()

                cov[i, j, :, :] = torch.sum((g_c.unsqueeze(-1).unsqueeze(-2) \
                                    * torch.matmul(z_mu, z_muT).unsqueeze(0)), dim=0) \
                                    / torch.sum(g_c, dim=0) # [c, c]

                cov_c = cov[i, j, :, :].clone()
                
                cov_c = self.get_pd(cov_c, c) # postitive semi defnite -> postitive defnite
                cov[i, j, :, :] = self._l2norm(cov_c, dim=(0, 1))
                # print(np.all(np.linalg.eigvals(cov[i, j, :, :].detach().cpu().numpy()) > 0))

        return cov

    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    def is_pos_def(self, x):
        return np.all(np.linalg.eigvals(x.detach().cpu().numpy()) > 0)

    def get_pd(self, x, size):
        d = x.diag()
        d = torch.where(d > 1e-6, d, 1e-6*torch.ones(d.size()))
        x = x + torch.eye(size) * d 
        return x

    def cal_energy(self, z, phi, mu, cov):
        '''
        z = [b, c, h*w]
        g = [b, k, h*w]

        phi => [b, k]
        mu => [b, k, c]
        cov => [b, k, c, c]
        '''
        b, c, h, w = z.size()
        _, k, _, = mu.size()

        z = z.view(b, c, h * w)

        energy = torch.zeros(b, k)

        for i in range(b):
            for j in range(k):
                z_ = z[i, :, :] #- mu[i, j, :].unsqueeze(-1)) / z[i, :, :].var() # [c, h*w] data centering
                mu_ = mu[i, j, :].clone() # [c]
                cov_ = cov[i, j, :, :].clone() #[c, c]

                mixture = MultivariateNormal(mu_.to(self.device), cov_.to(self.device))
                log_prob = mixture.log_prob(z_.permute(1, 0))
                weighted_logprob = log_prob + phi[i, j]

                log_sum = torch.logsumexp(weighted_logprob, dim=0)
                energy[i, j] = log_sum

        energy = -((torch.mean(energy, dim=(0, 1)))) / (h * w) 
        print('energy: ', energy)
        return energy

    def cal_energy_new(self, z, phi, mu, cov):
        '''
        z = [b, c, h*w]
        g = [b, k, h*w]

        phi => [b, k]
        mu => [b, k, c]
        cov => [b, k, c, c]
        '''

        b, c, h, w = z.size()
        _, k, _, = mu.size()

        z = z.view(b, c, h * w) # 2 x 256 x 2116
        # phi - 2x4
        # mu - 2x4x256
        # cov - 2x4x256x256
        # energy = torch.zeros(b, k)

        mix = Categorical(phi.to(self.device))
        # print(mix)
        comp = Independent(MultivariateNormal(mu.to(self.device), cov.to(self.device)), reinterpreted_batch_ndims=0)
        # print('COMP: ', comp.batch_shape, comp.event_shape) # torch.Size([b, k]) torch.Size([c])
        gmm = MixtureSameFamily(mix, comp)
        # print('GMM: ', gmm.batch_shape, gmm.event_shape) # torch.Size([b]) torch.Size([c])

        log_prob = gmm.log_prob(z.permute(2, 0, 1).to(self.device)) #[samples, b]
        # print('log_porb: ', log_prob.shape) 
        energy = - log_prob.mean()

        return energy, gmm

if __name__ == '__main__':

    z = torch.randn(2, 128, 46, 46)
    g = torch.randn(2, 4, 46, 46)

    gmm = GMM()

    phi = gmm.cal_phi(g)
    print('phi: ', phi.shape)

    mu = gmm.cal_mean(z, g)
    print('mu: ', mu.shape)

    cov = gmm.cal_cov(z, g, mu)
    print('cov: ', cov.shape)
    # print(cov)
    energy = gmm.cal_energy(z, phi, mu, cov)
    print('loss_gmm: ', energy)