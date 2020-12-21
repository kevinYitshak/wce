import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
from models.CE import estimator, compression

# import itertools
# from utils import *

class DaGMM(nn.Module):
    """Residual Block."""
    def __init__(self, c=64, latent_dim=65, k=4):
        super(DaGMM, self).__init__()

        # self.estimation = estimator(latent_dim, k)
        

        # layers = []
        # layers += [nn.Linear(512,350)]
        # layers += [nn.Tanh()]        
        # layers += [nn.Linear(350,255)]
        # layers += [nn.Tanh()]        
        # layers += [nn.Linear(64,32)]
        # layers += [nn.Tanh()]        
        # layers += [nn.Linear(10,1)]

        # self.encoder = nn.Sequential(*layers)


        # layers = []
        # layers += [nn.Linear(255,350)]
        # layers += [nn.Tanh()]        
        # layers += [nn.Linear(350,512)]
        # layers += [nn.Tanh()]        
        # layers += [nn.Linear(128,256)]
        # layers += [nn.Tanh()]        
        # layers += [nn.Linear(60,2116)]

        # self.decoder = nn.Sequential(*layers)

        # layers = []
        # layers += [nn.Linear(latent_dim,64)]
        # layers += [nn.Tanh()]
        # layers += [nn.Dropout(p=0.5)]
        # layers += [nn.Linear(64,n_gmm)]
        # layers += [nn.Softmax(dim=1)]


        # self.estimation = nn.Sequential(*layers)

        # self.register_buffer("phi", torch.zeros(n_gmm))
        # self.register_buffer("mu", torch.zeros(n_gmm,latent_dim))
        # self.register_buffer("cov", torch.zeros(n_gmm,latent_dim,latent_dim))

    def relative_euclidean_distance(self, a, b):
        return (a-b).norm(2, dim=1) #/ a.norm(2, dim=1)

    def forward(self, fg, bg, q, cos_fq, cos_bq):

        enc_fg, enc_bg, enc_q = self.compression(fg, bg, q) # [64, 46, 46]
        # print('enc_size: ', enc_fg.size())
        # dec = self.decoder(enc)
        # print('dec_size: ', dec.size())

        rec_cosine_fq = F.cosine_similarity(enc_fg, enc_q, dim=1)
        rec_cosine_bq = F.cosine_similarity(enc_bg, enc_q, dim=1)
        # print('rec_size: ', rec_cosine_fq.size())
        # rec_euclidean = self.relative_euclidean_distance(x, dec)
        # print('rec_euclidean_size: ', rec_euclidean.min(), rec_euclidean.max())
        
        z_fg = torch.cat([enc_fg, rec_cosine_fq.unsqueeze(1)], dim=1)
        z_bg = torch.cat([enc_bg, rec_cosine_bq.unsqueeze(1)], dim=1)
        z_q = torch.cat([enc_q, rec_cosine_fq.unsqueeze(1)], dim=1)
        #torch.cat([enc_bg, rec_cosine_bq.unsqueeze(-1)], dim=1)
        # print('z_size: ', z_fg.size())

        gamma_fg, gamma_bg, gamma_q = self.estimation(z_fg, z_bg, z_q)
        # print('gamma_size:', gamma_fg.size())
        # print(gamma)
        return enc_fg, enc_bg, enc_q, z_fg, z_bg, z_q, gamma_fg, gamma_bg, gamma_q

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        # print('batch_size: ', N)
        # K
        sum_gamma = torch.sum(gamma, dim=0)
        # print('sum_gamma_size: ', sum_gamma.size())
        # K
        phi = (sum_gamma / N)
        # print('phi_min: ', phi.min())

        # self.phi = phi.data

 
        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        # print('mu_size: ', mu.size())
        # print('z_min: ', z.min())
        # print('mu_min: ', mu.min())
        # self.mu = mu.data
        # z = N x D
        # mu = K x D
        # gamma N x K
        
        # z_mu = N x K x D
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        # print('z_mu_min: ', z_mu.min())
        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        # print('z_mu_outer_min: ', z_mu_outer.min())
        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        # print(cov.size())
        # print('cov_min: ', cov.min())
        # for i in range(self.n_gmm):
        #     cov_ = cov[i, :, :].clone()
        #     cov[i, :, :] = self.get_pd(cov_, 256)
        # self.cov = cov.data
        return phi, mu, cov

    # def get_pd(self, x, size):
    #     d = x.diag()
    #     d = torch.where(d > 1e-6, d, 1e-6*torch.ones(d.size()))
    #     x = x + torch.eye(size) * d 
    #     return x

    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        
        k, D, _ = cov.size()

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))
        print('z_mu: ', z_mu.shape)
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-6
        for i in range(k):
            # K x D x D
            cov_k = cov[i] #+ (torch.eye(D)*eps)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))

            #det_cov.append(np.linalg.det(cov_k.data.cpu().numpy()* (2*np.pi)))
            det_cov.append((torch.cholesky(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())
            # print('cov_diag: ', cov_diag.shape)
        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = torch.cat(det_cov).cpu()
        #det_cov = to_var(torch.from_numpy(np.float32(np.array(det_cov))))

        zTcov = z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0)
        print('zTcov: ', zTcov.shape)
        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(zTcov, dim=-2) * z_mu, dim=-1)
        print('exp_term_tmp: ', exp_term_tmp.shape)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (det_cov).unsqueeze(0), dim = 1) + eps)
        sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1) + eps)
        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt((2*np.pi)**D * det_cov)).unsqueeze(0), dim = 1) + eps)
        print('sample_energy: ', sample_energy.shape)

        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag

class train():
    
    def __init__(self):

        self.dagmm = DaGMM(2, 258)
        input = torch.randn(256, 2116)
        enc, dec, z, gamma = self.dagmm(input)
        
        phi, mu, cov = self.dagmm.compute_gmm_params(z, gamma)
        # print(phi, mu, cov)
        sample_energy, cov_diag = self.dagmm.compute_energy(z, phi, mu, cov)
        # print('energy_mean: ', sample_energy.mean())

if __name__ == '__main__':

    train()   