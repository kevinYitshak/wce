import math
import numpy as np
# from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.kl import kl_divergence

from models.CE import compression, estimator

from config import settings
from models.GMM import GMM
from models.DAGNN import DaGMM
from visualise import latent_vis

class VGPMMs(nn.Module):

    def __init__(self, c, latent_dim=129, k_shot=None, k=4):
        super(VGPMMs, self).__init__()
        self.device = settings.device
        self.num_pro = k
        self.shot = k_shot

        self.gmm = GMM(self.device)
        self.latent_dim = latent_dim
        self.compression = compression(c, self.latent_dim)
        self.estimation = estimator(self.latent_dim, k)

        self.dagmm = DaGMM(c, latent_dim=self.latent_dim, k=self.num_pro)
        
        # self.exp_ch = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, bias=True),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU()
        # )

        # self.red_ch = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, bias=True),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU()
        # )

        self.mse = nn.MSELoss()

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
        
        return phi, mu, cov

    def forward(self, support_feature, support_mask, query_feature):
        mask = F.interpolate(support_mask, support_feature.shape[-2:], mode='bilinear', align_corners=True)
        # support_feature = self.exp_ch(support_feature)
        # query_feature = self.exp_ch(query_feature)

        fg_feature = support_feature * mask #[b, c, 46, 46]
        # bg_feature = support_feature * (1 - mask)

        b, c, h, w = fg_feature.shape

        enc_fg, enc_q = self.compression.forward_enc(fg_feature, query_feature) # [b, 64, 46, 46]
        dec_fg, dec_q = self.compression.forward_dec(enc_fg, enc_q) # [b, 128, 46, 46]

        rec_cosine_fq = F.cosine_similarity(enc_fg, enc_q, dim=1) # [b, 46, 46]
        # rec_cosine_bq = F.cosine_similarity(enc_bg, enc_q, dim=1) # [b, 46, 46]

        z_fg = torch.cat([enc_fg, rec_cosine_fq.unsqueeze(1)], dim=1) # [b, 65, 46, 46]
        # z_bg = torch.cat([enc_bg, rec_cosine_bq.unsqueeze(1)], dim=1) # [b, 65, 46, 46]
        z_q = torch.cat([enc_q, rec_cosine_fq.unsqueeze(1)], dim=1) # [b, 65, 46, 46]

        gamma_fg, gamma_q = self.estimation(z_fg, z_q) # [b, k, 46, 46]

        # phi = torch.zeros(b, self.num_pro)
        # mu = torch.zeros(b, self.num_pro, self.latent_dims)
        # cov = torch.zeros(b, self.num_pro, self.latent_dims, self.latent_dims)
        for i in range(b):
            # input = torch.randn(10, 118)
            z_fg_, z_q_, gamma_fg_, gamma_q_ = z_fg[i, :, :], z_q[i, :, :], \
                                                            gamma_fg[i, :, :], gamma_q[i, :, :]

            phi_fg_, mu_fg_, cov_fg_ = \
            self.compute_gmm_params(z_fg_.view(self.latent_dim, h*w).permute(1, 0), gamma_fg_.view(self.num_pro, h*w).permute(1, 0))
            # phi_bg_, mu_bg_, cov_bg_ = \
            # self.compute_gmm_params(z_bg_.view(self.latent_dim, h*w).permute(1, 0), gamma_bg_.view(self.num_pro, h*w).permute(1, 0))
            phi_q_, mu_q_, cov_q_ = \
            self.compute_gmm_params(z_q_.view(self.latent_dim, h*w).permute(1, 0), gamma_q_.view(self.num_pro, h*w).permute(1, 0))
            # print(phi, mu, cov)
            # sample_energy_, cov_diag_ = self.dagmm.compute_energy(z, phi_, mu_, cov_)
            
            if i == 0:
                phi_fg = phi_fg_.unsqueeze(0)
                mu_fg = mu_fg_.unsqueeze(0)
                cov_fg = cov_fg_.unsqueeze(0)

                # phi_bg = phi_bg_.unsqueeze(0)
                # mu_bg = mu_bg_.unsqueeze(0)
                # cov_bg = cov_bg_.unsqueeze(0)

                phi_q = phi_q_.unsqueeze(0)
                mu_q = mu_q_.unsqueeze(0)
                cov_q = cov_q_.unsqueeze(0)
                # sample_energy = sample_energy_.unsqueeze(0)
                # cov_diag = cov_diag_.unsqueeze(0)
            else:
                phi_fg = torch.cat([phi_fg, phi_fg_.unsqueeze(0)], dim=0)
                mu_fg = torch.cat([mu_fg, mu_fg_.unsqueeze(0)], dim=0)
                cov_fg = torch.cat([cov_fg, cov_fg_.unsqueeze(0)], dim=0)
                
                # phi_bg = torch.cat([phi_bg, phi_bg_.unsqueeze(0)], dim=0)
                # mu_bg = torch.cat([mu_bg, mu_bg_.unsqueeze(0)], dim=0)
                # cov_bg = torch.cat([cov_bg, cov_bg_.unsqueeze(0)], dim=0)
                
                phi_q = torch.cat([phi_q, phi_q_.unsqueeze(0)], dim=0)
                mu_q = torch.cat([mu_q, mu_q_.unsqueeze(0)], dim=0)
                cov_q = torch.cat([cov_q, cov_q_.unsqueeze(0)], dim=0)
                # sample_energy = torch.cat([sample_energy, sample_energy_.unsqueeze(0)], dim=0)
                # cov_diag = torch.cat([cov_diag, cov_diag_.unsqueeze(0)], dim=0)
        for i in range(b):
            for j in range(self.num_pro):
                cov_f_ = cov_fg[i, j, :, :].clone()
                cov_fg[i, j, :, :] = self.get_pd(cov_f_, self.latent_dim)
                # cov_b_ = cov_bg[i, j, :, :].clone()
                # cov_bg[i, j, :, :] = self.get_pd(cov_b_, self.latent_dim)
                cov_q_ = cov_q[i, j, :, :].clone()
                cov_q[i, j, :, :] = self.get_pd(cov_q_, self.latent_dim)

        # print('phi_shape: ', phi.shape)
        # print('mu_shape: ', mu.shape)
        # print('cov_shape: ', cov.shape)

        gmm_q = self.cal_gmm(phi_q, mu_q, cov_q)
        gmm_fg = self.cal_gmm(phi_fg, mu_fg, cov_fg)
        # gmm_bg = self.cal_gmm(phi_bg, mu_bg, cov_bg)

        feature_sampling_fg = gmm_fg.sample(torch.Size([self.num_pro])).permute(1, 0, 2) # [2, 4, 65]
        # feature_sampling_bg = gmm_bg.sample(torch.Size([self.num_pro])).permute(1, 0, 2) # [2, 4, 65]
        feature_sampling_q = gmm_q.sample(torch.Size([self.num_pro])).permute(1, 0, 2) # [2, 4, 65]
        
        # Projection Loss
        enc_energy_fg = - gmm_fg.log_prob(z_fg.view(b, self.latent_dim, h*w).permute(2, 0, 1).to(self.device)).mean() 
        # enc_energy_bg = - gmm_bg.log_prob(z_bg.view(b, self.latent_dim, h*w).permute(2, 0, 1).to(self.device)).mean() 
        enc_energy_q = - gmm_q.log_prob(z_q.view(b, self.latent_dim, h*w).permute(2, 0, 1).to(self.device)).mean() 

        # Reconstruction Loss
        mse = self.mse(fg_feature, dec_fg) + self.mse(query_feature, dec_q) #+ self.mse(dec_fg, dec_q)

        # ELBO LOSS
        # kl_loss_fg_q = self.kl_loss(gmm_q, gmm_fg).mean() - gmm_fg.log_prob(feature_sampling_q.permute(1, 0, 2).to(self.device)).mean()
        kl_loss_fg_q = self.kl_loss(gmm_q, gmm_fg).mean() - gmm_fg.log_prob(feature_sampling_fg.permute(1, 0, 2).to(self.device)).mean()
        # kl_loss_bg_q = self.kl_loss(gmm_q, gmm_bg).mean() - gmm_bg.log_prob(feature_sampling_q.permute(1, 0, 2).to(self.device)).mean()

        total_projection_loss = enc_energy_fg + enc_energy_q
        total_kl_loss = kl_loss_fg_q #+ kl_loss_bg_q #/ self.num_pro ** 2

        projWt = 1e-4
        klWt = 1e-2
        total_projection_loss = (projWt * total_projection_loss)
        total_kl_loss = (klWt * total_kl_loss) + mse

        mu_foreground = []
        # mu_background = []
        mu_query = []
        for i in range(self.num_pro):
            mu_foreground.append(feature_sampling_fg[:, i, :].unsqueeze(dim=2).unsqueeze(dim=3))
            # mu_background.append(feature_sampling_bg[:, i, :].unsqueeze(dim=2).unsqueeze(dim=3))
            mu_query.append(feature_sampling_q[:, i, :].unsqueeze(dim=2).unsqueeze(dim=3))

        # print('total_projection_loss: ', total_projection_loss)
        # print('total_kl_loss: ', total_kl_loss)
        return mu_foreground, mu_query, total_projection_loss, total_kl_loss
    
    def get_pd(self, x, size):
        d = x.diag()
        d = torch.where(d > 1e-6, d, 1e-6*torch.ones(d.size()).to(self.device))
        d[torch.isnan(d)] = 1e-6 
        x = x + torch.eye(size).to(self.device) * d 
        # print(x.min())
        return x

    def cal_gmm(self, phi, mu, cov):
        mix = Categorical(phi.to(self.device))
        # print(mix)
        comp = Independent(MultivariateNormal(mu.to(self.device), cov.to(self.device)), reinterpreted_batch_ndims=0)
        # print('COMP: ', comp.batch_shape, comp.event_shape) # torch.Size([b, k]) torch.Size([c])
        gmm = MixtureSameFamily(mix, comp)
        return gmm

    def cal_args(self, gamma, features):
        phi = self.gmm.cal_phi(gamma)
        mu = self.gmm.cal_mean(features, gamma)
        cov = self.gmm.cal_cov(features, gamma, mu)
        energy, gmm = self.gmm.cal_energy_new(features, phi, mu, cov)

        return phi, mu, cov, energy, gmm

    def kl_loss(self, gmm_p, gmm_q):
        return kl_divergence(gmm_p.component_distribution, gmm_q.component_distribution)

    # def kl_loss(self, support_fd, query_fd): 
    #     return kl_divergence(support_fd, query_fd)


    def feature_sampling(self, _phi, _mu, _cov):
        '''
        Sampling of the features from Multivariate Normal distribution
        feature space for both foreground and background
        '''
        fd = []
        for i in range(_mu.size(0)):
            fs_ = []
            fd_ = []
            for j in range(self.num_pro):
                mu_f = _mu[i, j, :].clone() # [c]
                cov_f = _cov[i, j, :, :].clone() #[c, c]
                feature_sampling_mixture = MultivariateNormal(mu_f.to(self.device), cov_f.to(self.device))
                fs_.append(feature_sampling_mixture.sample().unsqueeze(0).unsqueeze(1) * _phi[i, j])
                fd_.append(feature_sampling_mixture)
                # print(len(fd_), fd_)
            if i == 0:
                fs = torch.cat(fs_, dim=1)
                fd.append(fd_)
                # print('---: ', fd_)
            else:
                fs_batch = torch.cat(fs_, dim=1)
                fs = torch.cat([fs, fs_batch], dim=0)
                fd.append(fd_)
        return fs, fd

    def forward_kshot(self, support_feature_all, support_mask_all, query_feature, k_shot):
        b, c, h, w = query_feature.size()        

        for i in range(k_shot):
            support_1feature = support_feature_all[:, i*c:(i+1)*c, :, :]
            support_1mask = support_mask_all[:, i:i+1, :, :]

            foreground_features = support_1feature * support_1mask
            background_features = support_1feature * (1 - support_1mask)

            # calculating gmm params foreground
            cosine_fq_ = self.cosine_similarity(foreground_features, query_feature)
            gamma_fg = self.estimator_fg(torch.cat((foreground_features, cosine_fq_), dim=1))
            phi_fg, mu_fg, cov_fg, energy_fg, gmm_fg = self.cal_args(gamma_fg, foreground_features)

            # calculating gmm params background
            cosine_bq_ = self.cosine_similarity(background_features, query_feature)
            gamma_bg = self.estimator_bg(torch.cat((background_features, cosine_bq_), dim=1))
            phi_bg, mu_bg, cov_bg, energy_bg, gmm_bg = self.cal_args(gamma_bg, background_features)

            # feature sampling
            feature_sampling_fg_ = gmm_fg.sample(torch.Size([self.num_pro])).permute(1, 0, 2) # [2, 4, 256]
            feature_sampling_bg_ = gmm_bg.sample(torch.Size([self.num_pro])).permute(1, 0, 2) # [2, 4, 256]

            if i == 0:
                feature_sampling_fg = feature_sampling_fg_
                feature_sampling_bg = feature_sampling_bg_
                cosine_fq = cosine_fq_
                cosine_bq = cosine_bq_
            else:
                feature_sampling_fg = torch.cat([feature_sampling_fg, feature_sampling_fg_], dim=1) #[2, k*shot, c]
                feature_sampling_bg = torch.cat([feature_sampling_bg, feature_sampling_bg_], dim=1) #[2, k*shot, c]
                cosine_fq = torch.cat([cosine_fq, cosine_fq_], dim=1)
                cosine_bq = torch.cat([cosine_bq, cosine_bq_], dim=1)
        
        with torch.no_grad():
            cosine_fq = self.k_shot_process(cosine_fq)
            cosine_bq = self.k_shot_process(cosine_bq)
        
        gamma_q = self.estimator_query(torch.cat([query_feature, cosine_fq, cosine_bq], dim=1))
        phi_q, mu_q, cov_q, energy_q, gmm_q = self.cal_args(gamma_q, query_feature)
        feature_sampling_q = gmm_q.sample(torch.Size([self.num_pro])).permute(1, 0, 2) # [2, 4, 256]

        # print('feature_samp_fg: ', feature_sampling_fg.shape)
        # print('feature_samp_q: ', feature_sampling_q.shape)
        mu_foreground = []
        mu_background = []
        mu_query = []
        for i in range(self.num_pro*k_shot):
            mu_foreground.append(feature_sampling_fg[:, i, :].unsqueeze(dim=2).unsqueeze(dim=3)) #[2, 256, 1, 1]
            mu_background.append(feature_sampling_bg[:, i, :].unsqueeze(dim=2).unsqueeze(dim=3))
        for i in range(self.num_pro):
            mu_query.append(feature_sampling_q[:, i, :].unsqueeze(dim=2).unsqueeze(dim=3))

        return mu_foreground, mu_background, mu_query, 0, 0 #total_projection_loss, total_kl_loss

    def k_shot_process(self, feature):
        b, c, h, w = feature.shape
        feature = feature.view(b, c, h*w).permute(0, 2, 1)
        avg = nn.AdaptiveAvgPool1d(1)
        feature = avg(feature).permute(0, 2, 1).view(b, 1, h, w)
        return feature

    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    def cosine_similarity(self, feature, query_feature):
        return F.cosine_similarity(feature, query_feature).unsqueeze(1)

    def Vis(self, f1, f2, name):
        visualise = latent_vis.visualise(f1, f2)
        embeddings, mu = visualise._tsne()
        visualise._plot(embeddings, name)
        visualise._plot(mu, name)