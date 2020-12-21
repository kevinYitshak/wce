import math
import numpy as np
# from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence

from config import settings
from models.GMM import GMM
from visualise import latent_vis
from models.CE import estimator, compression

class VGDMMs(nn.Module):

    def __init__(self, c, k_shot=None, k=4):
        super(VGDMMs, self).__init__()
        self.device = settings.device
        self.num_pro = k
        self.shot = k_shot

        self.gmm = GMM(self.device)
        self.estimator = estimator(c, k)
        self.compression = compression(c, k)

        self.mse = nn.MSELoss()

    def forward(self, support_feature, support_mask, query_feature):
        mask = F.interpolate(support_mask, support_feature.shape[-2:], mode='bilinear', align_corners=True)

        foreground_features = support_feature * mask #[b, c, 46, 46]
        background_features = support_feature * (1 - mask)

        # compression
        z_fg, x_hat_fg = self.compression.forward_fg(foreground_features)
        z_bg, x_hat_bg = self.compression.forward_bg(background_features)
        z_q, x_hat_q = self.compression.forward_q(query_feature)

        cosine_ff = self.cosine_similarity(foreground_features, x_hat_fg) # [b, 1, w, h]
        cosine_bb = self.cosine_similarity(background_features, x_hat_bg)
        cosine_qq = self.cosine_similarity(query_feature, x_hat_q)

        final_z_fg = torch.cat([z_fg, cosine_ff], dim=1) # [b, c, w, h]
        final_z_bg = torch.cat([z_bg, cosine_bb], dim=1)
        final_z_q = torch.cat([z_q, cosine_qq], dim=1)

        # estimation
        gamma_fg, gamma_bg, gamma_q = self.estimator.forward(final_z_fg, final_z_fg, final_z_q) # [b, k, w, h]
        # print('gamma_fg: ', gamma_fg.shape)
        # cal GMM params
        energy_fg, gmm_fg = self.cal_args(gamma_fg, foreground_features)
        energy_bg, gmm_bg = self.cal_args(gamma_bg, background_features)
        energy_q, gmm_q = self.cal_args(gamma_q, query_feature)

        feature_sampling_q = gmm_q.sample(torch.Size([self.num_pro])).permute(1, 0, 2) # [2, 4, 256]
        feature_sampling_fg = gmm_fg.sample(torch.Size([self.num_pro])).permute(1, 0, 2) # [2, 4, 256]
        feature_sampling_bg = gmm_bg.sample(torch.Size([self.num_pro])).permute(1, 0, 2) # [2, 4, 256]

        # Reconstruction Loss
        r_loss_fg = self.mse(foreground_features, x_hat_fg)
        r_loss_bg = self.mse(background_features, x_hat_bg)
        r_loss_q = self.mse(query_feature, x_hat_q)

        # ELBO LOSS
        kl_loss_fg_q = self.kl_loss(gmm_fg, gmm_q).mean() - gmm_q.log_prob(feature_sampling_fg.permute(1, 0, 2).to(self.device)).mean()
        kl_loss_bg_q = self.kl_loss(gmm_bg, gmm_q).mean() - gmm_q.log_prob(feature_sampling_bg.permute(1, 0, 2).to(self.device)).mean()

        mu_foreground = []
        mu_background = []
        mu_query = []
        for i in range(self.num_pro):
            mu_foreground.append(feature_sampling_fg[:, i, :].unsqueeze(dim=2).unsqueeze(dim=3))
            mu_background.append(feature_sampling_bg[:, i, :].unsqueeze(dim=2).unsqueeze(dim=3))
            mu_query.append(feature_sampling_q[:, i, :].unsqueeze(dim=2).unsqueeze(dim=3))

        total_projection_loss = energy_q + energy_fg + energy_bg
        total_kl_loss = (kl_loss_fg_q + kl_loss_bg_q) #/ self.num_pro ** 2
        total_r_loss = r_loss_fg + r_loss_bg + r_loss_q

        projWt = 1e-4
        klWt = 1e-2
        total_projection_loss = projWt * total_projection_loss
        total_kl_loss = klWt * total_kl_loss

        print('total_projection_loss: ', total_projection_loss)
        print('total_kl_loss: ', total_kl_loss)
        print('total_r_loss: ', total_r_loss)

        return mu_foreground, mu_background, mu_query, total_projection_loss + total_r_loss, total_kl_loss


    def cal_args(self, gamma, features):
        phi = self.gmm.cal_phi(gamma)
        mu = self.gmm.cal_mean(features, gamma)
        cov = self.gmm.cal_cov(features, gamma, mu)
        energy, gmm = self.gmm.cal_energy_new(features, phi, mu, cov)

        return energy, gmm

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