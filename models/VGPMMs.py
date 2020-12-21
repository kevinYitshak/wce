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

class VGPMMs(nn.Module):

    def __init__(self, c, k_shot=None, k=4):
        super(VGPMMs, self).__init__()
        self.device = settings.device
        self.num_pro = k
        self.shot = k_shot

        self.gmm = GMM(self.device)

        self.estimator_fg = nn.Sequential(
            nn.Conv2d(in_channels=256 + 1, out_channels=64, kernel_size=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=self.num_pro, kernel_size=1, bias=True),
            nn.Softmax(dim=1)
        )
        
        self.estimator_bg = nn.Sequential(
            nn.Conv2d(in_channels=256 + 1, out_channels=64, kernel_size=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=self.num_pro, kernel_size=1, bias=True),
            nn.Softmax(dim=1)
        )

        self.estimator_query = nn.Sequential(
            nn.Conv2d(in_channels=256 + 2, out_channels=64, kernel_size=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=self.num_pro, kernel_size=1, bias=True),
            nn.Softmax(dim=1)
        )

        self.estimator_fg.apply(self.init_estimator_weights)
        self.estimator_bg.apply(self.init_estimator_weights)
        self.estimator_query.apply(self.init_estimator_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 1/self.num_pro)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, support_feature, support_mask, query_feature):
        mask = F.interpolate(support_mask, support_feature.shape[-2:], mode='bilinear', align_corners=True)

        foreground_features = support_feature * mask #[b, c, 46, 46]
        background_features = support_feature * (1 - mask)

        # calculating gmm params foreground
        cosine_fq = self.cosine_similarity(foreground_features, query_feature)
        gamma_fg = self.estimator_fg(torch.cat((foreground_features, cosine_fq), dim=1))
        phi_fg, mu_fg, cov_fg, energy_fg, gmm_fg = self.cal_args(gamma_fg, foreground_features)

        # calculating gmm params background
        cosine_bq = self.cosine_similarity(background_features, query_feature)
        gamma_bg = self.estimator_bg(torch.cat((background_features,cosine_bq), dim=1))
        phi_bg, mu_bg, cov_bg, energy_bg, gmm_bg = self.cal_args(gamma_bg, background_features)

         #query
        gamma_q = self.estimator_query(torch.cat((query_feature, cosine_fq, cosine_bq), dim=1))
        phi_q, mu_q, cov_q, energy_q, gmm_q = self.cal_args(gamma_q, query_feature)
        
        #posterior and distribution calculation
        # feature_sampling_q, feature_dist_q = self.feature_sampling(phi_q, mu_q, cov_q) # [b, k, c]
        # feature_sampling_fg, feature_dist_fg = self.feature_sampling(phi_fg, mu_fg, cov_fg)
        # feature_sampling_bg, feature_dist_bg = self.feature_sampling(phi_bg, mu_bg, cov_bg)
        

        feature_sampling_q = gmm_q.sample(torch.Size([self.num_pro])).permute(1, 0, 2) # [2, 4, 256]
        feature_sampling_fg = gmm_fg.sample(torch.Size([self.num_pro])).permute(1, 0, 2) # [2, 4, 256]
        feature_sampling_bg = gmm_bg.sample(torch.Size([self.num_pro])).permute(1, 0, 2) # [2, 4, 256]
        
        # self.Vis(foreground_features, mu_fg, 'embeddings_Chylous_gmm_mu_fg_lr_50.png')
        # self.Vis(background_features, feature_sampling_bg, 'embeddings_Chylous_gmm_bg.png')
        # self.Vis(gamma_q, feature_sampling_q, 'embeddings_Chylous_gamma_q_lr_200.png')

        # print('feature_sampling_fg: ', feature_sampling_fg.shape)
        # print('feature_sampling_bg: ', feature_sampling_bg.shape)
        
        # Reconstruction Loss

        # ELBO LOSS
        kl_loss_fg_q = self.kl_loss(gmm_fg, gmm_q).mean() - gmm_q.log_prob(feature_sampling_fg.permute(1, 0, 2).to(self.device)).mean()
        kl_loss_bg_q = self.kl_loss(gmm_bg, gmm_q).mean() - gmm_q.log_prob(feature_sampling_bg.permute(1, 0, 2).to(self.device)).mean()

        # kl_loss_fg_q = 0
        # kl_loss_bg_q = 0
        # for i in range(mu_fg.size(0)):
        #     for j in range(mu_fg.size(1)):
        #         kl_loss_fg_q = kl_loss_fg_q + self.kl_loss(feature_dist_fg[i][j], feature_dist_q[i][j])
        #         kl_loss_bg_q = kl_loss_bg_q + self.kl_loss(feature_dist_bg[i][j], feature_dist_q[i][j])
        # print('kl_fg: ', kl_loss_fg_q)
        # print('kl_bg: ', kl_loss_bg_q)

        mu_foreground = []
        mu_background = []
        mu_query = []
        for i in range(self.num_pro):
            mu_foreground.append(feature_sampling_fg[:, i, :].unsqueeze(dim=2).unsqueeze(dim=3))
            mu_background.append(feature_sampling_bg[:, i, :].unsqueeze(dim=2).unsqueeze(dim=3))
            mu_query.append(feature_sampling_q[:, i, :].unsqueeze(dim=2).unsqueeze(dim=3))

        total_projection_loss = energy_q + energy_fg + energy_bg
        total_kl_loss = (kl_loss_fg_q + kl_loss_bg_q) #/ self.num_pro ** 2

        projWt = 1e-4
        klWt = 1e-2
        total_projection_loss = projWt * total_projection_loss
        total_kl_loss = klWt * total_kl_loss

        print('total_projection_loss: ', total_projection_loss)
        print('total_kl_loss: ', total_kl_loss)
        return mu_foreground, mu_background, mu_query, total_projection_loss, total_kl_loss


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