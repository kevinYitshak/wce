import math
import numpy as np
# from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

from config import settings
from models.GMM import GMM
# from visualise import latent_vis

class DAPMMs(nn.Module):

    def __init__(self, c, k_shot=None, k=4):
        super(DAPMMs, self).__init__()
        self.device = settings.device
        self.num_pro = k
        self.shot = k_shot

        self.gmm = GMM(self.device)

        self.estimator = nn.Sequential(
            nn.Conv2d(in_channels=64*2 + 1, out_channels=64, kernel_size=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=self.num_pro, kernel_size=1, bias=True),
            nn.Softmax(dim=1)
        )

        self.estimator.apply(self.init_estimator_weights)

    def init_estimator_weights(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, support_feature, support_mask, query_feature):
        mask = F.interpolate(support_mask, support_feature.shape[-2:], mode='bilinear', align_corners=True)

        foreground_features = support_feature * mask
        background_features = support_feature * (1 - mask)

        rel_cosine_fg = self.cosine_similarity(foreground_features, query_feature)
        rel_cosine_bg = self.cosine_similarity(background_features, query_feature)

        z_foreground = torch.cat((foreground_features, query_feature, rel_cosine_fg.unsqueeze(1)), dim=1)
        z_background = torch.cat((background_features, query_feature, rel_cosine_bg.unsqueeze(1)), dim=1)

        gamma_foreground = self.estimator(z_foreground)
        gamma_background = self.estimator(z_background)

        # with torch.no_grad():
        # calculating gmm params foreground
        phi_fg = self.gmm.cal_phi(gamma_foreground)
        mu_fg = self.gmm.cal_mean(z_foreground, gamma_foreground)
        cov_fg = self.gmm.cal_cov(z_foreground, gamma_foreground, mu_fg)
        energy_fg = self.gmm.cal_energy(z_foreground, phi_fg, mu_fg, cov_fg)
        # print('phi_fg: ', phi_fg.shape)
        # print('mu_fg: ', mu_fg.shape)
        # print('cov_fg: ', cov_fg.shape)
        # print(energy_fg)
        # calculating gmm params bcakground
        phi_bg = self.gmm.cal_phi(gamma_background)
        mu_bg = self.gmm.cal_mean(z_background, gamma_background)
        cov_bg = self.gmm.cal_cov(z_background, gamma_background, mu_bg)
        energy_bg = self.gmm.cal_energy(z_background, phi_bg, mu_bg, cov_bg)
        # print('phi_bg: ', phi_bg.shape)
        # print('mu_bg: ', mu_bg.shape)
        # print('cov_bg: ', cov_bg.shape)
        # print(energy_bg)
        # z_fg, phi_fg, mu_fg, cov_fg = self.compute_gmm(gamma_foreground, z_foreground)
        # z_bg, phi_bg, mu_bg, cov_bg = self.compute_gmm(gamma_background, z_background)

        feature_sampling_fg = self.feature_sampling(phi_fg, mu_fg, cov_fg)
        # print('feature_sampling_fg: ', feature_sampling_fg.shape)
        feature_sampling_bg = self.feature_sampling(phi_bg, mu_bg, cov_bg)
        # print('feature_sampling_bg: ', feature_sampling_bg.shape)

        mu_ = []
        for i in range(self.num_pro):
            mu_.append(feature_sampling_fg[:, i, :].unsqueeze(dim=2).unsqueeze(dim=3))

        # print('fs_size: ', fs.shape)
        Prob_map, P = self.discriminative_model(z_foreground, feature_sampling_fg, feature_sampling_bg)

        return mu_, Prob_map, energy_fg, energy_bg

    def feature_sampling(self, _phi, _mu, _cov):
        '''
        Sampling of the features from Multivariate Normal distribution
        feature space for both foreground and background
        '''
        for i in range(_mu.size(0)):
            fs_ = []
            for j in range(self.num_pro):
                mu_f = _mu[i, j, :].clone() # [c]
                cov_f = _cov[i, j, :, :].clone() #[c, c]
                feature_sampling_mixture = MultivariateNormal(mu_f.to(self.device), cov_f.to(self.device))
                fs_.append(feature_sampling_mixture.sample().unsqueeze(0).unsqueeze(1) * _phi[i, j])
            if i == 0:
                fs = torch.cat(fs_, dim=1)
            else:
                fs_batch = torch.cat(fs_, dim=1)
                fs = torch.cat([fs, fs_batch], dim=0)
        return fs

    def forward_kshot(self, support_feature_all, support_mask_all, query_feature, k_shot):
        b, c, h, w = query_feature.size()

        for i in range(k_shot):
            support_1feature = support_feature_all[:, :, i*h:(i+1)*h, :]
            support_1mask = support_mask_all[:, :, i*h:(i+1)*h, :]

            foreground_features = support_1feature * support_1mask
            background_features = support_1feature * (1 - support_1mask)

            rel_cosine_fg = self.cosine_similarity(foreground_features, query_feature).unsqueeze(1) # b x h x c
            rel_cosine_bg = self.cosine_similarity(background_features, query_feature).unsqueeze(1)

            if i == 0:
                cosine_fg = rel_cosine_fg
                cosine_bg = rel_cosine_bg
                z_foreground = foreground_features
                z_background = background_features
            else:
                cosine_fg = torch.cat([cosine_fg, rel_cosine_fg], dim=1)
                cosine_bg = torch.cat([cosine_bg, rel_cosine_bg], dim=1)
                z_foreground = torch.cat([z_foreground, foreground_features], dim=1)
                z_background = torch.cat([z_background, background_features], dim=1)

        # print('cosine_fg: ', cosine_fg.size())
        # print('cosine_bg: ', cosine_bg.size())
        # print('z_foreground: ', z_foreground.size())
        # print('z_background: ', z_background.size())

        with torch.no_grad():
            b, c, h, w = z_foreground.size()
            _, c_, _, _ = cosine_fg.size()
            z_foreground = z_foreground.view(b, c, h*w).permute(0, 2, 1)
            z_background = z_background.view(b, c, h*w).permute(0, 2, 1)

            cosine_fg = cosine_fg.view(b, c_, h*w).permute(0, 2, 1)
            cosine_bg = cosine_bg.view(b, c_, h*w).permute(0, 2, 1)
            # print(z_foreground.size())
            avg_pool_support = nn.AdaptiveAvgPool1d(256)
            avg_pool_cosine = nn.AdaptiveAvgPool1d(1)
            bn_s = nn.BatchNorm1d(256)
            bn_c = nn.BatchNorm1d(1)
            act_relu = nn.ReLU()

            z_foreground = act_relu(bn_s(avg_pool_support(z_foreground).permute(0, 2, 1)))
            z_background = act_relu(bn_s(avg_pool_support(z_background).permute(0, 2, 1)))

            cosine_fg = act_relu(bn_c(avg_pool_cosine(cosine_fg).permute(0, 2, 1)))
            cosine_bg = act_relu(bn_c(avg_pool_cosine(cosine_bg).permute(0, 2, 1)))
            # print(z_foreground.size())
            z_foreground = z_foreground.view(b, 256, h, w)
            z_background = z_background.view(b, 256, h, w)

            cosine_fg = cosine_fg.view(b, 1, h, w)
            cosine_bg = cosine_bg.view(b, 1, h, w)

        z_foreground = torch.cat((z_foreground, query_feature, cosine_fg), dim=1)
        z_background = torch.cat((z_background, query_feature, cosine_bg), dim=1)
        # print(z_foreground.size())
        gamma_foreground = self.estimator(z_foreground)
        # print(gamma_foreground)
        # print('gamma_foreground: ', gamma_foreground.size())
        gamma_background = self.estimator(z_background)
        # print('gamma_background: ', gamma_background.size())

        mu_fg = self.compute_gmm_params(gamma_foreground, z_foreground)
        # print('mu_fg_size: ', mu_fg.size())
        # print(mu_fg)
        mu_bg = self.compute_gmm_params(gamma_background, z_background)
        # print('mu_bg_size: ', mu_bg.size())

        # visualise = latent_vis.visualise(torch.cat((query_feature, foreground_features, rel_cosine_fg.unsqueeze(1)), dim=1), mu_fg.permute(0, 2, 1))
        # embeddings, mu = visualise._tsne()
        # visualise._plot(embeddings, 'embeddings_Chylous.png')
        # visualise._plot(mu, 'embeddings_Chylous.png')
        
        mu_ = []
        for i in range(self.num_pro):
            mu_.append(mu_fg[:, i, :].unsqueeze(dim=2).unsqueeze(dim=3))

        Prob_map, P = self.discriminative_model(z_foreground, mu_fg, mu_bg)

        return mu_, Prob_map

    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    def cosine_similarity(self, feature, query_feature):
        return F.cosine_similarity(feature, query_feature)

    def compute_gmm_params(self, gamma, z):
        b, c, h, w = z.size()

        z = z.view(b, c, h * w) # [b, c, h*w]

        gamma = gamma.view(b, self.num_pro, h * w).permute(0, 2, 1)
        # print('gamma_size: ', gamma.size())

        gamma_ = gamma / (1e-6 + gamma.sum(dim=2, keepdim=True))

        mu = torch.bmm(z, gamma_)
        mu = self._l2norm(mu, dim=1)
        mu = mu.permute(0, 2, 1)
        # print('mu_size: ', mu.shape)

        return mu

    def discriminative_model(self, query_feature, mu_f, mu_b):
        mu = torch.cat([mu_f, mu_b], dim=1)
        mu = mu.permute(0, 2, 1)
        # print('mu_dis_model_size: ', mu.size())

        b, c, h, w = query_feature.size()
        x = query_feature.view(b, c, h * w)  # b * c * n
        # print('x_dis_model_size: ', x.size())
        with torch.no_grad():
            x_t = x.permute(0, 2, 1)
            # print('x_t_dis_model_size: ', x_t.size())  # b * n * c
            z = torch.bmm(x_t, mu)  # b * n * k

            z = F.softmax(z, dim=2)  # b * n * k

        P = z.permute(0, 2, 1)

        P = P.view(b, self.num_pro * 2, h, w)  # b * k * w * h  probability map
        P_f = torch.sum(P[:, 0:self.num_pro], dim=1).unsqueeze(dim=1)  # foreground
        P_b = torch.sum(P[:, self.num_pro:], dim=1).unsqueeze(dim=1)  # background

        Prob_map = torch.cat([P_b, P_f], dim=1)

        return Prob_map, P