import math
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ASPP
from config import settings


# from visualise import latent_vis

class DAPMMs(nn.Module):

    def __init__(self, c, k_shot=None, k=4):
        super(DAPMMs, self).__init__()
        self.device = settings.device
        self.num_pro = k
        self.shot = k_shot

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=256 + 256, out_channels=256, kernel_size=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.fusion_1 = ASPP.PSPnet()
        self.fusion_2 = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.estimator = nn.Sequential(
            nn.Conv2d(in_channels=256 + 1, out_channels=128, kernel_size=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=self.num_pro, kernel_size=1, bias=True),
            # nn.BatchNorm2d(),
            # nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self, support_feature, support_mask, query_feature):
        mask = F.interpolate(support_mask, support_feature.shape[-2:], mode='bilinear', align_corners=True)

        foreground_features = support_feature * mask
        background_features = support_feature * (1 - mask)

        rel_cosine_fg = self.cosine_similarity(foreground_features, query_feature)
        rel_cosine_bg = self.cosine_similarity(background_features, query_feature)

        # eucledian_dist_fg = self.relative_euclidean_distance(foreground_features, query_feature)
        # eucledian_dist_bg = self.relative_euclidean_distance(background_features, query_feature)

        support_query_fusion_fg = self.fusion(torch.cat((foreground_features, query_feature), dim=1))
        support_query_fusion_bg = self.fusion(torch.cat((background_features, query_feature), dim=1))
        support_query_fusion_fg = self.fusion_1(support_query_fusion_fg)
        support_query_fusion_bg = self.fusion_1(support_query_fusion_bg)
        # print(support_query_fusion_fg.size())
        # print(support_query_fusion_bg.size())
        support_query_fusion_fg = self.fusion_2(support_query_fusion_fg)
        support_query_fusion_bg = self.fusion_2(support_query_fusion_bg)
        # print(support_query_fusion_fg.size())
        # print(support_query_fusion_bg.size())

        z_foreground = torch.cat((support_query_fusion_fg, rel_cosine_fg.unsqueeze(1)), dim=1)
        z_background = torch.cat((support_query_fusion_bg, rel_cosine_bg.unsqueeze(1)), dim=1)

        gamma_foreground = self.estimator(z_foreground)
        gamma_background = self.estimator(z_background)

        with torch.no_grad():
            mu_fg = self.compute_gmm_params(gamma_foreground, z_foreground)
            mu_bg = self.compute_gmm_params(gamma_background, z_background)
        
        mu_ = []
        for i in range(self.num_pro):
            mu_.append(mu_fg[:, i, :].unsqueeze(dim=2).unsqueeze(dim=3))

        Prob_map, P = self.discriminative_model(z_foreground, mu_fg, mu_bg)

        return mu_, Prob_map

    def forward_kshot(self, support_feature_all, support_mask_all, query_feature, k_shot):
        b, c, h, w = query_feature.size()

        for i in range(k_shot):
            support_1feature = support_feature_all[:, :, i*h:(i+1)*h, :]
            support_1mask = support_mask_all[:, :, i*h:(i+1)*h, :]

            foreground_features = support_1feature * support_1mask
            background_features = support_1feature * (1 - support_1mask)

            foreground_features = self.fusion(torch.cat((foreground_features, query_feature), dim=1))
            background_features = self.fusion(torch.cat((background_features, query_feature), dim=1))
            foreground_features = self.fusion_1(foreground_features)
            background_features = self.fusion_1(background_features)
            # print(support_query_fusion_fg.size())
            # print(support_query_fusion_bg.size())
            foreground_features = self.fusion_2(foreground_features)
            background_features = self.fusion_2(background_features)

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

        z_foreground = torch.cat((z_foreground, cosine_fg), dim=1)
        z_background = torch.cat((z_background, cosine_bg), dim=1)
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

    def relative_euclidean_distance(self, feature, query_feature):
        return (feature - query_feature).norm(2, dim=1) #/ (feature.norm(2, dim=1) + 1e-6)

    def compute_gmm_params(self, gamma, z):
        b, c, h, w = z.size()

        z = z.view(b, c, h * w)
        # print('z_size: ', z.size())
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