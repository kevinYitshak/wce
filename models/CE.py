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

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, .025)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def f_est_fg(in_planes, k):
    "3x3 convolution"
    return nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=32, kernel_size=3, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=k, kernel_size=3, bias=True),
            nn.BatchNorm2d(k),
            nn.Softmax(dim=1)
        )

# def f_est_bg(in_planes, k):
#     "3x3 convolution"
#     return nn.Sequential(
#             nn.Conv2d(in_channels=in_planes, out_channels=32, kernel_size=3, bias=True),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=32, out_channels=k, kernel_size=3, bias=True),
#             nn.BatchNorm2d(k),
#             # nn.ReLU(),
#             nn.Softmax(dim=1)
#         )

def f_est_q(in_planes, k):
    "3x3 convolution"
    return nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=32, kernel_size=3, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=k, kernel_size=3, bias=True),
            nn.BatchNorm2d(k),
            nn.Softmax(dim=1)
        )

class estimator(nn.Module):

    def __init__(self, latent_dim, k=4):
        super(estimator, self).__init__()
        self.device = settings.device

        self.estimator_fg = f_est_fg(latent_dim, k)
        # self.estimator_bg = f_est_bg(latent_dim, k)
        self.estimator_q = f_est_q(latent_dim, k)
        
        self.estimator_fg.apply(init_weights)
        # self.estimator_bg.apply(init_weights)
        self.estimator_q.apply(init_weights)

    def forward(self, fg, q):
        return self.estimator_fg(fg), self.estimator_q(q)


def f_ext_fg(in_planes, latent_dim):
    "3x3 convolution"
    return nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=128, kernel_size=3, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=latent_dim-1, kernel_size=3, bias=True),
            nn.BatchNorm2d(latent_dim-1),
            nn.ReLU(),
        )

def f_dec_fg(in_planes, latent_dim):
    "3x3 convolution"
    return nn.Sequential(
            nn.Conv2d(in_channels=latent_dim-1, out_channels=64, kernel_size=3, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=in_planes, kernel_size=3, bias=True),
            nn.BatchNorm2d(in_planes),
            nn.ReLU()
        )

# def f_ext_bg(in_planes, latent_dim):
#     "3x3 convolution"
#     return nn.Sequential(
#             nn.Conv2d(in_channels=in_planes, out_channels=128, kernel_size=3, bias=True),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=128, out_channels=latent_dim-1, kernel_size=3, bias=True),
#             nn.BatchNorm2d(latent_dim-1),
#             nn.ReLU()
#         )

# def f_dec_bg(in_planes, latent_dim):
#     "3x3 convolution"
#     return nn.Sequential(
#             nn.Conv2d(in_channels=latent_dim-1, out_channels=128, kernel_size=3, bias=True),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=128, out_channels=in_planes, kernel_size=3, bias=True),
#             nn.BatchNorm2d(in_planes),
#             nn.ReLU()
#         )

def f_ext_q(in_planes, latent_dim):
    "3x3 convolution"
    return nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=128, kernel_size=3, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=latent_dim-1, kernel_size=3, bias=True),
            nn.BatchNorm2d(latent_dim-1),
            nn.ReLU()
        )

def f_dec_q(in_planes, latent_dim):
    "3x3 convolution"
    return nn.Sequential(
            nn.Conv2d(in_channels=latent_dim-1, out_channels=64, kernel_size=3, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=in_planes, kernel_size=3, bias=True),
            nn.BatchNorm2d(in_planes),
            nn.ReLU()
        )
class compression(nn.Module):

    def __init__(self, c, latent_dim):
        super(compression, self).__init__()

        self.encoder_fg = f_ext_fg(c, latent_dim)
        # self.encoder_bg = f_ext_bg(c, latent_dim)
        self.encoder_q = f_ext_q(c, latent_dim)

        self.encoder_fg.apply(init_weights)
        # self.encoder_bg.apply(init_weights)
        self.encoder_q.apply(init_weights)
        
        self.decoder_fg = f_dec_fg(c, latent_dim)
        # self.decoder_bg = f_dec_bg(c, latent_dim)
        self.decoder_q = f_dec_q(c, latent_dim)

        self.decoder_fg.apply(init_weights)
        # self.decoder_bg.apply(init_weights)
        self.decoder_q.apply(init_weights)

    def forward_enc(self, fg, q):
        return self.encoder_fg(fg), self.encoder_q(q)
    
    def forward_dec(self, fg, q):
        return self.decoder_fg(fg), self.decoder_q(q)