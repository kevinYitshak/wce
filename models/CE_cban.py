import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.CBAM import CBAM


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, .025)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)

class ConvCBAM(nn.Module):

    def __init__(self, in_channels, out_channels, reduction_ratio=16, convT=False, cbam=True):
        super(ConvCBAM, self).__init__()
        self.cbam = cbam
        if not convT:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=True)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        if cbam:
            self.cbam = CBAM(out_channels, reduction_ratio)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.cbam:
            x = self.cbam(x)
        x = self.relu(x)
        return x

class est_fg(nn.Module):

    def __init__(self, latent_dim, k):
        super(est_fg, self).__init__()
        latent_dim = latent_dim-2
        self.conv_est_fg_1 = ConvCBAM(latent_dim+2, latent_dim//2, cbam=False)
        self.conv_est_fg_2 = ConvCBAM(latent_dim//2, latent_dim//4, cbam=False)

        self.conv_est_fg_3 = ConvCBAM(latent_dim//4, latent_dim//6, convT=True, cbam=False)
        self.conv_est_fg_4 = ConvCBAM(latent_dim//6, k, convT=True, cbam=False)
        
        self.bn_fg = nn.BatchNorm2d(k)
        self.softmax_fg = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv_est_fg_1(x)
        x = self.conv_est_fg_2(x)
        x = self.conv_est_fg_3(x)
        x = self.conv_est_fg_4(x)
        return self.softmax_fg(self.bn_fg(x))

class est_q(nn.Module):

    def __init__(self, latent_dim, k):
        super(est_q, self).__init__()
        latent_dim = latent_dim-2
        self.conv_est_q_1 = ConvCBAM(latent_dim+2, latent_dim//2,  cbam=False)
        self.conv_est_q_2 = ConvCBAM(latent_dim//2, latent_dim//4,  cbam=False)

        self.conv_est_q_3 = ConvCBAM(latent_dim//4, latent_dim//6, convT=True, cbam=False)
        self.conv_est_q_4 = ConvCBAM(latent_dim//6, k, convT=True, cbam=False)

        self.bn_q = nn.BatchNorm2d(k)
        self.softmax_q = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_est_q_1(x)
        x = self.conv_est_q_2(x)
        x = self.conv_est_q_3(x)
        x = self.conv_est_q_4(x)
        return self.softmax_q(self.bn_q(x))

class estimator(nn.Module):

    def __init__(self, latent_dim, k=4):
        super(estimator, self).__init__()

        self.estimator_fg = est_fg(latent_dim, k)
        self.estimator_q = est_q(latent_dim, k)

        self.estimator_fg.apply(init_weights)
        self.estimator_q.apply(init_weights)

    def forward(self, fg, q):
        return self.estimator_fg(fg), self.estimator_q(q)

class enc_fg(nn.Module):

    def __init__(self, in_channels, latent_dim):
        super(enc_fg, self).__init__()

        self.conv_enc_fg_1 = ConvCBAM(in_channels, in_channels//2)
        self.conv_enc_fg_2 = ConvCBAM(in_channels//2, in_channels//4)

        self.conv_enc_fg_3 = ConvCBAM(in_channels//4, in_channels//2, convT=True)
        self.conv_enc_fg_4 = ConvCBAM(in_channels//2, latent_dim-2, convT=True)

    def forward(self, x):
        x = self.conv_enc_fg_1(x)
        x = self.conv_enc_fg_2(x)
        x = self.conv_enc_fg_3(x)
        x = self.conv_enc_fg_4(x)
        return x

class dec_fg(nn.Module):

    def __init__(self, in_channels, latent_dim):
        super(dec_fg, self).__init__()
        latent_dim = latent_dim - 2

        self.conv_dec_fg_1 = ConvCBAM(latent_dim, latent_dim)
        self.conv_dec_fg_2 = ConvCBAM(latent_dim, latent_dim//2)

        self.conv_dec_fg_3 = ConvCBAM(latent_dim//2, latent_dim, convT=True)
        self.conv_dec_fg_4 = ConvCBAM(latent_dim, in_channels, convT=True)

    def forward(self, x):
        x = self.conv_dec_fg_1(x)
        x = self.conv_dec_fg_2(x)
        x = self.conv_dec_fg_3(x)
        x = self.conv_dec_fg_4(x)
        return x

class enc_q(nn.Module):

    def __init__(self, in_channels, latent_dim):
        super(enc_q, self).__init__()

        self.conv_enc_q_1 = ConvCBAM(in_channels, in_channels//2)
        self.conv_enc_q_2 = ConvCBAM(in_channels//2, in_channels//4)

        self.conv_enc_q_3 = ConvCBAM(in_channels//4, in_channels//2, convT=True)
        self.conv_enc_q_4 = ConvCBAM(in_channels//2, latent_dim-2, convT=True)

    def forward(self, x):
        x = self.conv_enc_q_1(x)
        x = self.conv_enc_q_2(x)
        x = self.conv_enc_q_3(x)
        x = self.conv_enc_q_4(x)
        return x

class dec_q(nn.Module):

    def __init__(self, in_channels, latent_dim):
        super(dec_q, self).__init__()
        latent_dim = latent_dim-2

        self.conv_dec_q_1 = ConvCBAM(latent_dim, latent_dim)
        self.conv_dec_q_2 = ConvCBAM(latent_dim, latent_dim//2)

        self.conv_dec_q_3 = ConvCBAM(latent_dim//2, latent_dim, convT=True)
        self.conv_dec_q_4 = ConvCBAM(latent_dim, in_channels, convT=True)

    def forward(self, x):
        x = self.conv_dec_q_1(x)
        x = self.conv_dec_q_2(x)
        x = self.conv_dec_q_3(x)
        x = self.conv_dec_q_4(x)
        return x

class compression(nn.Module):

    def __init__(self, in_channels, latent_dim):
        super(compression, self).__init__()

        self.encoder_fg = enc_fg(in_channels, latent_dim)
        self.encoder_q = enc_q(in_channels, latent_dim)

        self.decoder_fg = dec_fg(in_channels, latent_dim)
        self.decoder_q = dec_q(in_channels, latent_dim)

        self.encoder_fg.apply(init_weights)
        self.encoder_q.apply(init_weights)

        self.decoder_fg.apply(init_weights)
        self.decoder_q.apply(init_weights)

    def forward_enc(self, fg, q):
            return self.encoder_fg(fg), self.encoder_q(q)

    def forward_dec(self, fg, q):
            return self.decoder_fg(fg), self.decoder_q(q)
