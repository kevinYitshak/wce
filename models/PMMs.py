import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import settings
from visualise import latent_vis

class PMMs(nn.Module):
    '''Prototype Mixture Models
    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''

    def __init__(self, c, k=3, stage_num=10):
        super(PMMs, self).__init__()

        self.device = settings.device
        self.stage_num = stage_num
        self.num_pro = k
        mu = torch.Tensor(1, c, k).to(self.device) #torch.Tensor(1, c, k).cuda()
        mu.normal_(0, math.sqrt(2. / k))  # Init mu
        self.mu = self._l2norm(mu, dim=1)
        self.kappa = 20
        #self.register_buffer('mu', mu)


    def forward(self, support_feature, support_mask, query_feature):
        # print('support_feature_size: ', support_feature.size())
        # print('query_feature_size: ', query_feature.size())

        prototypes, mu_f, mu_b = self.generate_prototype(support_feature, support_mask)
        #####
        # q_prototypes, mu_qf, mu_qb = self.generate_prototype_query(query_feature, mu_f, mu_b)
        #####
        Prob_map, P = self.discriminative_model(query_feature, mu_f, mu_b)

        # Prob_map_support, P_support = self.discriminative_model_support(support_feature, mu_qf, mu_qb)

        # print('prototypes: ', len(prototypes))
        # print('prototypes_size: ', prototypes[0].size())
        # print('Prob_map: ', Prob_map.size())
        return prototypes, Prob_map

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.
        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.
        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.
        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    def EM(self,x, mu_init=None):
        '''
        EM method
        :param x: feauture  b * c * n
        :return: mu
        '''
        # b = x.shape[0]
        mu = mu_init #self.mu.repeat(b, 1, 1)  # b * c * k
        # print('mu_init_shape: ', mu_init.shape)
        with torch.no_grad():
            for i in range(self.stage_num):
                # E STEP:
                z = self.Kernel(x, mu)
                z = F.softmax(z, dim=2)  # b * n * k
                # print('z size: ', z.size())
                # M STEP:
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                # print('z_ size: ', z_.size())
                mu = torch.bmm(x, z_)  # b * c * k

                mu = self._l2norm(mu, dim=1)

        mu = mu.permute(0, 2, 1)  # b * k * c
        # print('mu: ', mu.size())
        return mu

    def Kernel(self, x, mu):
        x_t = x.permute(0, 2, 1)  # b * n * c
        # print('x_t shape: ', x_t.shape)
        z = self.kappa * torch.bmm(x_t, mu)  # b * n * k

        return z

    def get_prototype(self,x, mu_init=None):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n
        mu = self.EM(x, mu_init) # b * k * c

        return mu

    def generate_prototype(self, feature, mask):
        mask = F.interpolate(mask, feature.shape[-2:], mode='bilinear', align_corners=True)

        mask_bg = 1-mask

        # foreground
        z = mask * feature
        # print(z)
        # print('z_size: ', z.size())
        mu_f = self.get_prototype(z, mu_init=self.mu.repeat(z.shape[0], 1, 1))
        # print('mu_f size: ', mu_f.shape)
        
        # self.Vis(z, mu_f, 'embeddings_Chylous_pmm_fg.png')

        mu_ = []
        for i in range(self.num_pro):
            mu_.append(mu_f[:, i, :].unsqueeze(dim=2).unsqueeze(dim=3))
        # print(mu_[0].size())
        # background
        z_bg = mask_bg * feature
        mu_b = self.get_prototype(z_bg, mu_init=self.mu.repeat(z.shape[0], 1, 1))

        return mu_, mu_f, mu_b

    def Vis(self, f1, f2, name):
        visualise = latent_vis.visualise(f1, f2)
        embeddings, mu = visualise._tsne()
        visualise._plot(embeddings, name)
        visualise._plot(mu, name)

    def generate_prototype_query(self, query_feature, mu_f, mu_b):
        
        mu_qf = self.get_prototype(query_feature, mu_f.permute(0, 2, 1))
        # print('mu_qf shape: ', mu_qf.shape)
        mu_qb = self.get_prototype(query_feature, mu_b.permute(0, 2, 1))
        # print('mu_qb shape: ', mu_qb.shape)

        mu_ = []
        for i in range(self.num_pro):
            mu_.append(mu_qf[:, i, :].unsqueeze(dim=2).unsqueeze(dim=3))

        return mu_, mu_qf, mu_qb

    def discriminative_model(self, query_feature, mu_f, mu_b):

        mu = torch.cat([mu_f, mu_b], dim=1)
        mu = mu.permute(0, 2, 1)

        b, c, h, w = query_feature.size()
        x = query_feature.view(b, c, h * w)  # b * c * n
        with torch.no_grad():

            x_t = x.permute(0, 2, 1)  # b * n * c
            z = torch.bmm(x_t, mu)  # b * n * k

            z = F.softmax(z, dim=2)  # b * n * k

        P = z.permute(0, 2, 1)

        P = P.view(b, self.num_pro * 2, h, w) #  b * k * w * h  probability map
        # self.save_prob_map(P, prefix='FPMMs_fg')
        # self.save_cosine(mu_f, query_feature, prefix='FPMMs_mu_f')
        

        P_f = torch.sum(P[:, 0:self.num_pro], dim=1).unsqueeze(dim=1) # foreground
        P_b = torch.sum(P[:, self.num_pro:], dim=1).unsqueeze(dim=1) # background

        Prob_map = torch.cat([P_b, P_f], dim=1)

        return Prob_map, P

    def save_cosine(self, mu, query_feature, prefix=None):
        for i in range(self.num_pro):
            mu_ = mu[:, i, :].unsqueeze(2).unsqueeze(3)
            print(mu_.shape)
            final_fg = self.cosine_similarity(mu_, query_feature, query_feature.shape[-2:]) # [b, h, w]
            final_fg = final_fg[0].detach().cpu().numpy() * 255
            final_fg = cv2.resize(final_fg, (360, 360))
            # print(feature.shape)
            cv2.imwrite('feature_{0}_{1}.png'.format(i, prefix), final_fg)

    def cosine_similarity(self, vec_pos, query_feature, feature_size):
        feature = vec_pos.expand(-1, -1, feature_size[0], feature_size[1])
        return F.cosine_similarity(feature, query_feature)

    def save_prob_map(self, feature_, prefix=None):
        b, c, h, w = feature_.shape
        print(feature_.shape)
        # history_mask = F.interpolate(feature, feature.shape[], mode='bilinear', align_corners=True)
        for i in range(self.num_pro):
            feature = feature_[:, i, :, :]
            feature = feature[0].detach().numpy() * 255
            feature = cv2.resize(feature, (360, 360))
            # print(feature.shape)
            cv2.imwrite('feature_{0}_{1}.png'.format(i, prefix), feature)

    def discriminative_model_support(self, support_feature, mu_qf, mu_qb):

        mu = torch.cat([mu_qf, mu_qb], dim=1)
        mu = mu.permute(0, 2, 1)

        b, c, h, w = support_feature.size()
        x = support_feature.view(b, c, h * w)  # b * c * n
        with torch.no_grad():

            x_t = x.permute(0, 2, 1)  # b * n * c
            z = torch.bmm(x_t, mu)  # b * n * k

            z = F.softmax(z, dim=2)  # b * n * k

        P = z.permute(0, 2, 1)

        P = P.view(b, self.num_pro * 2, h, w) #  b * k * w * h  probability map
        P_f = torch.sum(P[:, 0:self.num_pro], dim=1).unsqueeze(dim=1) # foreground
        P_b = torch.sum(P[:, self.num_pro:], dim=1).unsqueeze(dim=1) # background

        Prob_map = torch.cat([P_b, P_f], dim=1)

        return Prob_map, P
