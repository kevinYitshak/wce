import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import resnet_dialated as resnet
from models import ASPP
# from models.VGPMMs import VGPMMs
# from models.VGDMMs import VGDMMs
from models.DAPMMs_phi import VGPMMs

from config import settings

class OneModel(nn.Module):

    def __init__(self, args):
        self.device = settings.device
        self.inplanes = 64
        self.num_pro = 3
        self.latent_dim = 130
        self.args = args
        if self.args.mode == 'train':
            self.shot = None 
        else:
            self.shot = settings.abnormalities_shot[self.args.abnormality]

        super(OneModel, self).__init__()

        self.model_res = resnet.Res50_Deeplab(pretrained=True)
        self.red_ch = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=self.latent_dim, kernel_size=1, bias=True),
            nn.BatchNorm2d(self.latent_dim),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # self.layer55 = nn.Sequential(
        #     nn.Conv2d(in_channels=64*3 + 1, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2,
        #               bias=True),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU()
        # )

        self.layer56 = nn.Sequential(
            nn.Conv2d(in_channels=self.latent_dim + 2*self.num_pro, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1,
                      bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.layer6 = ASPP.PSPnet()

        self.layer7 = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.layer9 = nn.Conv2d(256, 1, kernel_size=1, stride=1, bias=True)  # numclass = 2

        self.residule1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256+1*self.num_pro, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.residule2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.residule3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.VGPMMs = VGPMMs(256, self.latent_dim, self.shot, self.num_pro).to(self.device)
        # self.aspp = ASPP.PSPnet()
        
    def forward(self, query_rgb, support_rgb, support_mask):
        # extract support_ feature
        support_feature = self.extract_feature_res(support_rgb)
        # support_feature = self.red_ch(support_feature)
        # extract query feature
        query_feature = self.extract_feature_res(query_rgb)

        # PMMs
        vec_pos_fg, vec_pos_q, total_energy, total_kl_loss = \
                    self.VGPMMs(support_feature, support_mask, query_feature)

        # feature concate
        feature_size = query_feature.shape[-2:]
        query_feature = self.red_ch(query_feature)
        for i in range(self.num_pro):

            vec_fg = vec_pos_fg[i]
            # vec_bg = vec_pos_bg[i]
            vec_q = vec_pos_q[i]
            
            final_fg_ = self.cosine_similarity(vec_fg, query_feature, feature_size).unsqueeze(1) # [b, 1, h, w]
            # print('final_fg_: ', final_fg_.mean())
            # final_bg_ = self.cosine_similarity(vec_bg, query_feature, feature_size).unsqueeze(1)
            final_q_ = self.cosine_similarity(vec_q, query_feature, feature_size).unsqueeze(1)
            # print('final_q_: ', final_q_.mean())

            if i == 0:
                final_fg = final_fg_
                # final_bg = final_bg_
                final_q = final_q_
            else:
                final_fg = torch.cat((final_fg, final_fg_), dim=1)
                # final_bg = torch.cat((final_bg, final_bg_), dim=1)
                final_q = torch.cat((final_q, final_q_), dim=1)
                
        final_q = F.softmax(final_q, dim=1)
        final_fg = F.softmax(final_fg, dim=1)
        # final_bg = F.softmax(final_bg, dim=1)
        # Prob_map = torch.cat((final_fg, final_bg), dim=1)

        # self.save_prototypes(Prob_map, prefix='VGMMs_fg')
        
        final_feature_concat = torch.cat((final_fg, final_q, query_feature), dim=1) # [b, k+c, feature_size, feature_size]
        exit_feat_in = self.layer56(final_feature_concat)

        # segmentation
        out = self.Segmentation(exit_feat_in, final_fg)

        return support_feature, query_feature, vec_pos_fg, out, total_energy, total_kl_loss

    def save_prototypes(self, feature_, prefix=None):
        b, c, h, w = feature_.shape
        print(feature_.shape)
        for i in range(self.num_pro):
            feature = feature_[:, i, :, :]
            feature = feature[0].detach().cpu().numpy() * 255
            feature = cv2.resize(feature, (360, 360))
            # print(feature.shape)
            cv2.imwrite('feature_{0}_{1}.png'.format(i, prefix), feature)
        
    def forward_5shot(self, query_rgb, support_rgb_batch, support_mask_batch):
        # extract query feature
        query_feature = self.extract_feature_res(query_rgb)
        # feature concate
        feature_size = query_feature.shape[-2:]
        # print(feature_size.shape)

        for i in range(support_rgb_batch.shape[1]):
            support_rgb = support_rgb_batch[:, i]
            # print(support_rgb.shape)

            support_mask = support_mask_batch[:, i]
            # extract support feature
            support_feature = self.extract_feature_res(support_rgb)
            support_mask_temp = F.interpolate(support_mask, support_feature.shape[-2:], mode='bilinear',
                                              align_corners=True)
            if i == 0:
                support_feature_all = support_feature
                support_mask_all = support_mask_temp
            else:
                support_feature_all = torch.cat([support_feature_all, support_feature], dim=1)
                support_mask_all = torch.cat([support_mask_all, support_mask_temp], dim=1)
        # print(support_feature_all.shape)
        # print(support_mask_all.shape)
            
        vec_pos_fg, vec_pos_bg, vec_pos_q, total_energy, total_kl_loss \
         = self.VGPMMs.forward_kshot(support_feature_all, support_mask_all, query_feature, support_rgb_batch.shape[1])

        for i in range(self.shot):

            vec_fg = vec_pos_fg[i*self.num_pro:(i+1)*self.num_pro] #[2, 256*num_pro, 1, 1]
            vec_bg = vec_pos_bg[i*self.num_pro:(i+1)*self.num_pro]
            
            # print('len_vec_fg: ', len(vec_fg))
            for j in range(self.num_pro):
                # print('j: ', j)
                final_fg_ = self.cosine_similarity(vec_fg[j], query_feature, feature_size).unsqueeze(1)
                final_bg_ = self.cosine_similarity(vec_bg[j], query_feature, feature_size).unsqueeze(1)
                if j == 0 and i == 0:
                    final_fg = final_fg_
                    final_bg = final_bg_
                else:
                    final_fg = torch.cat((final_fg, final_fg_), dim=1) # [b, k*shot, 46, 46]
                    final_bg = torch.cat((final_bg, final_bg_), dim=1)
        
        for i in range(self.num_pro):
            vec_q = vec_pos_q[j]
            final_q_ = self.cosine_similarity(vec_q, query_feature, feature_size).unsqueeze(1)
            if i == 0:
                final_q = final_q_
            else:
                final_q = torch.cat((final_q, final_q_), dim=1)
        
        # print('final_fg: ', final_fg.shape)
        # print('final_bg: ', final_bg.shape)
        with torch.no_grad():

            final_fg = F.softmax(self.max_(final_fg), dim=1) #torch.max(F.softmax(final_fg, dim=1), dim=1) #
            final_bg = F.softmax(self.max_(final_bg), dim=1) #torch.max(F.softmax(final_bg, dim=1), dim=1) #
            final_q = F.softmax(final_q, dim=1)

        Prob_map = F.softmax(torch.cat((final_fg, final_bg), dim=1), dim=1)
        final_feature_concat = torch.cat((final_fg, final_q, query_feature), dim=1) # [b, k+k+c, feature_size, feature_size]
        exit_feat_in = self.layer56(final_feature_concat)

        # segmentation
        out = self.Segmentation(exit_feat_in, Prob_map)

        return support_feature, query_feature, vec_pos_fg, out, total_energy, total_kl_loss

    def max_(self, x):
        max_pool = nn.AdaptiveAvgPool1d(self.num_pro)
        b, c, h, w = x.size()
        x = x.view(b, c, h*w).permute(0, 2, 1)
        x = max_pool(x).permute(0, 2, 1)
        return x.view(b, self.num_pro, h, w)

    def cosine_similarity(self, vec_pos, query_feature, feature_size):
        feature = vec_pos.expand(-1, -1, feature_size[0], feature_size[1])
        return F.cosine_similarity(feature, query_feature)

    def extract_feature_res(self, rgb):
        out_resnet = self.model_res(rgb)
        stage2_out = out_resnet[1]
        stage3_out = out_resnet[2]
        out_23 = torch.cat([stage2_out, stage3_out], dim=1)
        feature = self.layer5(out_23)

        return feature

    def f_v_concate(self, feature, vec_pos, feature_size):
        fea_pos = vec_pos.expand(-1, -1, feature_size[0], feature_size[1])  # tile for cat
        exit_feat_in = torch.cat([feature, fea_pos], dim=1)

        return exit_feat_in

    def Segmentation(self, feature, history_mask):
        feature_size = feature.shape[-2:]

        history_mask = F.interpolate(history_mask, feature_size, mode='bilinear', align_corners=True)
        out = feature
        out_plus_history = torch.cat([out, history_mask], dim=1)
        out = out + self.residule1(out_plus_history)
        out = out + self.residule2(out)
        out = out + self.residule3(out)

        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer9(out)

        # out_softmax = F.softmax(out, dim=1)

        return out #, out_softmax