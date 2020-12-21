import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import resnet_dialated as resnet
from models import ASPP
from models.DAPMMs import DAPMMs

from config import settings

class OneModel(nn.Module):

    def __init__(self, args):
        self.device = settings.device
        self.inplanes = 64
        self.num_pro_list = [2,4,6]
        # self.num_pro = self.num_pro_list[0]
        self.args = args
        self.shot = 1 #settings.abnormalities_shot[self.args.abnormality]

        super(OneModel, self).__init__()

        self.model_res = resnet.Res50_Deeplab(pretrained=True)
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer55 = nn.Sequential(
            nn.Conv2d(in_channels=512 + 256 + 1, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2,
                      bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.layer56 = nn.Sequential(
            nn.Conv2d(in_channels=256 + 2, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1,
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
            nn.Conv2d(256+1, 256, kernel_size=3, stride=1, padding=1, bias=True),
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

        # self.DAPMMs = DAPMMs(256, self.shot, self.num_pro).to(self.device)
        # self.aspp = ASPP.PSPnet()

    
    def forward(self, query_rgb, support_rgb, support_mask):
        # extract support_ feature
        support_feature = self.extract_feature_res(support_rgb)

        # extract query feature
        query_feature = self.extract_feature_res(query_rgb)
        # feature concate
        feature_size = query_feature.shape[-2:]

        Pseudo_mask = (torch.zeros(query_feature.size(0), 1, 50, 50)).to(self.device) #cuda()
        out_list = []
        for num in self.num_pro_list:
            self.num_pro = num
            self.DAPMMs = DAPMMs(256, k_shot=self.shot, k=num).to(self.device) #cuda()
            vec_pos, Prob_map = self.DAPMMs(support_feature, support_mask, query_feature)

            for i in range(num):
                vec = vec_pos[i]
                exit_feat_in_ = self.f_v_concate(query_feature, vec, feature_size)
                exit_feat_in_ = self.layer55(exit_feat_in_)
                if i == 0:
                    exit_feat_in = exit_feat_in_
                else:
                    exit_feat_in = exit_feat_in + exit_feat_in_

            exit_feat_in = torch.cat([exit_feat_in, Prob_map], dim=1)
            exit_feat_in = self.layer56(exit_feat_in)

            # segmentation
            out, out_softmax = self.Segmentation(exit_feat_in, Pseudo_mask)
            Pseudo_mask = out_softmax
            out_list.append(out)

        return support_feature, out_list[0], out_list[1], out

    def forward_5shot(self, query_rgb, support_rgb_batch, support_mask_batch):
        # extract query feature
        query_feature = self.extract_feature_res(query_rgb)

        feature_size = query_feature.shape[-2:]

        out5 = 0

        for i in range(support_rgb_batch.shape[1]):
            support_rgb = support_rgb_batch[:, i]
            support_mask = support_mask_batch[:, i]
            # extract support feature
            support_feature = self.extract_feature_res(support_rgb)
            support_mask_temp = F.interpolate(support_mask, support_feature.shape[-2:], mode='bilinear',
                                              align_corners=True)
            if i == 0:
                support_feature_all = support_feature
                support_mask_all = support_mask_temp
            else:
                support_feature_all = torch.cat([support_feature_all, support_feature], dim=2)
                support_mask_all = torch.cat([support_mask_all, support_mask_temp], dim=2)

        Pseudo_mask = (torch.zeros(query_feature.size(0), 2, 50, 50)).to(self.device) #cuda()
        for num in self.num_pro_list:
            self.num_pro = num
            self.PMMs = PMMs(256, num).to(self.device) #cuda()
            vec_pos, Prob_map = self.DAGMMs(support_feature_all, support_mask_all, query_feature)
            # vector conduct feature
            for i in range(num):
                vec = vec_pos[i]
                exit_feat_in_ = self.f_v_concate(query_feature, vec, feature_size)
                exit_feat_in_ = self.layer55(exit_feat_in_)
                if i == 0:
                    exit_feat_in = exit_feat_in_
                else:
                    exit_feat_in = exit_feat_in + exit_feat_in_

            exit_feat_in = torch.cat([exit_feat_in, Prob_map], dim=1)
            exit_feat_in = self.layer56(exit_feat_in)

            # segmentation
            out, out_softmax = self.Segmentation(exit_feat_in, Pseudo_mask)
            Pseudo_mask = out_softmax

            out5 = out5 + out_softmax
        out5 = out5 / 5
        return out5, out5, out5, out5

        return logits

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

        out_softmax = F.softmax(out, dim=1)

        return out, out_softmax