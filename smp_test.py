
import os
import torch
import json
import numpy as np
import argparse
import time
import torch.nn.functional as F

from utils import losses, metrics, train
from tensorboardX import SummaryWriter
from torchvision import transforms

from data.LoadDataSeg import val_loader
from networks import *
from utils import NoteEvaluation
from utils.Restore import restore
from config import settings


class wce(object):

    def __init__(self):
        super(wce, self).__init__()

        self.DATASET = 'voc'
        self.SNAPSHOT_DIR = settings.SNAPSHOT_DIR
        self.device = settings.device
        self.args = self.get_arguments()

        ############ only for val WCE ##############
        self.best_weights_path = settings.best_weights_path[str(self.args.group)]
        self.abnormality_no_imgs = settings.abnormalities_imgs[self.args.abnormality]
        self.shot = settings.abnormalities_shot[self.args.abnormality]

        self.model = self.get_model(self.args)
        self._init_params(self.args)
        self._init_dataloader(self.args)

        snapshot_dir = './snapshots/{0}/{1}/group_{2}_of_3/'.format(self.args.arch, self.best_weights_path, self.args.group)

        if not os.path.exists(snapshot_dir + '/' + self.args.abnormality + '_log'):
            os.makedirs(snapshot_dir + '/' + self.args.abnormality + '_log')

        tb_path = './snapshots/{0}/{1}/group_{2}_of_3/{3}_log'.format(self.args.arch, self.best_weights_path, self.args.group, self.args.abnormality)
        self.writer = SummaryWriter(tb_path)

    def get_arguments(self):
        parser = argparse.ArgumentParser(description='OneShot')
        parser.add_argument("--arch", type=str,default='FPMMs')
        parser.add_argument("--disp_interval", type=int, default=1)
        parser.add_argument("--snapshot_dir", type=str, default=self.SNAPSHOT_DIR)

        parser.add_argument("--abnormality", type=str, default='ulcer')
        parser.add_argument("--group", type=int, default=1)
        parser.add_argument('--num_folds', type=int, default=3)
        parser.add_argument('--restore_step', type=int, default=50)
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--mode', type=str, default='val')
        parser.add_argument('--dataset', type=str, default=self.DATASET)

        return parser.parse_args()

    def get_model(self, args):

        model = eval(args.arch).OneModel(args)
        model = model.to(self.device)
        restore(args, model, self.best_weights_path)

        return model

    def check(self, args):
        val_dataloader = val_loader(args)
        
        for data in val_dataloader:

            query_img, query_mask, support_img, support_mask, class_ = data
            print(query_img.size(), query_mask.size(), support_img.size(), support_mask.size(), class_)

    def _init_params(self, args):

        self.loss_bce = losses.BCEWithLogitsLoss()
        self.loss_dice = losses.DiceLoss(activation='sigmoid')
        self.loss = losses.base.SumOfLosses(self.loss_bce, self.loss_dice)

        self.metrics = [
            metrics.IoU(activation='sigmoid'),
            metrics.Fscore(activation='sigmoid'),
            metrics.Recall(activation='sigmoid'),
            metrics.Specificity(activation='sigmoid'),
            metrics.Accuracy(activation='sigmoid')
        ]

    def _init_dataloader(self, args):

        self.val_loader = val_loader(args, self.abnormality_no_imgs, k_shot=self.shot)

    def _tensor2img(self, image):

        mean = [-x / y for x, y in zip(settings.mean_vals, settings.std_vals)]
        std = [1 / x for x in settings.std_vals]

        self.tf = transforms.Compose([
            transforms.Normalize(mean, std)
        ])

        for i in range(image.shape[0]):
            #img
            img = image[i, :, :, :].detach().cpu()
            img = self.tf(img) #.numpy()
            img = torch.unsqueeze(img, dim=0)
            if i == 0:
                im = img
            else:
                im = torch.cat((im, img), dim=0)
        return im


    def run(self, args):

        self.valid_epoch = train.ValidEpoch(
            self.model,
            loss=self.loss,
            arch=self.args.arch,
            metrics=self.metrics,
            device=self.device,
            verbose=True,
            shot=self.shot,
        )

        valid_logs, image, target, pred = self.valid_epoch.run(self.val_loader)

        image = self._tensor2img(image)

        self.writer.add_images('Test/Images', image, 1)
        self.writer.add_images('Test/Masks/True', target, 1)
        self.writer.add_images('Test/Masks/pred', (pred > .5).float(), 1)


if __name__ == '__main__':
    wce_obj = wce()
    args = wce_obj.get_arguments()
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    
    wce_obj.run(args)
