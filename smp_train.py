import os
import json
import argparse
import time
import numpy as np
import cv2
import torch

from config import settings
from utils import my_optim

from utils import losses, metrics, train
from tensorboardX import SummaryWriter
from datetime import datetime
from torchvision import transforms
from matplotlib import pyplot as plt

from utils.Restore import restore
from utils.Restore import get_model_para_number
from utils.Restore import get_save_dir
from utils.Restore import save_model
from data.LoadDataSeg import data_loader, val_loader
from utils import NoteLoss
from utils import NoteEvaluation
from utils import Visualize
from networks import *

torch.autograd.set_detect_anomaly(True)

class wce(object):

    def __init__(self):
        super(wce, self).__init__()

        self.LR = settings.LR
        self.DATASET = 'voc'
        self.SNAPSHOT_DIR = settings.SNAPSHOT_DIR
        self.device = settings.device

        self.args = self.get_arguments()
        self._init_logger(self.args)
        self.model = self.get_model(self.args)
        self._init_dataloader(self.args)
        self._init_params(self.args)


    def get_arguments(self):
        parser = argparse.ArgumentParser(description='OneShot')
        parser.add_argument("--arch", type=str,default='VGMMs') #
        parser.add_argument("--max_steps", type=int, default=15) #
        parser.add_argument("--lr", type=float, default=self.LR)
        parser.add_argument("--disp_interval", type=int, default=2)
        parser.add_argument("--save_interval", type=int, default=5)
        parser.add_argument("--snapshot_dir", type=str, default=self.SNAPSHOT_DIR)
        parser.add_argument("--resume", action='store_true')
        parser.add_argument("--start_count", type=int, default=0)

        # parser.add_argument("--abnormality", type=str, default='ulcer')
        parser.add_argument("--split", type=str, default='mlclass_train') # train mlclass_train mlclass_train_deeplab
        parser.add_argument("--group", type=int, default=0)
        parser.add_argument('--num_folds', type=int, default=3)
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--mode', type=str, default='train')
        parser.add_argument('--dataset', type=str, default=self.DATASET)
        parser.add_argument('--epochs', type=str, default=10)

        return parser.parse_args()

    def _init_logger(self, args):

        d = datetime.now().strftime('%Y-%m-%d~%H:%M:%S')
        self.snapshot_dir = os.path.join(args.snapshot_dir, args.arch, d, 'group_%d_of_%d'%(args.group, args.num_folds))
        
        if not os.path.exists(args.snapshot_dir):
            os.mkdir(args.snapshot_dir)

        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)
        if not os.path.exists(self.snapshot_dir + '/ckpt'):
            os.makedirs(self.snapshot_dir + '/ckpt')
            os.makedirs(self.snapshot_dir + '/log')
            os.makedirs(self.snapshot_dir + '/Grad_log')

        self.save_tbx_log = self.snapshot_dir + '/log'
        self.save_tbx_cktp = self.snapshot_dir + '/ckpt'
        self.writer = SummaryWriter(self.save_tbx_log)

    def get_model(self, args):

        self.model = eval(args.arch).OneModel(args)
        # opti_A = my_optim.get_finetune_optimizer(args, model)
        self.model = self.model.to(self.device) #model.cuda()
        print('Number of Parameters: %d' % (get_model_para_number(self.model)))

        if args.start_count > 0:
            restore(args, self.model, best_weights_path=None)
            print("Resume training...")

        return self.model #, opti_A

    def check(self, args):
        self.train_loader = data_loader(args, mode='train')
        self.val_loader = data_loader(args, mode='val')

        for data in self.train_loader:
            print('--------- TRAIN -----------')
            query_img, query_mask, support_img, support_mask, idx = data
            print(query_img.size(), query_mask.size(), support_img.size(), support_mask.size(), idx)

        for data in self.val_loader:
            print('--------- VAL -----------')
            query_img, query_mask, support_img, support_mask, idx = data
            print(query_img.size(), query_mask.size(), support_img.size(), support_mask.size(), idx)
        
    def _init_dataloader(self, args):

        self.train_loader = data_loader(args, mode='train')
        self.val_loader = data_loader(args, mode='val')

    def _init_params(self, args):

        self.end_epoch = args.epochs
        self.batch_size = args.batch_size

        self.loss_bce = losses.BCEWithLogitsLoss()
        self.loss_dice = losses.DiceLoss(activation='sigmoid')
        self.loss = losses.base.SumOfLosses(self.loss_bce, self.loss_dice)
        # self.loss = losses.base.MultipliedLoss(self.loss, 0.5)

        self.metrics = [
            metrics.IoU(activation='sigmoid'),
            metrics.Fscore(activation='sigmoid'),
            metrics.Recall(activation='sigmoid')
        ]
        self.optimizer = torch.optim.Adam([
            dict(params=self.model.parameters(), lr=args.lr),
        ])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max= len(self.train_loader) * self.batch_size * 2)

    def run(self, args):

        self.train_epoch = train.TrainEpoch(
            self.model,
            arch=self.args.arch,
            loss=self.loss,
            metrics=self.metrics,
            optimizer=self.optimizer,
            device=self.device,
            verbose=True,
            shot=None,
        )

        self.val_epoch = train.ValidEpoch(
            self.model,
            arch=self.args.arch,
            loss=self.loss,
            metrics=self.metrics,
            device=self.device,
            verbose=True,
            shot=1,
        )

        self.best_dice = 0

        for epoch in range(0, self.end_epoch):

            self.epoch = epoch
            print('-------------- Epoch: %d/%d --------------' % (self.epoch + 1, self.end_epoch))

            self._train()
            self.plot_grad_flow(self.model.named_parameters())
            self.scheduler.step()
            print('----- VAL -----')
            self._val()

        print('Best_Dice: {:.4f}, Best_Sen: {:.4f}, Best_epoch: {}'
              .format(self.best_dice, self.best_sen, self.best_epoch))

    def plot_grad_flow(self, named_parameters):
        ave_grads = []
        layers = []
        for n, p in named_parameters:
            if p.grad is not None and (p.requires_grad) and ("bias" not in n):
                # print(n, p.grad.sum())
                layers.append(n[:6])
                ave_grads.append(p.grad.abs().mean())
            else:
                continue # print(n, p.grad)                
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.savefig('./{0}/Grad_log/Grad_flow_{1}.png'.format(self.snapshot_dir, self.epoch))

    def _tensor2img(self, image):

        mean = [-x / y for x, y in zip(settings.mean_vals, settings.std_vals)]
        std = [1 / x for x in settings.std_vals]

        self.tf = transforms.Compose([
            transforms.Normalize(mean, std)
        ])

        for i in range(image.shape[0]):
            img = image[i, :, :, :].detach().cpu()
            img = self.tf(img) #.numpy()
            img = torch.unsqueeze(img, dim=0)
            if i == 0:
                im = img
            else:
                im = torch.cat((im, img), dim=0)
            
        # print('img_size: ', im.size())

        return im

    def _train(self):

        train_logs, image, target, pred = self.train_epoch.run(self.train_loader)

        image = self._tensor2img(image)

        # self.writer.add_histogram('Model Params', self.model.parameters().grad, self.epoch)
        self.writer.add_images('Train/Images', image, self.epoch)
        self.writer.add_images('Train/Masks/True', target, self.epoch)
        self.writer.add_images('Train/Masks/pred', (pred > .5).float(), self.epoch)

        self.writer.add_scalar('Train/loss', train_logs['bce_logit + dice_loss'], self.epoch)
        self.writer.add_scalar('Train/Sen', train_logs['sen'], self.epoch)
        self.writer.add_scalar('Train/IOU', train_logs['iou'], self.epoch)
        self.writer.add_scalar('Train/Dice', train_logs['dice'], self.epoch)

    def _val(self):

        val_logs, image, target, pred = self.val_epoch.run(self.val_loader)

        image = self._tensor2img(image)

        self.writer.add_images('Val/Images', image, self.epoch)
        self.writer.add_images('Val/Masks/True', target, self.epoch)
        self.writer.add_images('Val/Masks/pred', (pred > .5).float(), self.epoch)

        self.writer.add_scalar('Val/loss', val_logs['bce_logit + dice_loss'], self.epoch)
        self.writer.add_scalar('Val/Sen', val_logs['sen'], self.epoch)
        self.writer.add_scalar('Val/IOU', val_logs['iou'], self.epoch)
        self.writer.add_scalar('Val/Dice', val_logs['dice'], self.epoch)

        if self.best_dice < val_logs['dice']:
            self.best_dice = val_logs['dice']
            self.best_sen = val_logs['sen']
            self.best_epoch = self.epoch

            ckpt_file_path = self.save_tbx_cktp + '/best_weights.pth.tar'
            torch.save(
                {
                    'args': self.args,
                    'state_dict': self.model.state_dict(),
                }, ckpt_file_path)

            print('-----------------New Weights Saved------------------')


    def save_pred(self, pred1, save_log_dir):
        print('pred shape: ', pred1.size())
        k = 0
        for i in range(pred1.shape[0]):

            pred = pred1[i, :, :].detach().cpu().numpy()
            print('pred shape: ', pred.shape)
            # pred = np.transpose(pred, (1, 2, 0))
            # pred = pred > 0.5
            pred = pred * 255
            pred.astype('uint8')
            # pred = np.squeeze(pred, axis=-1)
            print(save_log_dir, str(k + i))
            cv2.imwrite(save_log_dir + '/' + str(k + i) + '.png', pred)
        k += 1

if __name__ == '__main__':
    
    wce_obj = wce()
    args = wce_obj.get_arguments()
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    
    wce_obj.run(args) 
    #train(args)
    # wce_obj.check(args)
