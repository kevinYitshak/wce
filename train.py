
import os
import json
import argparse
import time
import numpy as np
import cv2
import torch 

from config import settings
from utils import my_optim
from utils.Restore import restore
from utils.Restore import get_model_para_number
from utils.Restore import get_save_dir
from utils.Restore import save_model
from data.LoadDataSeg import data_loader
from utils import NoteLoss
from utils import NoteEvaluation
from utils import Visualize
from networks import *

LR = settings.LR
DATASET = 'voc'
SNAPSHOT_DIR = settings.SNAPSHOT_DIR
device = settings.device

#GPU_ID = '6'
#os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

def get_arguments():
    parser = argparse.ArgumentParser(description='OneShot')
    parser.add_argument("--arch", type=str,default='FPMMs') #
    parser.add_argument("--max_steps", type=int, default=10) #
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--disp_interval", type=int, default=2)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--start_count", type=int, default=0)

    parser.add_argument("--split", type=str, default='mlclass_train') # train mlclass_train mlclass_train_deeplab
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument('--num_folds', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--dataset', type=str, default=DATASET)

    return parser.parse_args()


def get_model(args):

    model = eval(args.arch).OneModel(args)
    opti_A = my_optim.get_finetune_optimizer(args, model)
    model = model.to(device) #model.cuda()
    print('Number of Parameters: %d' % (get_model_para_number(model)))

    if args.start_count > 0:
        restore(args, model)
        print("Resume training...")

    return model, opti_A

def check(args):
    train_loader = data_loader(args)
    
    for data in train_loader:

        query_img, query_mask, support_img, support_mask, idx = data
        print(query_img.size(), query_mask.size(), support_img.size(), support_mask.size(), idx)


def train(args):
    model, optimizer = get_model(args)
    model.train()
    train_loader = data_loader(args)
    losses = NoteLoss.Loss_total(args)
    evaluations = NoteEvaluation.Evaluation(args)
    watch = Visualize.visualize_loss_evl_train(args)

    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    if not os.path.exists(get_save_dir(args)):
        os.makedirs(get_save_dir(args))
    save_log_dir = get_save_dir(args)
    log_file = open(os.path.join(save_log_dir, 'log.txt'), 'w')

    count = args.start_count

    print('Start training')

    for data in train_loader:

        count += 1
        print('count: ', count)
        if count % args.disp_interval == 1:
            begin_time = time.time()
        if count > args.max_steps:
            break

        my_optim.adjust_learning_rate_poly(args, optimizer, count, power=0.9)

        query_img, query_mask, support_img, support_mask, idx = data
        query_img, query_mask, support_img, support_mask, idx \
            = query_img.to(device, dtype=torch.float32), query_mask.to(device, dtype=torch.float32), \
             support_img.to(device, dtype=torch.float32),support_mask.to(device, dtype=torch.float32), idx.to(device)
            # query_img.cuda(), query_mask.cuda(), support_img.cuda(),support_mask.cuda(), idx.cuda()

        logits = model(query_img, support_img, support_mask)
        loss_val, loss_part1, loss_part2 = model.get_loss(logits, query_mask, idx)

        losses.updateloss(loss_val,loss_part1, loss_part2)
        losses.logloss(log_file)

        # out_softmax, pred = model.get_pred(logits, query_img)
        out_sigmoid = model.get_pred(logits, query_img)
        evaluations.update_evl(idx, query_mask, out_sigmoid, count)

        # save_pred(pred, save_log_dir)
        
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        watch.visualize(args, count, losses, evaluations, begin_time)
        save_model(args, count, model, optimizer)

    log_file.close()

def save_pred(pred1, save_log_dir):
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
    args = get_arguments()
    if args.dataset == 'coco':
        args.snapshot_dir = args.snapshot_dir + '/coco'
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    train(args)
    # check(args)
