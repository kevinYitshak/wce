
import os
import os.path as osp
import torch
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
DATA_DIR = osp.abspath('./DATASETS/') #data/VOC_new

# print(ROOT_DIR)
# print(DATA_DIR)

# Dataloader
# mean_vals = [0.485, 0.456, 0.406]
# std_vals = [0.229, 0.224, 0.225]

# Dataloader for RGBA
mean_vals = [0.438, 0.273, 0.193] #, 0.561]
std_vals = [0.335, 0.239, 0.169] #, 0.051]

size = 360
abnormalities = {'angioectasia': 0,
                    'aphtha' : 1,
                    'bleeding': 4,
                    'Chylous': 5,
                    'lymphangiectasia': 2,
                    'polypoid': 3,
                    'stenosis': 6,
                    'ulcer': 7}

# -----------------------------------------------------------------------------
# model
# -----------------------------------------------------------------------------
LR = 3.5e-4
SNAPSHOT_DIR = os.path.join('snapshots')
best_weights_path = {'0': '2020-12-11~13:17:59',
                    '1': '2020-12-11~15:04:46', 
                    '2': '2020-11-14~11_05_54',
                    '3': '2020-12-05~17_08_23'}

# -----------------------------------------------------------------------------
# val
# -----------------------------------------------------------------------------
abnormalities_imgs = {'angioectasia': 27,
                    'aphtha' : 5,
                    'bleeding': 5,
                    'Chylous': 8,
                    'lymphangiectasia': 9,
                    'polypoid': 6,
                    'stenosis': 6,
                    'ulcer': 9}

abnormalities_shot = {'angioectasia': 1,
                    'aphtha' : 1,
                    'bleeding': 1,
                    'Chylous': 1,
                    'lymphangiectasia': 1,
                    'polypoid': 1,
                    'stenosis': 1,
                    'ulcer': 1}