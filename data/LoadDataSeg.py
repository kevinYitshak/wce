from __future__ import print_function
from __future__ import absolute_import

from config import settings
from data.transforms import transforms
from torch.utils.data import DataLoader
from data.voc_train import voc_train
from data.voc_val import voc_val
from PIL import Image
#from data.coco_train import coco_train
#from data.coco_val import coco_val


def data_loader(args, mode=None):

    batch = args.batch_size
    mean_vals = settings.mean_vals
    std_vals = settings.std_vals
    size = settings.size

    tsfm_train = transforms.Compose([transforms.ToPILImage(),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])

    if args.dataset == 'voc':
        img_train = voc_train(args, mode, transform=None)

    train_loader = DataLoader(img_train, batch_size=batch, shuffle=True, num_workers=0)

    return train_loader

def val_loader(args, abnormality_no_imgs, k_shot=1):
    mean_vals = settings.mean_vals
    std_vals = settings.std_vals

    tsfm_val = transforms.Compose([transforms.ToPILImage(),
                                   transforms.Resize(size=(360,360)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean_vals, std_vals)
                                   ])

    #if args.dataset == 'coco':
        #img_val = coco_val(args, transform=tsfm_val, k_shot=k_shot)
    if args.dataset == 'voc':
        img_val = voc_val(args, abnormality_no_imgs, transform=tsfm_val, k_shot=k_shot)


    val_loader = DataLoader(img_val, batch_size=args.batch_size, shuffle=False, num_workers=1)

    return val_loader
