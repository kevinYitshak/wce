from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import random
import PIL.Image as Image
import numpy as np
from config import settings
from utils import Aug
from torchvision import transforms

#random.seed(1385)

class voc_train():

    """Face Landmarks dataset."""

    def __init__(self, args, mode=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mode = mode
        self.num_classes = 8
        self.group = args.group
        self.num_folds = args.num_folds
        #self.binary_map_dir = os.path.join(settings.DATA_DIR, 'VOCdevkit2012/VOC2012/', 'Binary_map_aug/train') #val
        self.data_list_dir = os.path.join('data_list/train')
        self.img_dir = os.path.join(settings.DATA_DIR, 'IMAGES/')
        self.mask_dir = os.path.join(settings.DATA_DIR, 'MASKS/')
        #self.binary_mask_dir = os.path.join(settings.DATA_DIR, 'VOCdevkit2012/VOC2012/', 'Binary_map_aug/train/')

        self.abnormalities = settings.abnormalities

        self.train_id_list, self.val_id_list = self.get_train_id_list()
        print('Train set: ', self.train_id_list)
        print('Val set: ', self.val_id_list)

        self.list_splite = self.get_total_list()
        
        self.list_splite_len = len(self.list_splite)
        
        self.list_class = self.get_class_list()
        # print('list class: ', self.list_class)

        self.transform = transform
        self.aug = Aug.Augmentation()

        self._img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=settings.mean_vals, std=settings.std_vals)
        ])
        self._mask = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        self.count = 0
        self.random_generator = random.Random()
        if self.mode == 'train':
            self.len = args.max_steps * args.batch_size *2
        else:
            self.len = len(self.list_splite)
        #self.random_generator.shuffle(self.list_splite)
        #self.random_generator.seed(1385)

    def get_train_id_list(self):
        num = int(self.num_classes/ self.num_folds)
        val_set = [self.group * num + v for v in range(num)]
        train_set = [x for x in range(self.num_classes) if x not in val_set]

        return train_set, val_set

    def get_total_list(self):
        new_exist_class_list = []

        fold_list = [0, 1, 2, 3]
        print('Mode: ', self.mode)
        if self.mode == 'train':
            fold_list.remove(self.group)
            print('FOLD: ', fold_list)
            for fold in fold_list:
                f = open(os.path.join(self.data_list_dir, 'split%1d_train.txt' % (fold)))
                while True:
                    item = f.readline()
                    if item == '':
                        break
                    img_name = item[:len(item)] #item[:11]
                    # print('img name:', img_name)

                    if 'angioectasia' in img_name:
                        cat = self.abnormalities[img_name[:len(img_name)-6]]
                    else:
                        cat = self.abnormalities[img_name[:len(img_name)-2]]
                    new_exist_class_list.append([img_name, cat])
            print("Total images are : ", len(new_exist_class_list))
        else:
            fold = self.group
            print('FOLD: ', fold)
            f = open(os.path.join(self.data_list_dir, 'split%1d_train.txt' % (fold)))
            while True:
                item = f.readline()
                if item == '':
                    break
                img_name = item[:len(item)] #item[:11]
                # print('img name:', img_name)

                if 'angioectasia' in img_name:
                    cat = self.abnormalities[img_name[:len(img_name)-6]]
                else:
                    cat = self.abnormalities[img_name[:len(img_name)-2]]
                new_exist_class_list.append([img_name, cat])

            print("Total images are : ", len(new_exist_class_list))
        # if need filter
        new_exist_class_list = self.filte_multi_class(new_exist_class_list)
        return new_exist_class_list

    def filte_multi_class(self, exist_class_list):

        new_exist_class_list = []
        for name, class_ in exist_class_list:

            mask_path = self.mask_dir + name[:-1] + 'm.png'
            # print(name, len(name))
            mask = cv2.imread(str(mask_path))
            # print(mask.shape)
            labels = np.unique(mask[:,:,0])

            labels = [label - 1 for label in labels if label != 255 and label != 0]
            if set(labels).issubset(self.train_id_list):
                new_exist_class_list.append([name, class_])
        print("Total images after filted are : ", len(new_exist_class_list))
        return new_exist_class_list


    def get_class_list(self):
        list_class = {}
        for i in range(self.num_classes):
            list_class[i] = []
        for name, class_ in self.list_splite:
            # print(name, class_)
            list_class[class_].append(name)

        return list_class

    def read_img(self, name):
        path = self.img_dir + name[:-1] + '.png'
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def read_mask(self, name, category):
        path = self.mask_dir + name[:-1] + 'm.png'
        mask = cv2.imread(path, 0) 
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        # mask = mask / 255
        # mask[mask!=category+1] = 0
        # mask[mask==category+1] = 1

        return mask #.astype(np.float32)
    '''
    def read_binary_mask(self, name, category):
        path = self.binary_mask_dir +str(category+1)+'/'+ name + '.png'
        mask = cv2.imread(path)/255

        return mask[:,:,0].astype(np.float32)
    '''
    def load_frame(self, support_name, query_name, class_):
        support_img = self.read_img(support_name)
        query_img = self.read_img(query_name)
        support_mask = self.read_mask(support_name, class_)
        query_mask = self.read_mask(query_name, class_)

        #support_mask = self.read_binary_mask(support_name, class_)
        #query_mask = self.read_binary_mask(query_name, class_)

        return query_img, query_mask, support_img, support_mask, class_

    def random_choose(self):

        if self.mode == 'train':
            class_ = np.random.choice(self.train_id_list, 1, replace=False)[0]
            # print('class: ', class_)
        else:
            class_ = np.random.choice(self.val_id_list, 1, replace=False)[0]
            # print('class: ', class_)

        cat_list = self.list_class[class_]
        # print(cat_list, len(cat_list))

        sample_img_ids_1 = np.random.choice(len(cat_list), 2, replace=False)

        query_name = cat_list[sample_img_ids_1[0]]
        print('query image name: ', query_name)
        support_name = cat_list[sample_img_ids_1[1]]
        print('support image name: ', support_name)

        return support_name, query_name, class_

    def show(self, img):
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __len__(self):
        return 2 #self.len #len(self.train_id_list) 

    def __getitem__(self, idx):
        support_name, query_name, class_ = self.random_choose()

        query_img, query_mask, support_img, support_mask, class_ = self.load_frame(support_name, query_name, class_)

        _aug = self.aug._aug()
        query_augmented = _aug(image=query_img, mask=query_mask)
        query_img = query_augmented['image']
        query_mask = query_augmented['mask']

        # _, query_mask = cv2.threshold(query_mask, 200, 255, cv2.THRESH_BINARY)
        # self.show(query_img)
        # self.show(query_mask)

        query_img = self._img(query_img)
        query_mask = self._mask(query_mask)

        support_augmented = _aug(image=support_img, mask=support_mask)
        support_img = support_augmented['image']
        support_mask = support_augmented['mask']

        # _, support_mask = cv2.threshold(support_mask, 200, 255, cv2.THRESH_BINARY)
        # self.show(support_img)
        # self.show(support_mask)

        support_img = self._img(support_img)
        support_mask = self._mask(support_mask)

        # if self.transform is not None:
        #     query_img, query_mask = self.transform(query_img, query_mask)
        #     support_img, support_mask = self.transform(support_img, support_mask)

        self.count = self.count + 1

        return query_img, query_mask, support_img, support_mask, class_
