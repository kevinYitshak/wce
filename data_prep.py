import os
import cv2
import natsort

abnormalities = ['Angioectasias',
                 'Apthae',
                 'Bleeding',
                 'ChylousCysts',
                 'Lymphangectasias',
                 'Polypoids',
                 'Stenoses',
                 'Ulcers']

root = 'Documents/IISc/WCE/kid1'

for i in abnormalities:
    img_list = os.listdir(os.path.join(root, i, 'train/images/'))
    
    names_txt = open(i + '.txt','w')
    img_list = map(lambda x:x+'\n', img_list)
    names_txt.writelines(img_list)
    names_txt.close()