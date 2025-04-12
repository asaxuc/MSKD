import glob
import os
import os.path as osp
import xml.etree.ElementTree as ET
import mmcv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.coco import CutoutPIL


class FlickrDataset(Dataset):

    def __init__(self, args, img_prefix, ann_file, class_name, img_size=224, train_mode=False, use=None):
        super(FlickrDataset, self)

        self.CLASSES = class_name
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.img_premix = img_prefix
        # self.ann_file = ann_file
        self.length = 25000
        self.train_mode = train_mode 
        self.gt_labels = np.zeros((int(self.length), len(self.CLASSES)), dtype=np.int64)
        if train_mode: 
            self.transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    CutoutPIL(cutout_factor=0.2),
                    transforms.RandAugment(),
                    transforms.ToTensor(),
                ]) 
        else: 
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
        self.update_whole() 
        
        if self.train_mode:
            self.length = int(self.length*0.8)
            self.gt_labels = self.gt_labels[:20000]
        else:
            self.length = int(self.length*0.2)
            self.gt_labels = self.gt_labels[20000:]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.train_mode:
            img_path = osp.join(self.img_premix,"mirflickr", "im{}.jpg".format(idx+1)) 
        else:
            img_path = osp.join(self.img_premix,"mirflickr", "im{}.jpg".format(idx+1+20000)) 
            
        pil_img = Image.open(img_path).convert("RGB")
        img = self.transform(pil_img)
        
        return img, self.gt_labels[idx] 

    def update_whole(self):
        for cls in self.CLASSES:
            clll = self.cat2label[cls]
            file = os.path.join(self.img_premix,"annotations", "{}.txt".format(cls))
            fli = open(file,"r").readlines()
            if fli == []:
                print("wrong") 
            for idx, item in enumerate(fli): 
                item = int(item.strip("\n"))
                self.gt_labels[item-1][clll] = 1 
