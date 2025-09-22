import torch
import os
import cv2
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
import numpy as np
from PIL import Image

def round_to_nearest_multiple(x, p):
    return int(((x - 1) // p + 1) * p)

def read_image(path):

    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    return img

#/shared-local/datasets/Video_Anomaly_Segmentation/SOS/Labels

class SOS(Dataset):

    def __init__(self, hparams, transforms, mode=[]):
        super().__init__()

        self.hparams = hparams
        self.transforms = transforms
        self.mode = mode
        self.images = []
        self.labels = []

        hparams.dataset_root = '/shared-local/datasets/Video_Anomaly_Segmentation/SOS/'

        for dirname in os.listdir(os.path.join(hparams.dataset_root, 'Labels')):
            for filename in sorted(os.listdir(os.path.join(hparams.dataset_root, 'Labels',dirname))):
                
                label_path = os.path.join(hparams.dataset_root, 'Labels', dirname , filename)
                self.labels.append(label_path)

                image_path = label_path.replace('Labels', 'Images')                   
                image_path = image_path.replace('_semantic_ood.png', '_raw_data.jpg')
                self.images.append(image_path)                

        self.num_samples = len(self.images)

    def __getitem__(self, index):

        image = np.array(Image.open(self.images[index]))
        label = np.array(Image.open(self.labels[index]))
        label = np.where((label==0), 255, label)          # Background
        label = np.where((label==1), 0, label)            # Road 
        label = np.where((label>1)&(label<201), 1, label) # Anomalies
        aug = self.transforms(image=image, mask=label)
        image = aug['image']
        label = aug['mask']
        return image, label.type(torch.LongTensor), self.images[index]

    def __len__(self):

        return self.num_samples