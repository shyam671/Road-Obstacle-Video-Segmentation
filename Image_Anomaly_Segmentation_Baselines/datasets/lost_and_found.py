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


class LostAndFound(Dataset):

    def __init__(self, hparams, transforms, mode=['test','train']):
        super().__init__()

        self.hparams = hparams
        self.transforms = transforms
        self.mode = mode
        self.images = []
        self.labels = []

        hparams.dataset_root = '/shared-local/DATASETS/Video_Anomaly_Segmentation/LostAndFound/'


        for mode in ['test', 'train']:            
            for root, _, filenames in os.walk(os.path.join(hparams.dataset_root, 'Labels', mode)):
                for filename in sorted(filenames):
                    if os.path.splitext(filename)[1] == '.png':
                        filename_base = '_'.join(filename.split('_')[:-2])
                        city = '_'.join(filename.split('_')[:-4])
                        #print(city, filename_base)
                        self.labels.append(os.path.join(root, filename_base + '_gtCoarse_labelIds.png'))
                        target_root = os.path.join(hparams.dataset_root, 'Images', mode)
                        self.images.append(os.path.join(target_root, city, filename_base + '_leftImg8bit.png'))
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
