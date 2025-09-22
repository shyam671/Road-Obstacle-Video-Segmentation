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


#srai@jupiter:/shared-local/DATASETS/Video_Anomaly_Segmentation/LidarSOD/Images/train/file_1/image
#srai@jupiter:/shared-local/DATASETS/Video_Anomaly_Segmentation/LidarSOD/Labels/test/stadium_3/labels

class LidarSOD(Dataset):

    def __init__(self, hparams, transforms, mode=['test','train','val']):
        super().__init__()

        self.hparams = hparams
        self.transforms = transforms
        self.mode = mode
        self.images = []
        self.labels = []
        
        hparams.dataset_root = '/home/shyam/VAS/LidarSOD/'

        for mode in ['test', 'train', 'val']:
            for dirname in os.listdir(os.path.join(hparams.dataset_root, 'Labels', mode)):
                for filename in sorted(os.listdir(os.path.join(hparams.dataset_root, 'Labels', mode, dirname,'labels'))):
                    label_path = os.path.join(hparams.dataset_root, 'Labels', mode, dirname ,'labels', filename)
                    self.labels.append(label_path)
                    image_path = label_path.replace('Labels', 'Images')                   
                    image_path = image_path.replace('labels', 'image')
                    self.images.append(image_path)
        self.num_samples = len(self.images)

    def __getitem__(self, index):

        image = np.array(Image.open(self.images[index]).convert('RGB'))
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
