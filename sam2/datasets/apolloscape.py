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

class Apolloscape(Dataset):

    def __init__(self, hparams, transforms, mode=['road01','road02','road03', 'road04']):
        super().__init__()

        self.hparams = hparams
        self.transforms = transforms
        self.mode = mode
        self.images = []
        self.labels = []
        
        hparams.dataset_root = '/shared-local/DATASETS/Video_Anomaly_Segmentation/ApolloScapes_anomaly/'
        for mode in ['road01','road02','road03', 'road04']:
            for dirname in os.listdir(os.path.join(hparams.dataset_root, 'Labels', mode, 'Label')):
                for camera_number in ['Camera 5', 'Camera 6']:
                    if len(sorted(os.listdir(os.path.join(hparams.dataset_root, 'Labels', mode, 'Label', dirname, camera_number)))) > 25:
                        for filename in sorted(os.listdir(os.path.join(hparams.dataset_root, 'Labels', mode, 'Label', dirname, camera_number))):
                            label_path = os.path.join(hparams.dataset_root, 'Labels', mode, 'Label', dirname, camera_number, filename)
                            image_path = label_path.replace('.png', '.jpg') 
                            image_path = image_path.replace('Labels', 'Images')   
                            image_path = image_path.replace('Label/', '')
                            image_path = image_path.replace('_bin', '')                            
                            if os.path.exists(image_path):
                                self.labels.append(label_path)
                                self.images.append(image_path)
                    
        self.num_samples = len(self.images)
        print(self.num_samples)

    def __getitem__(self, index):
        image = np.array(Image.open(self.images[index]).resize((1355, 1692), Image.BILINEAR), np.float16)      
        label = np.array(Image.open(self.labels[index]).resize((1355, 1692), Image.NEAREST))
        label = np.where((label==0), 255, label)          # Background
        label = np.where((label==49), 0, label)           # Road 
        label = np.where((label>1)&(label<250), 1, label) # Anomalies
        aug = self.transforms(image=image, mask=label)
        image = aug['image']
        label = aug['mask']

        return image, label.type(torch.LongTensor), self.images[index]

    def __len__(self):

        return self.num_samples
