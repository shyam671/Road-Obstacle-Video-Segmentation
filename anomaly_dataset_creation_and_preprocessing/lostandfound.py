from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import filter_AS
from multiprocessing import Pool
cm = plt.get_cmap('hot')

path = '/shared-local/DATASETS/Video_Anomaly_Segmentation/LostAndFound/'
iters = 0
iters_total = 0

for label_path in Path(path).glob('gtCoarse/**/**/*_labelIds.png'):

    iters_total = iters_total + 1
  
    label = np.array(Image.open(label_path))

    label[label == 255] = 0      # Void   
    
    label_sum = label.copy()
    label_sum[label_sum==1] = 0 
    label_sum = np.where((label_sum>1)&(label_sum<201), 1, label_sum)

    save_path_file = str(label_path).split('/')[:-1]
    save_path_file = '/'.join(save_path_file)    
    save_path_file = save_path_file.replace('/shared-local/DATASETS/Video_Anomaly_Segmentation/LostAndFound/','../anomaly_maps/LostAndFound/')
    Path(save_path_file).mkdir(parents=True, exist_ok=True)
    

    # If the anomaly sum is < 100 then change it into void.
    if np.sum(label_sum) < 225:
        label[label > 1] = 0
        iters = iters + 1


    label = Image.fromarray(np.uint8(label))
    label.save(save_path_file +'/'+ str(label_path).split('/')[-1])
    print(iters_total, iters)