from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import filter_AS
from multiprocessing import Pool


path = '/shared-local/DATASETS/Video_Anomaly_Segmentation/ApolloScape/road04/Label/'
#paths = list(Path(path).glob('**/*.png'))
#text_file_path = 'anomaly_maps/road01/anomaly_label.txt'
anomaly_labels = [65, 66, 40, 168, 85]

label_path_list = []
iters = 0
iters_total = 0
for label_path in Path(path).glob('**/**/*.png'):
    if '_instanceIds' not in str(label_path):
        #if '_bin' not in str(label_path):
            if iters == 128:
                p = Pool(128)
                p.map(filter_AS, label_path_list)
                label_path_list = []
                iters_total = iters_total + iters
                iters = 0
            else:
                label_path_list.append(str(label_path))
                iters = iters + 1
    print(iters_total)

