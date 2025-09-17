from pathlib import Path
import cv2
from PIL import Image, ImageFile
import numpy as np
import matplotlib.pyplot as plt
import PIL

# 65=trafic_cone, 66=roadpile,  40= tricycle, 168 = tricycle group, 85 = dustbin
anomaly_labels = [65, 66, 40, 168, 85]
global_count_image = 0
ImageFile.LOAD_TRUNCATED_IMAGES = True

def filter_AS(label_path):
    try:
        label = np.array(Image.open(label_path))
        save_path_file = str(label_path).split('/')[:-1]
        save_path_file = '/'.join(save_path_file)
        save_path_file = save_path_file.replace('/shared-local/DATASETS/Video_Anomaly_Segmentation/ApolloScape/','../anomaly_maps/')
        Path(save_path_file).mkdir(parents=True, exist_ok=True)

        filtered_anomaly_label = np.zeros_like(label)
        filtered_anomaly_label = np.where((label==49), 49, filtered_anomaly_label) 
        filtered_anomaly_label = Image.fromarray(np.uint8(filtered_anomaly_label))
        filtered_anomaly_label.save(save_path_file +'/'+ str(label_path).split('/')[-1])

        if np.isin(anomaly_labels,np.unique(label)).any():
            anomaly_label = np.zeros_like(label)
            # Road Label
            road_label = np.zeros_like(label)
            road_label = np.where((label==49), 1, road_label)

            # Create anomaly label
            anomaly_label = np.where((label==65), 65, anomaly_label) 
            anomaly_label = np.where((label==66), 66, anomaly_label) 
            anomaly_label = np.where((label==40), 40, anomaly_label) 
            anomaly_label = np.where((label==168),168, anomaly_label) 
            anomaly_label = np.where((label==85), 85, anomaly_label)
            anomaly_label_th = np.copy(anomaly_label)
            anomaly_label_th[anomaly_label_th>0] = 255

            contours, _ = cv2.findContours(anomaly_label_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            hull = []
            for i in range(len(contours)):
                hull.append(cv2.convexHull(contours[i], False))
            
            filtered_anomaly_label = np.zeros_like(label)
            # draw contours and hull points
            for i in range(len(contours)):
                temp_anomaly_label = np.zeros_like(label)
                M=cv2.moments(contours[i])
                if M['m00'] !=0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    if cx<filtered_anomaly_label.shape[0] and cy<filtered_anomaly_label.shape[1]:
                        if road_label[cx,cy].astype(int)==1:
                            cv2.drawContours(temp_anomaly_label, hull, i, 1, -1, 8)
                            if np.sum(temp_anomaly_label.astype(int) & road_label.astype(int)) >=1.0:
                                cv2.drawContours(filtered_anomaly_label, hull, i, 1, -1, 8)


            if np.sum(filtered_anomaly_label)>=400: # atleat 1% 
                filtered_anomaly_label = np.where((filtered_anomaly_label==1), anomaly_label, filtered_anomaly_label) 
                filtered_anomaly_label = np.where((label==49), 49, filtered_anomaly_label) 
                filtered_anomaly_label = Image.fromarray(np.uint8(filtered_anomaly_label))
                filtered_anomaly_label.save(save_path_file +'/'+ str(label_path).split('/')[-1])
    except (PIL.UnidentifiedImageError, OSError, AttributeError):
        return