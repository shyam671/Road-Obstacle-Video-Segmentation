import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import random

from albumentations.pytorch import ToTensorV2
from easydict import EasyDict as edict

from datasets.lidarSOD import LidarSOD
from datasets.sos import SOS
from datasets.apolloscape import Apolloscape

from datasets.cityscapes import Cityscapes

from datasets.bdd100k import BDD100KSeg
from datasets.road_anomaly import RoadAnomaly
from datasets.fishyscapes import FishyscapesLAF, FishyscapesStatic
from datasets.segment_me_if_you_can import RoadAnomaly21, RoadObstacle21
from datasets.lost_and_found import LostAndFound
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image, ImageOps
from scipy.ndimage.measurements import label
from easydict import EasyDict
import scipy.stats as st
from typing import Callable
from sklearn.metrics import roc_curve, auc, average_precision_score
from ood_metrics import fpr_at_95_tpr
#from my_ood_plots import plot_roc, plot_barcode, plot_pr
import cv2 as cv2
cm = plt.get_cmap('gist_heat')
from PIL import Image
import matplotlib.pyplot as plt
#import numba as nb
import random
import glob
from PIL import Image
from matplotlib import cm
# @nb.njit('float64[:,:](int_, int_)', parallel=True)
# def genRandom(n, m):
#     res = np.empty((n, m))

#     # Parallel loop
#     for i in nb.prange(n):
#         for j in range(m):
#             res[i, j] = np.random.rand()

#     return res

def aggregate(frame_results: list):

    iou_thresholds = np.linspace(0.25, 0.75, 11, endpoint=True)
    sIoU_gt_mean = sum(np.sum(r.sIoU_gt) for r in frame_results) / sum(len(r.sIoU_gt) for r in frame_results)
    sIoU_pred_mean = sum(np.sum(r.sIoU_pred) for r in frame_results) / sum(len(r.sIoU_pred) for r in frame_results)
    prec_pred_mean = sum(np.sum(r.prec_pred) for r in frame_results) / sum(len(r.prec_pred) for r in frame_results)
    ag_results = {"tp_mean" : 0., "fn_mean" : 0., "fp_mean" : 0., "f1_mean" : 0.,
                  "sIoU_gt" : sIoU_gt_mean, "sIoU_pred" : sIoU_pred_mean, "prec_pred": prec_pred_mean}
    print("sIoU GT   :", "{:.2f}".format(sIoU_gt_mean*100))
    # print("Mean sIoU PRED :", sIoU_pred_mean)
    print("PPV :", "{:.2f}".format(prec_pred_mean*100))
    for t in iou_thresholds:
        tp = sum(r["tp_" + str(int(t*100))] for r in frame_results)
        fn = sum(r["fn_" + str(int(t*100))] for r in frame_results)
        fp = sum(r["fp_" + str(int(t*100))] for r in frame_results)
        f1 = (2 * tp) / (2 * tp + fn + fp)
        if t in [0.25, 0.50, 0.75]:
            ag_results["tp_" + str(int(t * 100))] = tp
            ag_results["fn_" + str(int(t * 100))] = fn
            ag_results["fp_" + str(int(t * 100))] = fp
            ag_results["f1_" + str(int(t * 100))] = f1

        ag_results["tp_mean"] += tp
        ag_results["fn_mean"] += fn
        ag_results["fp_mean"] += fp
        ag_results["f1_mean"] += f1

    ag_results["tp_mean"] /= len(iou_thresholds)
    ag_results["fn_mean"] /= len(iou_thresholds)
    ag_results["fp_mean"] /= len(iou_thresholds)
    ag_results["f1_mean"] /= len(iou_thresholds)
    
    print("F1 score       :", "{:.2f}".format(ag_results["f1_mean"]*100))
    return sIoU_gt_mean, prec_pred_mean, ag_results["f1_mean"]

def default_instancer(anomaly_p: np.ndarray, label_pixel_gt: np.ndarray, thresh_p: float,
                      thresh_segsize: int, thresh_instsize: int = 0):

    """segmentation from pixel-wise anoamly scores"""
    segmentation = np.copy(anomaly_p)
    segmentation[anomaly_p > thresh_p] = 1
    segmentation[anomaly_p <= thresh_p] = 0

    anomaly_gt = np.zeros(label_pixel_gt.shape)
    anomaly_gt[label_pixel_gt == 1] = 1
    anomaly_pred = np.zeros(label_pixel_gt.shape)
    anomaly_pred[segmentation == 1] = 1
    anomaly_pred[label_pixel_gt == 255] = 0

    """connected components"""
    structure = np.ones((3, 3), dtype=np.int64)
    anomaly_instances, n_anomaly = label(anomaly_gt, structure)
    anomaly_seg_pred, n_seg_pred = label(anomaly_pred, structure)

    """remove connected components below size threshold"""
    if thresh_segsize is not None:
        minimum_cc_sum  = thresh_segsize
        labeled_mask = np.copy(anomaly_seg_pred)
        for comp in range(n_seg_pred+1):
            if len(anomaly_seg_pred[labeled_mask == comp]) < minimum_cc_sum:
                anomaly_seg_pred[labeled_mask == comp] = 0
    labeled_mask = np.copy(anomaly_instances)
    label_pixel_gt = label_pixel_gt.copy() # copy for editing
    for comp in range(n_anomaly + 1):
        if len(anomaly_instances[labeled_mask == comp]) < thresh_instsize:
            label_pixel_gt[labeled_mask == comp] = 255

    """restrict to region of interest"""
    mask_roi = label_pixel_gt < 255
    segmentation_filtered = np.copy(anomaly_seg_pred).astype("uint8")
    segmentation_filtered[anomaly_seg_pred>0] = 1
    segmentation_filtered[mask_roi==255] = 0

    return anomaly_instances[mask_roi], anomaly_seg_pred[mask_roi], segmentation_filtered

def segment_metrics(anomaly_instances, anomaly_seg_pred, iou_thresholds=np.linspace(0.25, 0.75, 11, endpoint=True)):
    """
    function that computes the segments metrics based on the adjusted IoU
    anomaly_instances: (numpy array) anomaly instance annoation
    anomaly_seg_pred: (numpy array) anomaly instance prediction
    iou_threshold: (float) threshold for true positive
    """

    """Loop over ground truth instances"""
    sIoU_gt = []
    size_gt = []

    for i in np.unique(anomaly_instances[anomaly_instances>0]):
        tp_loc = anomaly_seg_pred[anomaly_instances == i]
        seg_ind = np.unique(tp_loc[tp_loc != 0])

        """calc area of intersection"""
        intersection = len(tp_loc[np.isin(tp_loc, seg_ind)])
        adjustment = len(
            anomaly_seg_pred[np.logical_and(~np.isin(anomaly_instances, [0, i]), np.isin(anomaly_seg_pred, seg_ind))])

        adjusted_union = np.sum(np.isin(anomaly_seg_pred, seg_ind)) + np.sum(
            anomaly_instances == i) - intersection - adjustment
        sIoU_gt.append(intersection / adjusted_union)
        size_gt.append(np.sum(anomaly_instances == i))

    """Loop over prediction instances"""
    sIoU_pred = []
    size_pred = []
    prec_pred = []
    for i in np.unique(anomaly_seg_pred[anomaly_seg_pred>0]):
        tp_loc = anomaly_instances[anomaly_seg_pred == i]
        seg_ind = np.unique(tp_loc[tp_loc != 0])
        intersection = len(tp_loc[np.isin(tp_loc, seg_ind)])
        adjustment = len(
            anomaly_instances[np.logical_and(~np.isin(anomaly_seg_pred, [0, i]), np.isin(anomaly_instances, seg_ind))])
        adjusted_union = np.sum(np.isin(anomaly_instances, seg_ind)) + np.sum(
            anomaly_seg_pred == i) - intersection - adjustment
        sIoU_pred.append(intersection / adjusted_union)
        size_pred.append(np.sum(anomaly_seg_pred == i))
        prec_pred.append(intersection / np.sum(anomaly_seg_pred == i))
        #print('here',intersection, adjustment, adjusted_union, intersection / adjusted_union, np.sum(anomaly_seg_pred == i))

    sIoU_gt = np.array(sIoU_gt)
    sIoU_pred = np.array(sIoU_pred)
    size_gt = np.array((size_gt))
    size_pred = np.array(size_pred)
    prec_pred = np.array(prec_pred)

    """create results dictionary"""
    results = EasyDict(sIoU_gt=sIoU_gt, sIoU_pred=sIoU_pred, size_gt=size_gt, size_pred=size_pred, prec_pred=prec_pred)
    for t in iou_thresholds:
        results["tp_" + str(int(t*100))] = np.count_nonzero(sIoU_gt >= t)
        results["fn_" + str(int(t*100))] = np.count_nonzero(sIoU_gt < t)
        # results["fp_" + str(int(t*100))] = np.count_nonzero(sIoU_pred < t)
        results["fp_" + str(int(t*100))] = np.count_nonzero(prec_pred < t)

    return results

def process_frame(label_pixel_gt: np.ndarray, anomaly_p: np.ndarray, fid : str=None, dset_name : str=None,
                  method_name : str=None, visualize : bool = False, **_):
    """
    @param label_pixel_gt: HxW uint8
        0 = in-distribution / road
        1 = anomaly / obstacle
        255 = void / ignore
    @param anomaly_p: HxW float16
        heatmap of per-pixel anomaly detection, higher values correspond to anomaly / obstacle class
    @param visualize: bool
        saves an image with segment predictions
    """

    anomaly_gt, anomaly_pred, mask = default_instancer(anomaly_p, label_pixel_gt, 0.5, 50, 10)  ############### change here

    results = segment_metrics(anomaly_gt, anomaly_pred, np.linspace(0.25, 0.75, 11, endpoint=True))

    return results

def get_datasets(datasets_folder):

    laf_config = edict(
        dataset_root=os.path.join(datasets_folder, 'LostAndFound'),

    )

    lidarSOD_config = edict(
        dataset_root=os.path.join(datasets_folder, 'LidarSOD'),

    )

    sos_config = edict(
        dataset_root=os.path.join(datasets_folder, 'SOS'),

    )

    # apolloscape_config = edict(
    #     dataset_root=os.path.join(datasets_folder, 'SOS'),

    # )

    transform = A.Compose([
        ToTensorV2()
    ])
    DATASETS = edict(lost_and_found=LostAndFound(laf_config, transform), lidarSOD=LidarSOD(lidarSOD_config, transform), sos=SOS(sos_config, transform))
    #Uncomment to use apolloscape
    #DATASETS = edict(lost_and_found=LostAndFound(laf_config, transform), lidarSOD=LidarSOD(lidarSOD_config, transform), sos=SOS(sos_config, transform), apolloscape=Apolloscape(apolloscape_config, transform))
    return DATASETS

def get_logits_plus(model, x, **kwargs):
    
    with torch.no_grad():
        out = model([{"image": x[0].cuda()}], **kwargs)
    
    if "return_aux" in kwargs and kwargs["return_aux"]:
        return out[0][0]["sem_seg"].unsqueeze(0), out[1]

    return out[0]['sem_seg'].unsqueeze(0)

def get_logits(model, x, **kwargs):
    
    with torch.no_grad():
        out = model([{"image": x[0].cuda()}])
    
    return out[0]['sem_seg'].unsqueeze(0)

def get_common(list_,predlist,clip_num,h,w):
    accs = []
    accs_a = []
    accs_b = []
    for i in range(len(list_)-clip_num):

        global_common = np.ones((h,w))
        global_n_common = np.zeros((h,w))

        predglobal_common = np.ones((h,w))  
        predglobal_n_common = np.zeros((h,w))

        for j in range(1,clip_num):
            
            common = (list_[i] == list_[i+j])
            common_n = (np.abs(list_[i]-1) == np.abs(list_[i+j]-1)) # Flip the labels 0--> 1 and 1 --> 0
            
            global_common = np.logical_and(global_common,common)
            global_n_common = np.logical_or(global_n_common,common) # Nor operator 

            pred_common = (predlist[i]==predlist[i+j])
            pred_n_common = (np.abs(predlist[i]-1) == np.abs(predlist[i+j]-1)) # Flip the labels 0--> 1 and 1 --> 0

            predglobal_common = np.logical_and(predglobal_common,pred_common)
            predglobal_n_common = np.logical_or(predglobal_n_common,pred_n_common) # Nor operator 

            # print('common', common.sum())
            # print('common_n', common_n.sum())
            
            # print('pred_common', pred_common.sum())
            # print('pred_n_common', pred_n_common.sum())

            # print('global_common', global_common.sum())
            # print('global_n_common', global_n_common.sum())

            # print('predglobal_common', predglobal_common.sum())
            # print('predglobal_n_common', predglobal_n_common.sum())

        pred = (predglobal_common*global_common)
        n_pred = (np.logical_not(predglobal_n_common)*np.logical_not(global_n_common))
        
        a_acc = pred.sum()/global_common.sum() # Anomaly accuracy
        b_acc = n_pred.sum()/np.logical_not(global_n_common).sum() # Background accuracy
        
        if np.isnan(a_acc):
            a_acc = 1e-6
            
        if np.isnan(b_acc):
            b_acc = 1e-6

        acc = (a_acc + b_acc)/2 
        
        #print(a_acc, b_acc, acc)
        
        accs.append(acc)
        accs_a.append(a_acc)
        accs_b.append(b_acc)
    accs_a = sum(accs_a)/len(accs_a)
    accs_b = sum(accs_b)/len(accs_b)
    accs = sum(accs)/len(accs)
    print(accs, accs_a, accs_b)
    return 2*(accs_a*accs_b)/(accs_a+accs_b)

def get_neg_logit_sum(model, x, **kwargs):
    """
    This function computes the negative logits sum of a given logits map as an anomaly score.

    Expected input:
    - model: detectron2 style pytorch model
    - x: image of shape (1, 3, H, W)

    Expected Output:
    - neg_logit_sum (torch.Tensor) of shape (H, W)
    """

    with torch.no_grad():
        out = model([{"image": x[0].cuda()}])

    logits = out[0]['sem_seg']
    
    return -logits.sum(dim=0)

def get_RbA(model, x, **kwargs):
    
    with torch.no_grad():
        out = model([{"image": x[0].cuda()}])

    logits = out[0]['sem_seg']
    
    return -logits.tanh().sum(dim=0)


def logistic(x, k=1, x0=0, L=1):
    
    return L/(1 + torch.exp(-k*(x-x0)))

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def show_anns(anns, strength=0.35):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*strength)))

def save_outputs(input_, output_, gt_, path):
    
    output_[gt_==255] = 0
    gt_[gt_==255] = 0
    if gt_.sum() > 5000:
        gt_[gt_==1] = 255
        input_ = Image.open(path[0]).convert('RGBA')
        gt_ = cm(gt_)
        gt_ = Image.fromarray(np.uint8(gt_[:, :, :3]*255)).convert('RGBA')  
        gt_ = Image.blend(input_, gt_, 0.75)
        output_ = cm(output_)
        output_ = Image.fromarray(np.uint8(output_[:, :, :3]*255)).convert('RGBA')  
        output_ = Image.blend(input_, output_, 0.75)
        
        image_name = '_'.join(path[0].split('/'))
        dir_name = 'none'
        if 'LostAndFound' in path[0]:
            dir_name = 'lf'
        elif 'LidarSOD' in path[0]:
            dir_name = 'lidersod'
        else:
            dir_name = 'sos'
        
        file_path = os.path.join('output/output/', dir_name, 'gt_' + image_name) 
        file_path = file_path.replace('.jpeg', '.png')
        file_path = file_path.replace('.jpg', '.png')
        gt_.save(file_path)

        file_path = os.path.join('output/output/', dir_name, 'output_' + image_name) 
        file_path = file_path.replace('.jpeg', '.png')
        file_path = file_path.replace('.jpg', '.png')
        output_.save(file_path)

        return 

def get_seg_colormap(preds, colors):
    """
    Assuming preds.shape = (H,W)
    """
    H, W = preds.shape
    color_map = torch.zeros((H, W, 3)).long()
    
    for i in range(len(colors)):
        mask = (preds == i)
        if mask.sum() == 0:
            continue
        color_map[mask, :] = torch.tensor(colors[i])
    
    return color_map

def proc_img(img):
    
    if isinstance(img, torch.Tensor):
        ready_img = img.clone()
        if len(ready_img.shape) == 3 and ready_img.shape[0] == 3:
            ready_img = ready_img.permute(1, 2, 0)
        ready_img = ready_img.cpu()

    elif isinstance(img, np.ndarray):
        ready_img = img.copy()
        if len(ready_img.shape) == 3 and ready_img.shape[0] == 3:
            ready_img = ready_img.transpose(1, 2, 0)
    else:
        raise ValueError(
            f"Unsupported type for image: ({type(img)}), only supports numpy arrays and Pytorch Tensors")

    return ready_img

def resize_mask(m, shape):
    
    m = F.interpolate(
        m,
        size=(shape[0], shape[1]),
        mode="bilinear",
        align_corners=False,
    )
    return m

class OODEvaluator:

    def __init__(
        self,
        model: nn.Module,
        inference_func: Callable,
        anomaly_score_func: Callable,
    ):

        self.model = model
        self.inference_func = inference_func
        self.anomaly_score_func = anomaly_score_func

    def get_logits(self, x, **kwargs):
        return self.inference_func(self.model, x, **kwargs)

    def get_anomaly_score(self, x, **kwargs):
        return self.anomaly_score_func(self.model, x, **kwargs)

    def calculate_auroc(self, conf, gt):
        fpr, tpr, threshold = roc_curve(gt, conf)
        roc_auc = auc(fpr, tpr)
        fpr_best = 0
        # print('Started FPR search.')
        for i, j, k in zip(tpr, fpr, threshold):
            if i > 0.95:
                fpr_best = j
                break
        # print(k)
        return roc_auc, fpr_best, k

    def calculate_ood_metrics(self, out, label, save_path):

        fpr, tpr, _ = roc_curve(label, out)
        
        if save_path is not None:
            auroc_plot = plot_roc(out, label, save_path)
            auprc_plot = plot_pr(out, label, save_path)
            plot_barcodee = plot_barcode(out, label, save_path) 
        

        roc_auc = auc(fpr, tpr)
        prc_auc = average_precision_score(label, out)
        fpr = fpr_at_95_tpr(out, label)


        # prc_auc = average_precision_score(label, out)
        # roc_auc, fpr, _ = self.calculate_auroc(out, label)
        # roc_auc = auc(fpr, tpr)
        # fpr = fpr_at_95_tpr(out, label)

        return roc_auc, prc_auc, fpr

    def save_anomaly_images(
        self,
        loader,
        device=torch.device('cpu'),
        return_preds=False,
        use_gaussian_smoothing=False,
        upper_limit=450
    ):

        jj = 0

        main_path = '/shared-network/srai/VAS/saved-outputs/LiderSOD/'

        for x, y, path in tqdm(loader, desc="Dataset Iteration"):
            
            if jj >= upper_limit:
                break
            jj += 1
            x = x.to(device)
            y = y.to(device)
            y = y.squeeze().cpu().numpy().astype('float16')  

            if np.sum(y==1)>400:
                score = self.model.forward_vps(x.squeeze(0).permute(1, 2, 0).cpu().numpy())
                score = F.interpolate(score, size=(y.shape[0], y.shape[1]), mode='bilinear', align_corners=False)
                score = score[:,:19,:,:]
                score, _ = torch.max(score, dim=1)
                score = -score.squeeze(0).cpu().numpy()
                score = (score - score.min())/(score.max() -  score.min())
                score[y==255] = 0
                y[y==255] = 0 
                y[y==1] = 255
                im1 = Image.fromarray(np.uint8(cm.hot(score)*255)).convert('RGB')
                #im1 = ImageOps.expand(im1,border=10,fill='white')
                im2 = Image.fromarray(np.uint8(y)).convert('RGB')
                #im2 = ImageOps.expand(im2,border=10,fill='white')
                im3 = Image.open(path[0]).convert('RGB')
                #im3 = ImageOps.expand(im3,border=10,fill='white')
                im3.save(main_path + 'InpImg/' +'_'.join(path[0].split('/')[-2:]))
                im2.save(main_path + 'GT/' +'_'.join(path[0].split('/')[-2:]))
                im1.save(main_path + 'CA-SAM2/' +'_'.join(path[0].split('/')[-2:]))
                

                
                

            # score = self.model.forward_vps(image)
            # score = F.interpolate(score, size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False)
            # score = score[:,:19,:,:]
            # score, _ = torch.max(score, dim=1)
            # score = -score.squeeze(0).cpu().numpy()

            # images = [im3, im1, im2]
            # widths, heights = zip(*(i.size for i in images))

            # total_width = sum(widths)
            # max_height = max(heights)
            # new_im = Image.new('RGB', (total_width+100, max_height))

            # x_offset = 0
            # for im in images:
            #   new_im.paste(im, (x_offset + 20,0))
            #   x_offset += im.size[0]
            #new_im.save('output/saved_anomaly_images/' + str(path[0].split('/')[-1]))
            

        return

    def evaluate_ood(self, anomaly_score, ood_gts, save_path, verbose=True):

        ood_gts = ood_gts.squeeze()
        anomaly_score = anomaly_score.squeeze()
        
        ood_mask = (ood_gts == 1)
        ind_mask = (ood_gts == 0)

        ood_out = anomaly_score[ood_mask]
        ind_out = anomaly_score[ind_mask]
        del anomaly_score
        
        ood_label = np.ones(len(ood_out),dtype=np.float16)
        ind_label = np.zeros(len(ind_out),dtype=np.float16)

        val_out = np.concatenate((ind_out, ood_out)).astype('float16')
        del ind_out, ood_out
        val_label = np.concatenate((ind_label, ood_label)).astype('float16')
        del ind_label, ood_label

        if verbose:
            print(f"Calculating Metrics for {len(val_out)} Points ...")
        auroc, aupr, fpr = self.calculate_ood_metrics(val_out, val_label, save_path)

        if verbose:
            print(f'Max Logits: AUROC score: {auroc}')
            print(f'Max Logits: AUPRC score: {aupr}')
            print(f'Max Logits: FPR@TPR95: {fpr}')

        result = [auroc*100, aupr*100, fpr*100]

        return result

    def evaluate_ood_bootstrapped(
        self,
        dataset,
        ratio,
        trials,
        device=torch.device('cpu'),
        batch_size=1,
        num_workers=10,
    ):
        results = edict()

        dataset_size = len(dataset)
        sample_size = int(dataset_size * ratio)

        for i in range(trials):

            indices = np.random.choice(
                np.arange(dataset_size), sample_size, replace=False)
            loader = DataLoader(Subset(dataset, indices),
                                batch_size=batch_size, num_workers=num_workers)

            anomaly_score, ood_gts = self.compute_anomaly_scores(
                loader=loader,
                device=device,
                return_preds=False
            )

            metrics = self.evaluate_ood(
                anomaly_score=anomaly_score,
                ood_gts=ood_gts,
                verbose=False
            )

            for k, v in metrics.items():
                if k not in results:
                    results[k] = []
                results[k].extend([v])

        means = edict()
        stds = edict()
        for k, v in results.items():

            values = np.array(v)
            means[k] = values.mean() * 100.0
            stds[k] = values.std() * 100.0

        return means, stds



    def save_anomaly_scores(
        self,
        loader,
        device=torch.device('cpu'),
        return_preds=False,
        use_gaussian_smoothing=False,
        upper_limit=450
    ):

        jj = 0
            
        for x, y, path in tqdm(loader, desc="Dataset Iteration"):
            
            if jj >= upper_limit:
                break
            jj += 1

            x = x.to(device)
            y = y.to(device)

            score = self.get_anomaly_score(x)  
            score = score.cpu().numpy()         
            #score =  (score - score.min())/(score.max() -  score.min())
            y = y.squeeze().cpu().numpy() 

            save_path = os.path.join('output/LostAndFound_m2a/','_'.join(path[0].split('/')[6:]))
            save_path = save_path.replace('.jpg', '.npy')
            data = {'output': score, 'gt':y}
            np.save(save_path, data) 

        return

    def compute_and_save_anomaly_scores_from_single_images(
        self,
        loader,
        device=torch.device('cpu'),
        return_preds=False,
        use_gaussian_smoothing=False,
        upper_limit=450
    ):

        jj = 0
            
        for x, y, path in tqdm(loader, desc="Dataset Iteration"):
            component_level_results = []
            if jj >= upper_limit:
                break
            jj += 1

            x = x.to(device)
            y = y.to(device)

            score = self.get_anomaly_score(x)  
            score = score.cpu().numpy()         
            #score =  (score - score.min())/(score.max() -  score.min())
            y = y.squeeze().cpu().numpy() 
            AuROC, AuPRC, FPR95 = self.evaluate_ood(score, y, save_path=None)
            
            
            if AuPRC>20.0:
                component_level_results.append(process_frame(np.array(y), np.array(score)))
                sIoU, PPV, F1 = aggregate(component_level_results)
                save_path = 'AuROC_' + str(round(AuROC,2)) + '_AuPRC_' + str(round(AuPRC,2)) + '_FPR95_' + str(round(FPR95,2)) + '_sIoU_' + str(round(sIoU, 2)*100) + '_PPV_' + str(round(PPV, 2)*100) + '_F1_' + str(round(F1, 2)*100)
                image_name = str(save_path+path[0].split('/')[-1])
                if not os.path.exists('output/final_images/'+image_name):
                    os.mkdir('output/final_images/'+image_name) 

                AuROC, AuPRC, FPR95 = self.evaluate_ood(score, y, 'output/final_images/'+ str(save_path+path[0].split('/')[-1]))
                print(str(save_path+path[0].split('/')[-1]))
                score[y==255] = 0
                y[y==255] = 0 
                y[y==1] = 255

                im1 = Image.fromarray(np.uint8(cm.hot(score)*255)).convert('RGB')
                im1 = ImageOps.expand(im1,border=10,fill='white')
                im2 = Image.fromarray(np.uint8(y)).convert('RGB')
                im2 = ImageOps.expand(im2,border=10,fill='white')


                images = [im1, im2]
                widths, heights = zip(*(i.size for i in images))

                total_width = sum(widths)
                max_height = max(heights)
                new_im = Image.new('RGB', (total_width+100, max_height))

                # auroc_plot.savefig('output/final_images/'+ image_name +'/'+'auroc_plot.png')
                # auprc_plot.savefig('output/final_images/'+ image_name +'/'+'auprc_plot.png')
                # plot_barcode.savefig('output/final_images/'+ image_name +'/'+'plot_barcode.png')

                x_offset = 0
                for im in images:
                  new_im.paste(im, (x_offset + 40,0))
                  x_offset += im.size[0]
                new_im.save('output/final_images/'+ image_name +'/'+'prediction_gt.png')
                
        return

    def compute_anomaly_scores_from_images(self):

        anomaly_score = []
        ood_gts = []
        predictions = []
        component_level_results = []
        ood_gts_single_list = []
        score_single_list = []
        ood_gts_single_list_da = []
        score_single_list_da = []
        counter = 0
        for i in sorted(glob.glob("/shared-local/srai/VAS/code/IAS-Baselines/RbA/output/LostAndFound/*")):
            
            data = np.load(i, allow_pickle=True)
            score = data[()]['output']
            y = data[()]['gt']    
            anomaly_score.extend([score])
            ood_gts.extend([y])
            component_level_results.append(process_frame(np.array(y), np.array(score)))

            th_score = np.array(score).copy()
            th_score[score > 0.5] = 1
            th_score[score < 0.5] = 0
            y[y==255] = 0
            ood_gts_single_list.append(y)
            score_single_list.append(th_score)
            counter = counter + 1
            print(counter)


        print('this is tc', get_common(ood_gts_single_list, score_single_list, 7, th_score.shape[0], th_score.shape[1]))
        aggregate(component_level_results)

        ood_gts = np.array(ood_gts)
        anomaly_score = np.array(anomaly_score)

        return anomaly_score, ood_gts

    def compute_anomaly_scores_LandF(
        self,
        loader,
        device=torch.device('cpu'),
        return_preds=False,
        use_gaussian_smoothing=False,
        upper_limit=450
    ):

        anomaly_score = []
        ood_gts = []
        component_level_results = []
        ood_gts_single_list = []
        score_single_list = []  
        pbar = tqdm(total = 2421)

        dataset_root = '/shared-local/DATASETS/Video_Anomaly_Segmentation/LostAndFound/'
        for mode in ['test', 'train']:            
            for root, _, filenames in os.walk(os.path.join(dataset_root, 'Labels', mode)):
                for filename in sorted(filenames):
                    if os.path.splitext(filename)[1] == '.png':
                        
                        filename_base = '_'.join(filename.split('_')[:-2])
                        city = '_'.join(filename.split('_')[:-4])
                        label_path = os.path.join(root, filename_base + '_gtCoarse_labelIds.png')
                        
                        target_root = os.path.join(dataset_root, 'Images', mode)
                        image_path = os.path.join(target_root, city, filename_base + '_leftImg8bit.png')
                        image = np.array(Image.open(image_path).convert('RGB'))
                        label = np.asarray(Image.open(label_path))
                        label = np.where((label==0), 255, label)          # Background
                        label = np.where((label==1), 0, label)            # Road 
                        label = np.where((label>1)&(label<201), 1, label) # Anomalies
                        label = torch.from_numpy(label)
                        score = self.model.forward_vps(image)
                        score = F.interpolate(score, size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False)
                        score, _ = torch.max(score, dim=1)
                        score = -score.squeeze(0).cpu().numpy()
# ######################Saving
                        # score[label==255] = 0
                        # im1 = Image.fromarray(np.uint8(cm.hot(score)*255)).convert('RGB')
                        # im1.save('/shared-network/srai/VAS/saved-outputs/LostAndFound/CA-SAM2/'+image_path.split('/')[-2] + '_'+image_path.split('/')[-1])
                        #import pdb; pdb.set_trace()
# ############################

                        y = label.to(device)
                        ood_gts.extend([y.cpu().numpy()])
                        anomaly_score.extend([score])
                        y = y.squeeze().cpu().numpy()
                        component_level_results.append(process_frame(np.array(y), np.array(score)))
                        th_score = np.array(score).copy()
                        th_score =  (th_score - th_score.min())/(th_score.max() -  th_score.min())

                        th_score[th_score > 0.5] = 1
                        th_score[th_score < 0.5] = 0
                        
                        th_score[y==255] = 0
                        y[y==255] = 0
                        ood_gts_single_list.append(y)
                        score_single_list.append(th_score)
                        pbar.update(1)

        VC = get_common(ood_gts_single_list, score_single_list, 7, th_score.shape[0], th_score.shape[1])
        sIoU_gt, prec_pred, f1_mean = aggregate(component_level_results)
        ood_gts = np.array(ood_gts)
        anomaly_score = np.array(anomaly_score)
        px_l_ev = self.evaluate_ood(anomaly_score, ood_gts, save_path=None)
        print(px_l_ev[0], px_l_ev[1], px_l_ev[2], sIoU_gt*100, prec_pred*100, f1_mean*100, VC*100)
        return 


    def compute_anomaly_scores_SOS(
        self,
        loader,
        device=torch.device('cpu'),
        return_preds=False,
        use_gaussian_smoothing=False,
        upper_limit=450
    ):
        anomaly_score = []
        ood_gts = []
        component_level_results = []
        ood_gts_single_list = []
        score_single_list = []  
        pbar = tqdm(total = 1129)
        jj = 0

        dataset_root = '/shared-local/datasets/Video_Anomaly_Segmentation/SOS/'
        for dirname in os.listdir(os.path.join(dataset_root, 'Labels')):
            for filename in sorted(os.listdir(os.path.join(dataset_root, 'Labels',dirname))):
                if jj >= upper_limit:
                   break
                jj += 1

                label_path = os.path.join(dataset_root, 'Labels', dirname , filename)
                image_path = label_path.replace('Labels', 'Images')                   
                image_path = image_path.replace('_semantic_ood.png', '_raw_data.jpg')
                image = np.array(Image.open(image_path).convert('RGB'))
                label = np.asarray(Image.open(label_path))
                label = np.where((label==0), 255, label)          # Background
                label = np.where((label==1), 0, label)            # Road 
                label = np.where((label>1)&(label<201), 1, label) # Anomalies
                label = torch.from_numpy(label)
                #score = self.slide_inference(torch.from_numpy(image), 512, 512, 512, 512, 20)
                score = self.model.forward_vps(image)
                score = F.interpolate(score, size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False)
        
                # th_score, _ = torch.max(F.softmax(score, dim=1), dim=1)
                # th_score = 1.0 - th_score.squeeze(0).cpu().numpy()
                # th_score[th_score > 0.5] = 1
                # th_score[th_score < 0.5] = 0

                score, _ = torch.max(score, dim=1)
                score = -score.squeeze(0).cpu().numpy()
# ######################Saving
#                 score[label==255] = 0
#                 im1 = Image.fromarray(np.uint8(cm.hot(score)*255)).convert('RGB')
#                 im1.save('/shared-network/srai/VAS/saved-outputs/SOS/CA-SAM2/'+image_path.split('/')[-2] + '_'+image_path.split('/')[-1])
# ############################

                y = label.to(device)
                ood_gts.extend([y.cpu().numpy()])
                anomaly_score.extend([score])
                y = y.squeeze().cpu().numpy()
                component_level_results.append(process_frame(np.array(y), np.array(score)))
                
                th_score = np.array(score).copy()
                th_score =  (th_score - th_score.min())/(th_score.max() -  th_score.min())

                th_score[th_score > 0.5] = 1
                th_score[th_score < 0.5] = 0
                
                th_score[y==255] = 0
                y[y==255] = 0
                
                ood_gts_single_list.append(y)
                score_single_list.append(th_score)
                pbar.update(1)

        VC = get_common(ood_gts_single_list, score_single_list, 7, th_score.shape[0], th_score.shape[1])
        sIoU_gt, prec_pred, f1_mean = aggregate(component_level_results)
        ood_gts = np.array(ood_gts)
        anomaly_score = np.array(anomaly_score)
        px_l_ev = self.evaluate_ood(anomaly_score, ood_gts, save_path=None)
        print(px_l_ev[0], px_l_ev[1], px_l_ev[2], sIoU_gt*100, prec_pred*100, f1_mean*100, VC*100)
        return 


    def compute_anomaly_scores_SOD(
        self,
        loader,
        device=torch.device('cpu'),
        return_preds=False,
        use_gaussian_smoothing=False,
        upper_limit=450
    ):
        counter = 0.0
        anomaly_score = []
        ood_gts = []
        component_level_results = []
        ood_gts_single_list = []
        score_single_list = []  
        pbar = tqdm(total = 1900)

        dataset_root = '/shared-local/DATASETS/Video_Anomaly_Segmentation/LidarSOD/'
        for mode in ['test', 'train', 'val']:
            for dirname in os.listdir(os.path.join(dataset_root, 'Labels', mode)):  
                for filename in sorted(os.listdir(os.path.join(dataset_root, 'Labels', mode, dirname,'labels'))):

                    label_path = os.path.join(dataset_root, 'Labels', mode, dirname ,'labels', filename)
                    image_path = label_path.replace('Labels', 'Images')                   
                    image_path = image_path.replace('labels', 'image')
                    image = np.array(Image.open(image_path).convert('RGB'))

                    label = np.asarray(Image.open(label_path))
                    label = np.where((label==0), 255, label)          # Background
                    label = np.where((label==1), 0, label)            # Road 
                    label = np.where((label>1)&(label<201), 1, label) # Anomalies
                    label = torch.from_numpy(label)
                    score = self.model.forward_vps(image)
                    score = F.interpolate(score, size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False)
                    #score = F.softmax(score[:,:19,:,:], dim=1)
                    score = score[:,:19,:,:]
                    score, _ = torch.max(score, dim=1)
                    score = -score.squeeze(0).cpu().numpy()
                    y = label.to(device)
                    ood_gts.extend([y.cpu().numpy()])
                    anomaly_score.extend([score])
                    y = y.squeeze().cpu().numpy()
                    component_level_results.append(process_frame(np.array(y), np.array(score)))
                    th_score = np.array(score).copy()
                    th_score =  (th_score - th_score.min())/(th_score.max() -  th_score.min())

                    th_score[th_score > 0.5] = 1
                    th_score[th_score < 0.5] = 0
                    
                    th_score[y==255] = 0
                    y[y==255] = 0
                    ood_gts_single_list.append(y)
                    score_single_list.append(th_score)
                    pbar.update(1)

        VC = get_common(ood_gts_single_list, score_single_list, 7, th_score.shape[0], th_score.shape[1])
        sIoU_gt, prec_pred, f1_mean = aggregate(component_level_results)
        ood_gts = np.array(ood_gts)
        anomaly_score = np.array(anomaly_score)
        px_l_ev = self.evaluate_ood(anomaly_score, ood_gts, save_path=None)
        print(px_l_ev[0], px_l_ev[1], px_l_ev[2], sIoU_gt*100, prec_pred*100, f1_mean*100, VC*100)
        return 


    def compute_anomaly_scores_appolloscapes(
        self,
        loader,
        device=torch.device('cpu'),
        return_preds=False,
        use_gaussian_smoothing=False,
        upper_limit=450
    ):

        global_VC = 0.0
        global_sIoU_gt = 0.0
        global_prec_pred = 0.0
        global_f1_mean = 0.0
        counter = 0.0
        global_AuPRC = 0.0
        global_FPR95 = 0.0
        global_AuROC = 0.0
        transform = A.Compose([ToTensorV2()])

        dataset_root = '/shared-local/DATASETS/Video_Anomaly_Segmentation/ApolloScapes_anomaly/'
        for mode in ['road01','road02','road03', 'road04']:
            for dirname in os.listdir(os.path.join(dataset_root, 'Labels', mode, 'Label')):
                for camera_number in ['Camera 5', 'Camera 6']:
                    if len(sorted(os.listdir(os.path.join(dataset_root, 'Labels', mode, 'Label', dirname, camera_number)))) > 1000:
                        pbar = tqdm(total = len(sorted(os.listdir(os.path.join(dataset_root, 'Labels', mode, 'Label', dirname, camera_number)))))
                        anomaly_score = []
                        ood_gts = []
                        component_level_results = []
                        ood_gts_single_list = []
                        score_single_list = []
                        for filename in sorted(os.listdir(os.path.join(dataset_root, 'Labels', mode, 'Label', dirname, camera_number))): 

                            label_path = os.path.join(dataset_root, 'Labels', mode, 'Label', dirname, camera_number, filename)
                            
                            image_path = label_path.replace('.png', '.jpg') 
                            image_path = image_path.replace('Labels', 'Images')   
                            image_path = image_path.replace('Label/', '')
                            image_path = image_path.replace('_bin', '')
                            
                            if os.path.exists(image_path):

                                image = np.array(Image.open(image_path).resize((512, 1024), Image.BILINEAR))
                                label = np.array(Image.open(label_path).resize((512, 1024), Image.NEAREST))

                                # if len(np.unique(label))>2:
                                #     print(np.unique(label))
                                
                                label = np.where((label==0), 255, label)          
                                label = np.where((label==49), 0, label)             
                                label = np.where((label>1)&(label<250), 1, label) 
                                # image = torch.from_numpy(image.transpose(2, 0, 1))
                                label = torch.from_numpy(label)
                                # x = image.to(device)
                                y = label.to(device)

                  
                                ood_gts.extend([y.cpu().numpy()])
                                score = self.model.forward_vps(image)
                                score = F.interpolate(score, size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False)
                                score = score[:,:19,:,:]
                                score, _ = torch.max(score, dim=1)
                                score = -score.squeeze(0).cpu().numpy()
                                #score = genRandom(1024, 512)
                                anomaly_score.extend([score])
                                y = y.squeeze().cpu().numpy()
                                component_level_results.append(process_frame(np.array(y), np.array(score)))
                                # For VC
                                th_score = np.array(score).copy()
                                th_score =  (th_score - th_score.min())/(th_score.max() -  th_score.min())
                                th_score[th_score > 0.5] = 1
                                th_score[th_score < 0.5] = 0
                                
                                th_score[y==255] = 0
                                y[y==255] = 0

                                ood_gts_single_list.append(y)
                                score_single_list.append(th_score)
                                pbar.update(1)
                            
                        
                        VC = get_common(ood_gts_single_list, score_single_list, 7, th_score.shape[0], th_score.shape[1])
                        sIoU_gt, prec_pred, f1_mean = aggregate(component_level_results)

                        global_VC = VC + global_VC
                        global_sIoU_gt = sIoU_gt + global_sIoU_gt
                        global_prec_pred = prec_pred + global_prec_pred
                        global_f1_mean = f1_mean + global_f1_mean

                        ood_gts = np.array(ood_gts)
                        anomaly_score = np.array(anomaly_score)
                        px_l_ev = self.evaluate_ood(anomaly_score, ood_gts, save_path=None)
                        global_AuROC = px_l_ev[0] + global_AuROC
                        global_AuPRC = px_l_ev[1] + global_AuPRC
                        global_FPR95 = px_l_ev[2] + global_FPR95
                        counter = counter + 1
                        print(px_l_ev[0], px_l_ev[1], px_l_ev[2], sIoU_gt, prec_pred, f1_mean, VC*100, counter)                  
                        pbar.close()
                    
        print('printing global stats', global_VC/counter, global_sIoU_gt/counter, global_prec_pred/counter, global_f1_mean/counter, global_AuROC/counter, global_AuPRC/counter, global_FPR95/counter)
        return 



    # def compute_anomaly_scores_SOS_img_save(
    #     self,
    #     loader,
    #     device=torch.device('cpu'),
    #     return_preds=False,
    #     use_gaussian_smoothing=False,
    #     upper_limit=450
    # ):
    #     anomaly_score = []
    #     ood_gts = []
    #     component_level_results = []
    #     ood_gts_single_list = []
    #     score_single_list = []  
    #     pbar = tqdm(total = 1129)
    #     jj = 0

    #     dataset_root = '/shared-local/datasets/Video_Anomaly_Segmentation/SOS/'
    #     for dirname in os.listdir(os.path.join(dataset_root, 'Labels')):
    #         for filename in sorted(os.listdir(os.path.join(dataset_root, 'Labels',dirname))):
    #             if jj >= upper_limit:
    #                break
    #             jj += 1

    #             label_path = os.path.join(dataset_root, 'Labels', dirname , filename)
    #             image_path = label_path.replace('Labels', 'Images')                   
    #             image_path = image_path.replace('_semantic_ood.png', '_raw_data.jpg')
    #             image = np.array(Image.open(image_path).convert('RGB'))
    #             label = np.asarray(Image.open(label_path))
    #             label = np.where((label==0), 255, label)          # Background
    #             label = np.where((label==1), 0, label)            # Road 
    #             label = np.where((label>1)&(label<201), 1, label) # Anomalies
    #             label = torch.from_numpy(label)
    #             score, score_sam_d = self.model.forward_vps(image)                

    #             score = F.interpolate(score, size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False)
    #             score_sam_d = F.interpolate(score_sam_d, size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False)
                
    #             score, _ = torch.max(score, dim=1)
    #             score = -score.squeeze(0).squeeze(0).cpu().numpy()
    #             score_sam_d = -score_sam_d.squeeze(0).squeeze(0).cpu().numpy()
    #             label = label.squeeze(0).cpu().numpy()
    #             score[label==255] = 0
    #             score_sam_d[label==255] = 0
    #             label[label==255] = 0 
    #             label[label==1] = 255

    #             im1 = Image.fromarray(np.uint8(cm.hot(score)*255)).convert('RGB')
    #             im1 = ImageOps.expand(im1,border=10,fill='white')

    #             im2 = Image.fromarray(np.uint8(cm.hot(score_sam_d)*255)).convert('RGB')
    #             im2 = ImageOps.expand(im2,border=10,fill='white')

    #             im3 = Image.fromarray(np.uint8(cm.hot(label)*255)).convert('RGB')
    #             im3 = ImageOps.expand(im3,border=10,fill='white')

    #             images = [im1, im2, im3]
    #             widths, heights = zip(*(i.size for i in images))

    #             total_width = sum(widths)
    #             max_height = max(heights)
    #             new_im = Image.new('RGB', (total_width+100, max_height))

    #             x_offset = 0
    #             for im in images:
    #               new_im.paste(im, (x_offset + 40,0))
    #               x_offset += im.size[0]
    #             new_im.save('sam2_logs/temp_images/'+ filename +'_prediction_gt.png')              
                

    #             # y = label.to(device)
    #             # ood_gts.extend([y.cpu().numpy()])
    #             # anomaly_score.extend([score])
    #             # y = y.squeeze().cpu().numpy()
    #             # component_level_results.append(process_frame(np.array(y), np.array(score)))
    #             # th_score = np.array(score).copy()
    #             # th_score[score > 0.5] = 1
    #             # th_score[score < 0.5] = 0
    #             # y[y==255] = 0
    #             # ood_gts_single_list.append(y)
    #             # score_single_list.append(th_score)
    #             pbar.update(1)

    #     # VC = get_common(ood_gts_single_list, score_single_list, 16, th_score.shape[0], th_score.shape[1])
    #     # sIoU_gt, prec_pred, f1_mean = aggregate(component_level_results)
    #     # ood_gts = np.array(ood_gts)
    #     # anomaly_score = np.array(anomaly_score)
    #     # px_l_ev = self.evaluate_ood(anomaly_score, ood_gts, save_path=None)
    #     # print(px_l_ev[0], px_l_ev[1], px_l_ev[2], sIoU_gt, prec_pred, f1_mean, VC*100)
    #     return 


    def slide_inference(self, img, h_stride, w_stride, h_crop, w_crop, num_classes):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        img =  img.unsqueeze(0)
        img =  torch.permute(img, (0, 3, 1, 2))
        batch_size, _, h_img, w_img = img.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = torch.zeros((batch_size, num_classes, h_img, w_img))
        count_mat = torch.zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_img = crop_img.squeeze(0)
                crop_img = crop_img.permute(1,2,0)
                crop_seg_logit = self.model.forward_vps(crop_img.cpu().numpy())
                crop_seg_logit = F.interpolate(crop_seg_logit, size=(h_crop, w_crop), mode='bilinear', align_corners=False)
                preds += F.pad(crop_seg_logit.cpu(), (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        return preds

    # def compute_anomaly_scores(
    #     self,
    #     loader,
    #     device=torch.device('cpu'),
    #     return_preds=False,
    #     use_gaussian_smoothing=False,
    #     upper_limit=450
    # ):

    #     anomaly_score = []
    #     ood_gts = []
    #     predictions = []
    #     component_level_results = []
    #     ood_gts_single_list = []
    #     score_single_list = []
    #     ood_gts_single_list_da = []
    #     score_single_list_da = []
    #     jj = 0
    #     if use_gaussian_smoothing:
    #         gaussian_smoothing = transforms.GaussianBlur(7, sigma=1)

    #     for x, y, path in tqdm(loader, desc="Dataset Iteration"):
    #         if jj >= upper_limit:
    #             break
    #         jj += 1

    #         x = x.to(device)
    #         y = y.to(device)

    #         ood_gts.extend([y.cpu().numpy()])
    #         score = self.get_anomaly_score(x)  # -> (H, W)
    #         score = score.cpu().numpy()
    #         anomaly_score.extend([score])
            
    #         y = y.squeeze().cpu().numpy()
    #         #score = (score + 1.0)/2

    #         # save_outputs(x, score, y, path)
    #         component_level_results.append(process_frame(np.array(y), np.array(score)))

    #         # For VC
    #         th_score = np.array(score).copy()
    #         th_score[score > 0.5] = 1
    #         th_score[score < 0.5] = 0
    #         y[y==255] = 0
    #         ood_gts_single_list.append(y)
    #         score_single_list.append(th_score)
        

    #     self.evaluate_ood(np.array(anomaly_score), np.array(ood_gts), save_path='output/')
    #     VC = get_common(ood_gts_single_list, score_single_list, 7, th_score.shape[0], th_score.shape[1])
    #     print(VC)
    #     print('this is vc', VC)
    #     torch.cuda.empty_cache()
    #     aggregate(component_level_results)

    #     # ood_gts = np.array(ood_gts)
    #     # anomaly_score = np.array(anomaly_score)

    #     # if return_preds:
    #     #     predictions = np.array(predictions)
    #     #     return anomaly_score, ood_gts, predictions

    #     return anomaly_score, ood_gts
