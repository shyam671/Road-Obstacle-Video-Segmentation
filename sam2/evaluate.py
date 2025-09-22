import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import pickle
import matplotlib.image as mpimg
import albumentations as A
from pathlib import Path
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.cityscapes import Cityscapes
from datasets.bdd100k import BDD100KSeg
from datasets.road_anomaly import RoadAnomaly
from datasets.fishyscapes import FishyscapesLAF, FishyscapesStatic
from datasets.lost_and_found import LostAndFound
from pprint import pprint
from support import get_datasets, OODEvaluator
from easydict import EasyDict as edict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

parser = argparse.ArgumentParser(description='OOD Evaluation')

parser.add_argument('--batch_size', type=int, default=1,
                    help="Batch Size used in evaluation")
parser.add_argument('--num_workers', type=int, default=4,
                    help="Number of threads used in data loader")
parser.add_argument('--device', type=str, default='cuda',
                    help="cpu or cuda, the device used for evaluation")
parser.add_argument('--out_path', type=str, default='results',
                    help='output file for saving the results as a pickel file')
parser.add_argument('--verbose', type=bool, default=True,
                    help="If True, the records will be printed every time they are saved")
parser.add_argument('--datasets_folder', type=str, default='./',
                    help='the path to the folder that contains all datasets for evaluation')
parser.add_argument('--models_folder', type=str, default='ckpts/',
                    help='the path that contains the models to be evaluated')
parser.add_argument("--store_anomaly_scores", action='store_true',
                    help="""If passed, store anomaly score maps that are extracted in full evaluation. 
                    The map will be stored in a folder for each model, and under it a folder for each dataset. 
                    All will be stored under anomaly_scores/ folder""")
parser.add_argument('--model_mode', type=str, default='all',
                    help="""One of [all, selective]. Defines which models to evaluate, the default behavior is all, which is to 
                            evaluate all models in model_logs dir. You can also choose particular models
                            for evaluation, in which case you need to pass the names of the models to --selected_models""")
parser.add_argument("--selected_models", nargs="*", type=str, default=[],
                    help="Names of models to be evaluated, these should be name of directories in model_logs")
parser.add_argument('--dataset_mode', type=str, default='all',
                    help="""One of [all, selective]. Defines which datasets to evaluate on, the default behavior is all, which is to 
                            evaluate all available datasets. You can also choose particular datasets
                            for evaluation, in which case you need to pass the names of the datasets to --selected_datasets.
                            Available Datasets are: [
                                road_anomaly,
                                fishyscapes_laf,
                            ]
                            """)
parser.add_argument("--selected_datasets", nargs="*", type=str, default=[],
                    help="""Names of datasets to be evaluated.
                        Available Datasets are: [
                        lidarSOD
                            ]
                    """)
parser.add_argument("--score_func", type=str, default="rba", choices=["rba", "energy", "entropy", "eam", "aem", "m2a", "msp", "ml", "void"],
                    help="outlier scoring function to be used in evaluations")

parser.add_argument("--config_pathh", type=str, default=[])    
parser.add_argument("--model_pathh", type=str, default=[])    

args = parser.parse_args()

DATASETS = get_datasets(args.datasets_folder)
dataset_group = [(name, dataset) for (name, dataset) in DATASETS.items() ]

# filter dataset group according to chosen option
if args.dataset_mode == 'selective':
    dataset_group = [g for g in dataset_group if g[0]
                     in args.selected_datasets]
    if len(dataset_group) == 0:
        raise ValueError(
            "Selective Mode is chosen but number of selected datasets is 0")
else:
    dataset_group = [g for g in dataset_group if g[0] in ['lidarSOD']]

print("Datasets to be evaluated:")
[print(g[0]) for g in dataset_group]
print("-----------------------")

# Device for computation
if args.device == 'cuda' and (not torch.cuda.is_available()):
    print("Warning: Cuda is requested but cuda is not available. CPU will be used.")
    args.device = 'cpu'
DEVICE = torch.device(args.device)


def get_model(config_path, model_path):
    """
    Creates a Mask2Former model give a config path and ckpt path
    """
    args = edict({'config_file': config_path, 'eval-only': True, 'opts': [
        "OUTPUT_DIR", "output/",
    ]})
    config = setup(args)

    model = Trainer.build_model(config)
    DetectionCheckpointer(model, save_dir=config.OUTPUT_DIR).resume_or_load(
        model_path, resume=False
    )
    model.to(DEVICE)
    _ = model.eval()

    return model


def get_logits(model, x, **kwargs):
    """
    Extracts the logits of a single image from Mask2Former Model. Works only for a single image currently.

    Expected input:
    - x: torch.Tensor of shape (1, 3, H, W)

    Expected output:
    - Logits (torch.Tensor) of shape (1, 19, H, W)
    """
    with torch.no_grad():
        out = model([{"image": x[0].to(DEVICE)}])

    return out[0]['sem_seg'].unsqueeze(0)

def get_msp(model, x, **kwargs):
    
    with torch.no_grad():
        out = model([{"image": x[0].to(DEVICE)}])
    logits = out[0]['sem_seg']

    return 1 - (logits.softmax(dim=0)).max(dim=0)[0]

def get_void(model, x, **kwargs):
    
    with torch.no_grad():
        out = model([{"image": x[0].to(DEVICE)}])
    logits = out[0]['sem_seg']

    return logits[19,:,:]

def get_ml(model, x, **kwargs):
    
    with torch.no_grad():
        out = model([{"image": x[0].to(DEVICE)}])
    logits = out[0]['sem_seg']

    return 1 - logits.max(dim=0)[0]


def get_entropy(model, x, **kwargs):
    
    with torch.no_grad():
        out = model([{"image": x[0].to(DEVICE)}])
    logits = out[0]['sem_seg']
    logits = logits.softmax(dim=0)
    return torch.div(torch.sum(-logits * torch.log(logits), dim=0), torch.log(torch.tensor(logits.shape[0])))

def get_aem(model, x, **kwargs):
    
    with torch.no_grad():
        out = model([{"image": x[0].to(DEVICE)}])
    logits = out[0]['sem_seg']

    return 1 - logits.sum(dim=0)

def get_m2a(model, x, **kwargs):
    
    with torch.no_grad():
        out = model([{"image": x[0].to(DEVICE)}])
    logits = out[0]['sem_seg']

    return 1 - logits.max(dim=0)[0]

def get_eam(model, x, **kwargs):
    
    with torch.no_grad():
        out = model([{"image": x[0].to(DEVICE)}])
    logits = out[0]['sem_seg']

    return 1 - logits.sum(dim=0)

def get_RbA(model, x, **kwargs):
    
    with torch.no_grad():
        out = model([{"image": x[0].to(DEVICE)}])

    logits = out[0]['sem_seg']

    return -logits.tanh().sum(dim=0)

def get_energy(model, x, **kwargs):

    with torch.no_grad():
        out = model([{"image": x[0].cuda()}])

    logits = out[0]['sem_seg']

    return -torch.logsumexp(logits, dim=0)


def save_dict(d, name):
    """
    Save the records into args.out_path. 
    Print the records to console if verbose=True
    """
    if args.verbose:
        pprint(d)
    store_path = os.path.join(args.out_path, name)
    Path(store_path).mkdir(exist_ok=True, parents=True)
    with open(os.path.join(store_path, f'results.pkl'), 'wb') as f:
        pickle.dump(d, f)


def current_result_exists(model_name):
    """
    Check if the current results exist in the args.out_path
    """
    store_path = os.path.join(args.out_path, model_name)
    return os.path.exists(os.path.join(store_path, f'results.pkl'))

def run_evaluations(model, dataset, model_name, dataset_name):
    """
    Run evaluations for a particular model over all designated datasets.
    """

    score_func = None
    if args.score_func == "rba":
        score_func = get_RbA
    elif args.score_func == "energy":
        score_func = get_energy
    elif args.score_func == "dense_hybrid":
        score_func = get_densehybrid_score
    elif args.score_func == "eam":
        score_func = get_eam
    elif args.score_func == "aem":
        score_func = get_aem
    elif args.score_func == "m2a":
        score_func = get_m2a
    elif args.score_func == "msp":
        score_func = get_msp  
    elif args.score_func == "ml":
        score_func = get_ml
    elif args.score_func == "entropy":
        score_func = get_entropy
    elif args.score_func == "void":
        score_func = get_entropy


    evaluator = OODEvaluator(model, get_logits, score_func)
    loader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    #evaluator.save_anomaly_scores(loader=loader, device=DEVICE, return_preds=False, upper_limit= 1000000)
    #evaluator.compute_and_save_anomaly_scores_from_single_images(loader=loader, device=DEVICE, return_preds=False, upper_limit= 100000)
    #anomaly_score, ood_gts = evaluator.compute_anomaly_scores_from_images()
    # anomaly_score, ood_gts = evaluator.save_anomaly_images(
    #     loader=loader,
    #     device=DEVICE,
    #     return_preds=False,
    #     upper_limit= 100000
    # )

    # anomaly_score, ood_gts = evaluator.compute_anomaly_scores(
    #     loader=loader,
    #     device=DEVICE,
    #     return_preds=False,
    #     upper_limit= 500
    # )

    # evaluator.compute_anomaly_scores_SOS(
    #     loader=loader,
    #     device=DEVICE,
    #     return_preds=False,
    #     upper_limit= 1000000
    # )

    evaluator.compute_anomaly_scores_LandF(
        loader=loader,
        device=DEVICE,
        return_preds=False,
        upper_limit= 100000
    )

    # evaluator.compute_anomaly_scores_SOD(
    #     loader=loader,
    #     device=DEVICE,
    #     return_preds=False,
    #     upper_limit= 1000000
    # )

    # evaluator.compute_anomaly_scores_appolloscapes(
    #     loader=loader,
    #     device=DEVICE,
    #     return_preds=False,
    #     upper_limit= 10
    # )

    #metrics = evaluator.evaluate_ood(anomaly_score=anomaly_score, ood_gts=ood_gts, verbose=False)
    #print(metrics)

    # anomaly_score, ood_gts = evaluator.temporal_compute_anomaly_scores(
    #     loader=loader,
    #     device=DEVICE,
    #     return_preds=False,
    #     upper_limit=1000000,
    #     temporal_frame=100
    # )

    # if args.store_anomaly_scores:
    #     vis_path = os.path.join(f"anomaly_scores/{model_name}/{dataset_name}")
    #     os.makedirs(vis_path, exist_ok=True)
    #     for i in tqdm(range(len(anomaly_score)), desc=f"storing anomaly scores at {vis_path}"):
            
    #         mpimg.imsave(os.path.join(vis_path, f"score_{i}.png"), anomaly_score[i].squeeze(), cmap='viridis')
    
    # moving_metrics = {'auroc': 0.0, 'aupr': 0.0, 'fpr95': 0.0}

    # counter = 1
    # for i in range(0, len(ood_gts), temporal_frame):

    #     if i+temporal_frame>len(ood_gts):
    #         metrics = evaluator.evaluate_ood(anomaly_score=anomaly_score[i:i+len(ood_gts)], ood_gts=ood_gts[i:i+len(ood_gts)], verbose=False)
    #     else:
    #         metrics = evaluator.evaluate_ood(anomaly_score=anomaly_score[i:i+temporal_frame], ood_gts=ood_gts[i:i+temporal_frame], verbose=False)
        
    #     moving_metrics['auroc'] = (moving_metrics['auroc']*(counter-1) + metrics['auroc'])/counter
    #     moving_metrics['aupr'] = (moving_metrics['aupr']*(counter-1) + metrics['aupr'])/counter
    #     moving_metrics['fpr95'] = (moving_metrics['fpr95']*(counter-1) + metrics['fpr95'])/counter
    #     counter = counter + 1
    #     print(moving_metrics)

    return


def main():

    # The name of every directory inside args.models_folder is expected to be the model name.
    # Inside a model's folder there should be 2 files (doesn't matter if there are extra stuff).
    # these 2 files are: config.yaml and [model_final.pth or model_final.pkl]
    # sam2_checkpoint = "/shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_1e-4.yaml/checkpoints/checkpoint_8.pt"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    sam2_checkpoint = args.model_pathh
    model_cfg = args.config_pathh
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    results = edict()
    for dataset_name, dataset in dataset_group:
        if dataset_name not in results:
            results[dataset_name] = edict()
        print(dataset_name)
        results[dataset_name] = run_evaluations(predictor, dataset, 'SAM2', dataset_name)
        break

if __name__ == '__main__':
    main()
