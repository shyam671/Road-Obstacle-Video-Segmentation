# Road-Obstacle-Video-Segmentation (GCPR 2025)

[[`arXiv`](https://arxiv.org/pdf/2509.13181)]

https://github.com/user-attachments/assets/47a5ce9c-73d9-4412-9f56-140a379be7a6

### Installation
Please follow the Installation Instructions to set up the codebase for [image anomaly segmentation](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md) and [CC-SAM2](https://github.com/facebookresearch/sam2).

### Datasets

* **Inlier Dataset(CityscapesVPS/Cityscapes):** To train image models we use Cityscapes and for video models Cityscapes VPS is utilized.
* **Anomaly Dataset (validation):** can be obtained and pre-processed using scripts [here](https://github.com/shyam671/Road-Obstacle-Video-Segmentation/tree/main/anomaly_dataset_creation_and_preprocessing).

### Training and Inference

* For training and inference CC-SAM 2, refer to [cmd.sh](https://github.com/shyam671/Road-Obstacle-Video-Segmentation/blob/main/sam2/cmd.sh).
* Image anomaly segmentation methods can be trained and inferred using [run.sh](https://github.com/shyam671/Road-Obstacle-Video-Segmentation/blob/main/Image_Anomaly_Segmentation_Baselines/run.sh).


### Acknowledgement

We thank the authors of the codebases mentioned below, which helped build the repository.
* [RbA](https://github.com/NazirNayal8/RbA)
* [Mask2Former](https://github.com/facebookresearch/Mask2Former/tree/main)
* [SAM2](https://github.com/facebookresearch/sam2)
