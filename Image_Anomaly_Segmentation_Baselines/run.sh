

############## IAS Baselines ##################

# # Entropy
# CUDA_VISIBLE_DEVICES=4 python evaluate_ood.py --out_path results_test/ --models_folder /shared-network/srai/VAS/ckpts/M2F/ --model_mode selective --selected_models swin-l --score_func entropy
# # # Energy
# CUDA_VISIBLE_DEVICES=3 python evaluate_ood.py --out_path results_test/ --models_folder /shared-network/srai/VAS/ckpts/M2F/ --model_mode selective --selected_models swin-l --score_func energy 
# # # # MSP
# CUDA_VISIBLE_DEVICES=3 python evaluate_ood.py --out_path results_test/ --models_folder /shared-network/srai/VAS/ckpts/M2F/ --model_mode selective --selected_models swin-l --score_func msp
# # # # ML
# CUDA_VISIBLE_DEVICES=3 python evaluate_ood.py --out_path results_test/ --models_folder /shared-network/srai/VAS/ckpts/M2F/ --model_mode selective --selected_models swin-l --score_func ml 

# # # Mask2Anomaly
#CUDA_VISIBLE_DEVICES=3 python evaluate_ood.py --out_path results_test/ --models_folder /shared-network/srai/VAS/ckpts/M2A/ --model_mode selective --selected_models without_oe --score_func m2a
# # # AEM
# CUDA_VISIBLE_DEVICES=3 python evaluate_ood.py --out_path results_test/ --models_folder /shared-network/srai/VAS/ckpts/EAM-AEM/ --model_mode selective --selected_models swin-l --score_func aem 

## EAM
## In this case, change network inference
#CUDA_VISIBLE_DEVICES=2 python evaluate_ood.py --out_path results_test/ --models_folder /shared-network/srai/VAS/ckpts/EAM-AEM/ --model_mode selective --selected_models swin-l --score_func eam 
# # # # Void
## In this case, change network inference
#CUDA_VISIBLE_DEVICES=3 python evaluate_ood.py --out_path results_test/ --models_folder /shared-network/srai/VAS/ckpts/M2F/ --model_mode selective --selected_models swin-l --score_func void
## RbA
## In this case uncomment
## score = (score + 1.0)/2
## For VC keep threshold -0.5
#CUDA_VISIBLE_DEVICES=4 python evaluate_ood.py --out_path results_test/ --models_folder /shared-network/srai/VAS/ckpts/RbA --model_mode selective --selected_models swin_b_1dl --score_func rba

# CUDA_VISIBLE_DEVICES=7 python evaluate_ood.py --out_path results_test/ --models_folder /shared-network/srai/VAS/ckpts/RbA --model_mode selective --selected_models swin_b_1dl_rba_ood_coco --score_func m2a
# CUDA_VISIBLE_DEVICES=7 python evaluate_ood.py --out_path results_test/ --models_folder /shared-network/srai/VAS/ckpts/



## EAM-AEM
#CUDA_VISIBLE_DEVICES=0 python evaluate_ood.py --out_path results_test/ --models_folder /shared-network/srai/VAS/ckpts/EAM-AEM/ --model_mode selective --selected_models swin-l --score_func eam > output/eam.txt
#CUDA_VISIBLE_DEVICES=0 python evaluate_ood.py --out_path results_test/ --models_folder /shared-network/srai/VAS/ckpts/EAM-AEM/ --model_mode selective --selected_models swin-l --score_func aem  > output/aem.txt
#CUDA_VISIBLE_DEVICES=0 python evaluate_ood.py --out_path results_test/ --models_folder /shared-network/srai/VAS/ckpts/EAM-AEM/ --model_mode selective --selected_models swin-l-oe --score_func eam > output/eam-oe.txt
#CUDA_VISIBLE_DEVICES=7 python evaluate_ood.py --out_path results_test/ --models_folder /shared-network/srai/VAS/ckpts/EAM-AEM/ --model_mode selective --selected_models swin-l-oe --score_func aem --selected_datasets lost_and_found

#CUDA_VISIBLE_DEVICES=0 python evaluate_ood.py --out_path results_test/ --models_folder /shared-network/srai/VAS/ckpts/EAM-AEM/ --model_mode selective --selected_models swin-l-oe --score_func eam 

#CUDA_VISIBLE_DEVICES=3 python train_net_video.py --config-file configs/vss/r50.yaml --num-gpus 1

 
#CUDA_VISIBLE_DEVICES=2 python evaluate_ood.py --out_path results_test/ --models_folder /shared-network/srai/VAS/ckpts/RbA/ --model_mode selective --selected_models swin_b_1dl --score_func rba 



#CUDA_VISIBLE_DEVICES=0 python evaluate_ood.py --out_path results_test/ --models_folder /shared-network/srai/VAS/ckpts/M2F/ --model_mode selective --selected_models swin-l --score_func void


#SAM
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_1e-4.yaml/checkpoints/checkpoint_2.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/1e-4_2.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_1e-4.yaml/checkpoints/checkpoint_4.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/1e-4_4.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_1e-4.yaml/checkpoints/checkpoint_6.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/1e-4_6.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_1e-4.yaml/checkpoints/checkpoint_8.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/1e-4_8.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_1e-4.yaml/checkpoints/checkpoint_10.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/1e-4_10.txt

# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_1e-5.yaml/checkpoints/checkpoint_2.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/1e-5_2.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_1e-5.yaml/checkpoints/checkpoint_4.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/1e-5_4.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_1e-5.yaml/checkpoints/checkpoint_6.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/1e-5_6.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_1e-5.yaml/checkpoints/checkpoint_8.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/1e-5_8.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_1e-5.yaml/checkpoints/checkpoint.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/1e-5_10.txt

# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_1e-6.yaml/checkpoints/checkpoint_2.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/1e-6_2.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_1e-6.yaml/checkpoints/checkpoint_4.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/1e-6_4.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_1e-6.yaml/checkpoints/checkpoint_6.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/1e-6_6.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_1e-6.yaml/checkpoints/checkpoint_8.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/1e-6_8.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_1e-6.yaml/checkpoints/checkpoint.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/1e-6_10.txt

# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_5e-5.yaml/checkpoints/checkpoint_2.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/5e-5_2.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_5e-5.yaml/checkpoints/checkpoint_4.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/5e-5_4.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_5e-5.yaml/checkpoints/checkpoint_6.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/5e-5_6.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_5e-5.yaml/checkpoints/checkpoint_8.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/5e-5_8.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_5e-5.yaml/checkpoints/checkpoint_10.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/5e-5_10.txt


# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_5e-6.yaml/checkpoints/checkpoint_2.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/5e-6_2.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_5e-6.yaml/checkpoints/checkpoint_4.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/5e-6_4.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_5e-6.yaml/checkpoints/checkpoint_6.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/5e-6_6.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_5e-6.yaml/checkpoints/checkpoint_8.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/5e-6_8.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/cvps_5e-6.yaml/checkpoints/checkpoint_10.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/5e-6_10.txt

#CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/frame_1.yaml/checkpoints/checkpoint.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/frame_1.txt
#CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/frame_2.yaml/checkpoints/checkpoint.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/frame_2.txt

# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/frame_2_large_3e-5.yaml/checkpoints/checkpoint_50.pt --config_pathh configs/sam2.1/sam2.1_hiera_l.yaml > output/frame_2_large_3e-5.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/frame_2_large_5e-5.yaml/checkpoints/checkpoint_50.pt --config_pathh configs/sam2.1/sam2.1_hiera_l.yaml > output/frame_2_large_5e-5.txt
# CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-local/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/frame_2_large_7e-5.yaml/checkpoints/checkpoint_50.pt --config_pathh configs/sam2.1/sam2.1_hiera_l.yaml > output/frame_2_large_7e-5.txt

#CUDA_VISIBLE_DEVICES=2 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-network/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/frame_2.yaml/checkpoints/checkpoint_40.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml
#CUDA_VISIBLE_DEVICES=0 python evaluate_SAM_ood.py --out_path results_test/ --score_func msp --model_path  /shared-network/srai/VAS/code/sam2.1/sam2_logs/configs/sam2.1_training/frame_2.yaml/checkpoints/checkpoint_60.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml > output/frame_2_60b+.txt

