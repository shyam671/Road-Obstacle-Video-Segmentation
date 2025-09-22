#CUDA_VISIBLE_DEVICES=1 python training/train.py -c configs/sam2.1_training/sam2.1_VAS_b_ft.yaml --use-cluster 0 --num-gpus 1
#CUDA_VISIBLE_DEVICES=1 python evaluate.py --out_path results_test/ --score_func msp --model_path /shared-network/srai/VAS/code/sam2/sam2_logs/configs/sam2.1_training/best_sam2_config_b/checkpoints/checkpoint_25.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml
CUDA_VISIBLE_DEVICES=1 python evaluate.py --out_path results_test/ --score_func msp --model_path /home/shyam/VAS/VAS/code/sam2/sam2_logs/configs/sam2.1_training/best_sam2_config_b/checkpoints/checkpoint_20.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml



# CUDA_VISIBLE_DEVICES=7 python evaluate.py --out_path results_test/ --score_func msp --model_path /shared-network/srai/VAS/code/sam2/sam2_logs/configs/sam2.1_training/sam2.1_VAS_b_ft.yaml/checkpoints/checkpoint_15.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml
# CUDA_VISIBLE_DEVICES=7 python evaluate.py --out_path results_test/ --score_func msp --model_path /shared-network/srai/VAS/code/sam2/sam2_logs/configs/sam2.1_training/sam2.1_VAS_b_ft.yaml/checkpoints/checkpoint_20.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml
# CUDA_VISIBLE_DEVICES=7 python evaluate.py --out_path results_test/ --score_func msp --model_path /shared-network/srai/VAS/code/sam2/sam2_logs/configs/sam2.1_training/sam2.1_VAS_b_ft.yaml/checkpoints/checkpoint_25.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml
# CUDA_VISIBLE_DEVICES=7 python evaluate.py --out_path results_test/ --score_func msp --model_path /shared-network/srai/VAS/code/sam2/sam2_logs/configs/sam2.1_training/sam2.1_VAS_b_ft.yaml/checkpoints/checkpoint_30.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml
# CUDA_VISIBLE_DEVICES=7 python evaluate.py --out_path results_test/ --score_func msp --model_path /shared-network/srai/VAS/code/sam2/sam2_logs/configs/sam2.1_training/sam2.1_VAS_b_ft.yaml/checkpoints/checkpoint_35.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml
# CUDA_VISIBLE_DEVICES=7 python evaluate.py --out_path results_test/ --score_func msp --model_path /shared-network/srai/VAS/code/sam2/sam2_logs/configs/sam2.1_training/sam2.1_VAS_b_ft.yaml/checkpoints/checkpoint_40.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml

# CUDA_VISIBLE_DEVICES=7 python training/train.py -c configs/sam2.1_training/sam2.1_VAS_l_ft.yaml --use-cluster 0 --num-gpus 1
# CUDA_VISIBLE_DEVICES=7 python evaluate.py --out_path results_test/ --score_func msp --model_path /shared-network/srai/VAS/code/sam2/sam2_logs/configs/sam2.1_training/sam2.1_VAS_l_ft.yaml/checkpoints/checkpoint_15.pt --config_pathh configs/sam2.1/sam2.1_hiera_l.yaml
# CUDA_VISIBLE_DEVICES=7 python evaluate.py --out_path results_test/ --score_func msp --model_path /shared-network/srai/VAS/code/sam2/sam2_logs/configs/sam2.1_training/sam2.1_VAS_l_ft.yaml/checkpoints/checkpoint_20.pt --config_pathh configs/sam2.1/sam2.1_hiera_l.yaml
# CUDA_VISIBLE_DEVICES=7 python evaluate.py --out_path results_test/ --score_func msp --model_path /shared-network/srai/VAS/code/sam2/sam2_logs/configs/sam2.1_training/sam2.1_VAS_l_ft.yaml/checkpoints/checkpoint_25.pt --config_pathh configs/sam2.1/sam2.1_hiera_l.yaml
# CUDA_VISIBLE_DEVICES=7 python evaluate.py --out_path results_test/ --score_func msp --model_path /shared-network/srai/VAS/code/sam2/sam2_logs/configs/sam2.1_training/sam2.1_VAS_l_ft.yaml/checkpoints/checkpoint_30.pt --config_pathh configs/sam2.1/sam2.1_hiera_l.yaml
# CUDA_VISIBLE_DEVICES=7 python evaluate.py --out_path results_test/ --score_func msp --model_path /shared-network/srai/VAS/code/sam2/sam2_logs/configs/sam2.1_training/sam2.1_VAS_l_ft.yaml/checkpoints/checkpoint_35.pt --config_pathh configs/sam2.1/sam2.1_hiera_l.yaml
# CUDA_VISIBLE_DEVICES=7 python evaluate.py --out_path results_test/ --score_func msp --model_path /shared-network/srai/VAS/code/sam2/sam2_logs/configs/sam2.1_training/sam2.1_VAS_l_ft.yaml/checkpoints/checkpoint_40.pt --config_pathh configs/sam2.1/sam2.1_hiera_l.yaml

#CUDA_VISIBLE_DEVICES=7 python training/train.py -c configs/sam2.1_training/sam2.1_VAS_b_ftv2.yaml --use-cluster 0 --num-gpus 1
#CUDA_VISIBLE_DEVICES=7 python evaluate.py --out_path results_test/ --score_func msp --model_path /shared-network/srai/VAS/code/sam2/sam2_logs/configs/sam2.1_training/sam2.1_VAS_b_ftv2.yaml/checkpoints/checkpoint.pt --config_pathh configs/sam2.1/sam2.1_hiera_b+.yaml
#CUDA_VISIBLE_DEVICES=7 python training/train.py -c configs/sam2.1_training/sam2.1_VAS_l_ft.yaml --use-cluster 0 --num-gpus 1
#CUDA_VISIBLE_DEVICES=7 python evaluate.py --out_path results_test/ --score_func msp --model_path /shared-network/srai/VAS/code/sam2/sam2_logs/configs/sam2.1_training/sam2.1_VAS_l_ft.yaml/checkpoints/checkpoint.pt --config_pathh configs/sam2.1/sam2.1_hiera_l.yaml
