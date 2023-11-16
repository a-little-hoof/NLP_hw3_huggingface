#! /bin/bash

#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=sbatch_example
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=singlegpu
#SBATCH --cpus-per-task=10
#SBATCH --time 12:00:00

seed=(111 222 333 444 555 666 777 888 999)

for round in 0 1 2 3 4;
do
  for ft_task in 'restaurant' 'acl' 'agnews_sup';
  do
    for model_name in 'roberta-base';
    do
      CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python train_combined.py \
      --dataset_name ${ft_task} \
      --do_train \
      --do_eval \
      --max_seq_length 64 \
      --learning_rate 5e-4 \
      --cache_dir ./cache \
      --model_name_or_path $model_name \
      --seed ${seed[$round]} \
      --num_train_epochs 5 \
      --per_device_train_batch_size 32 \
      --output_dir ./result/$ft_task'_'$model_name'_seed'$round'_lora' \
      --save_strategy no \
      --use_lora True \
      --logging_strategy epoch \
      --evaluation_strategy epoch
    done
  done
done


