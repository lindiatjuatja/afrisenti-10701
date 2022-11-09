#!/bin/bash                                                                     
#SBATCH --job-name=afrisenti_metaxl_metaxl
#SBATCH --output="/home/ttlu/afrisenti-10701/metaxl/logs/%x_%j.out"  
#SBATCH --error="/home/ttlu/afrisenti-10701/metaxl/errors/%x_%j.out"
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=13000                                
cd /home/ttlu/afrisenti-10701/metaxl
source /user_data/ttlu/miniconda3/bin/activate
module load cuda-11.6
module load cudnn-11.6-v8.4.1.50
stdbuf -oL -eL python -u mtrain.py --tgt_lang sw --do_train --do_eval --gradient_accumulation_steps 4