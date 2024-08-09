#!/bin/bash
#SBATCH --account=qmei
#SBATCH --partition=qmei-a100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=500g
#SBATCH --time=48:00:00
#SBATCH --job-name=llama3-70b-text-generate!
#SBATCH --output=/scratch/qmei_root/qmei/xziyang/logs/%x-%A-%j.log
#SBATCH --mail-user=xziyang@umich.edu
#SBATCH --mail-type=END,FAIL

PROJECT_DIR=/scratch/qmei_root/qmei/xziyang/llama-recipes
SCRITP=src/llama_recipes/finetuning.py
CONFIG_FILE="configs/llama.yaml"

echo "hostname:"
hostname
echo "nvidia-smi"
nvidia-smi

module load cuda/12.1.1
source ~/.bashrc 
conda activate test

cd $PROJECT_DIR
echo $CONFIG_FILE

torchrun --nnodes 1 --nproc_per_node 4 $SCRITP $CONFIG_FILE
