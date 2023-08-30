#!/bin/bash
#SBATCH --partition=RMC-C01-BATCH
#SBATCH --job-name="paltorchgpu"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/home/patr_jy/Github/PAL_DLR/src/logs/slurm-%j.out
#SBATCH --error=/home/patr_jy/Github/PAL_DLR/src/logs/slurm-%j.out

source ~/mambaforge/etc/profile.d/conda.sh
conda activate nerlstm

export WANDB_CACHE_DIR=$HOME/wandbdir/

export budget=200
export train="en.train,en.testa,en.testb,cc.en.40.vec,models/en.model.saved"
export test="de.train,de.testa,de.testb,cc.de.40.vec,models/de.model.saved"
export data_dir="data/conll2003/"
export max_seq_len=120
export max_vocab_size=20000
export embedding_size=40
export model_name="CRF"


python ~/Github/PAL_DLR/src/tune_lstm.py \
    --budget "$budget" \
    --train "$train" \
    --test "$test" \
    --data_dir "$data_dir" \
    --max_seq_len "$max_seq_len" \
    --max_vocab_size "$max_vocab_size" \
    --embedding_size "$embedding_size" \
    --model_name "$model_name"

