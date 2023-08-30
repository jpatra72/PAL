#!/bin/bash
#SBATCH --partition=RMC-C01-BATCH
#SBATCH --job-name="pal-v1.0.i"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=8-00:00:00
#SBATCH --exclude=rmc-gpu03
#SBATCH --mail-type=ALL
#SBATCH --output=/home/patr_jy/Github/PAL_DLR/src/logs/slurm-%j.out
#SBATCH --error=/home/patr_jy/Github/PAL_DLR/src/logs/slurm-%j.out


source ~/mambaforge/etc/profile.d/conda.sh
conda activate nerlstm

export WANDB_CACHE_DIR=$HOME/wandbdir/

export agent="cnndqn"
export episode=8000
export budget=200
export train="en.train,en.testa,en.testb,cc.en.40.vec,models/en.model.saved"
export test="de.train,de.testa,de.testb,cc.de.40.vec,models/de.model.saved"
export data_dir="data/conll2003/"
export max_seq_len=80
export max_vocab_size=20000
export embedding_size=40
export model_name="LSTM"
export log_path="logs/log.txt"
export log_level="INFO"
export wandb_tracking=True
export wandb_project="PAL_LSTM"
export epochs=20
export target_update_time=1
export explore_steps=150000

python ~/Github/PAL_DLR/src/launcher_ner_bilingual.py \
    --agent "$agent" \
    --episode "$episode" \
    --budget "$budget" \
    --train "$train" \
    --test "$test" \
    --data_dir "$data_dir" \
    --max_seq_len "$max_seq_len" \
    --max_vocab_size "$max_vocab_size" \
    --embedding_size "$embedding_size" \
    --model_name "$model_name" \
    --log_path "$log_path" \
    --log_level "$log_level" \
    --wandb_tracking "$wandb_tracking" \
    --wandb_project "$wandb_project" \
    --epochs "$epochs" \
    --target_update_time "$target_update_time" \
    --explore_steps "$explore_steps" &

wait