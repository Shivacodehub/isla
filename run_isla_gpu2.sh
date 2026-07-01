#!/bin/bash
# run_isla_gpu2.sh
# =================
# Runs ISLA protection across GPUs 2 and 4 (mapped locally to cuda:0, cuda:1).
# Does NOT touch GPUs 0,1,3,5,6,7 — safe to run alongside other users' jobs.
#
# Primary GPU (local cuda:0 / physical 2): pipe.unet, unet_temp, vae, text_encoder
# Secondary GPU (local cuda:1 / physical 4): unet_ref (frozen preservation reference)
#
# Usage: bash run_isla_gpu2.sh isla002

export CUDA_VISIBLE_DEVICES=2

MODEL="models/stable-diffusion-v1-5"
LR=5e-6
ITER=50
INNER=3
BATCH=1
SEED=0

EXP=$1

if [ "$EXP" = "isla004" ]; then
    python protect1.py \
        --pretrained_model_name_or_path $MODEL \
        --exp isla004 \
        --lr $LR \
        --iter $ITER \
        --num_inner_iter $INNER \
        --batch_size $BATCH \
        --seed $SEED \
        --instance_data_dir data/person1/set_A \
        --instance_prompt "a photo of sks1 person" \
        --class_data_dir class_images/person \
        --class_prompt "a photo of a person" \
        --num_samples 200 \
        --with_prior_preservation \
        --negative_loss \
        --in_ppl \
        --grad_accum_type sum \
        --print_freq 10 \
        --save_freq 10 \
        --isla_mode \
        --num_subjects 4 \
        --subject_data_dirs \
            data/person1/set_A \
            data/person2/set_A \
            data/person3/set_A \
            data/person4/set_A \
        --subject_prompts \
            "a photo of sks1 person" \
            "a photo of sks2 person" \
            "a photo of sks3 person" \
            "a photo of sks4 person" \
        --arcface_model_path none \
        --top_k_heads 16 \
        --lambda_id 1.0 \
        --lambda_head 0.1 \
        --lambda_pres 0.5 \
        --id_tau 0.3 \
        --phase0_timesteps 100 300 500 700 900 \
        --secondary_gpu 1
fi

i#!/bin/bash
# run_isla_gpu2.sh
# Usage: bash run_isla_gpu2.sh isla002
#        bash run_isla_gpu2.sh isla002_attack

export CUDA_VISIBLE_DEVICES=2

BASE=/home/shrikrishnal/isla
MODEL=$BASE/models/stable-diffusion-v1-5
DATA=$BASE/data
CLASS=$BASE/class_images/person
LR=5e-6
ITER=201
INNER=30
BATCH=1
SEED=0

EXP=$1

if [ "$EXP" = "isla002" ]; then
    cd $BASE
    python protect1.py \
        --pretrained_model_name_or_path $MODEL \
        --exp isla002 \
        --lr $LR \
        --iter $ITER \
        --num_inner_iter $INNER \
        --batch_size $BATCH \
        --seed $SEED \
        --instance_data_dir $DATA/person1/set_B \
        --instance_prompt "a photo of sks1 person" \
        --class_data_dir $CLASS \
        --class_prompt "a photo of a person" \
        --num_samples 200 \
        --with_prior_preservation \
        --negative_loss \
        --in_ppl \
        --grad_accum_type sum \
        --print_freq 10 \
        --save_freq 100000 \
        --isla_mode \
        --num_subjects 4 \
        --subject_data_dirs \
            $DATA/person1/set_B \
            $DATA/person2/set_B \
            $DATA/person3/set_B \
            $DATA/person4/set_B \
        --subject_prompts \
            "a photo of sks1 person" \
            "a photo of sks2 person" \
            "a photo of sks3 person" \
            "a photo of sks4 person" \
        --arcface_model_path none \
        --top_k_heads 16 \
        --lambda_id 1.0 \
        --lambda_head 0.1 \
        --lambda_pres 0.5 \
        --id_tau 0.3 \
        --phase0_timesteps 100 300 500 700 900
fi

if [ "$EXP" = "isla002_attack" ]; then
    cd $BASE
    for SUBJ in 1 2 3 4; do
        echo "=== Attacking subject ${SUBJ} ==="
        python train_dreambooth.py \
            --pretrained_model_name_or_path $MODEL \
            --additional_unet_path $BASE/experiments/isla002/unet_200.pt \
            --instance_data_dir $DATA/person${SUBJ}/set_B \
            --instance_prompt "a photo of sks${SUBJ} person" \
            --class_data_dir $CLASS \
            --class_prompt "a photo of a person" \
            --with_prior_preservation \
            --output_dir $BASE/experiments/isla002_attack_s${SUBJ} \
            --max_train_steps 1000 \
            --train_batch_size 1 \
            --seed $SEED

        echo "=== Evaluating subject ${SUBJ} ==="
        python evaluate_db.py \
            --checkpoint $BASE/experiments/isla002_attack_s${SUBJ} \
            --class_noun person \
            --identifier sks${SUBJ} \
            --data_dir $DATA/person${SUBJ}/set_B \
            --output_dir $BASE/experiments/isla002/eval_s${SUBJ} \
            --dino_score
    done
fi