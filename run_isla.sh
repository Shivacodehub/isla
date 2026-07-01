#!/bin/bash
# run_isla.sh

MODEL="models/stable-diffusion-v1-5"
LR=5e-6
ITER=201
INNER=30
BATCH=1
SEED=0

EXP=$1

if [ "$EXP" = "isla001" ]; then
    python protect.py \
        --pretrained_model_name_or_path $MODEL \
        --exp isla001 \
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
        --save_freq 100000 \
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
        --phase0_timesteps 100 300 500 700 900
fi

if [ "$EXP" = "isla001_attack" ]; then
    for SUBJ in 1 2 3 4; do
        echo "=== Attacking subject ${SUBJ} ==="
        python train_dreambooth.py \
            --pretrained_model_name_or_path $MODEL \
            --additional_unet_path experiments/isla001_debug/unet_002.pt \
            --instance_data_dir data/person${SUBJ}/set_A \
            --instance_prompt "a photo of sks${SUBJ} person" \
            --class_data_dir class_images/person \
            --class_prompt "a photo of a person" \
            --with_prior_preservation \
            --output_dir experiments/isla001_attack_s${SUBJ} \
            --max_train_steps 1000 \
            --train_batch_size 1 \
            --seed $SEED

        echo "=== Evaluating subject ${SUBJ} ==="
        python evaluate_db.py \
            --checkpoint experiments/isla001_attack_s${SUBJ} \
            --class_noun person \
            --identifier sks${SUBJ} \
            --data_dir data/person${SUBJ}/set_A \
            --output_dir experiments/isla001_debug/eval_s${SUBJ} \
            --dino_score
    done
fi