#!/usr/bin/env bash
python train.py --epochs 20 \
                --batch_size 32 \
                --batch_size_u 96 \
                --max_seq_length 64 \
                --lrmain 1e-5 \
                --lrlast 1e-3 \
                --gpu 0,1,2,3 \
                --output_dir ../output_models/ \
                --data_path ../processed_data/ \
                --tsa \
                --uda \
                --weight_decay 0.0 \
                --adam_epsilon 1e-8 \
                --average 'macro' \
                --warmup_steps 100 \
                --lambda_u 1.0 \
                --T 1.0 \
                --no_class 0 \
                --tsa_type 'exp'
