#!/usr/bin/env bash
python predict.py   --batch_size 512 \
                    --max_seq_length 64 \
                    --gpu 0,1,2,3 \
                    --output_dir ../output_models/ \
                    --data_path ../processed_data/ \
                    --average 'macro' \
                    --no_class 0 \
                    --model_path ../output_models/