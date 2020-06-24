# Semi-Supervised Models via Data Augmentationfor Classifying Interactive Affective Responses

This is a PyTorch implementation of our paper "Semi-Supervised Models via Data Augmentationfor Classifying Interactive Affective Responses" for AAAI shared task: CL-Aff Shared Task - Get it #OffMyChest. 

If you would like to use our codes, please cite the above paper.

Currently, we have provided the core code for SMDA and  preprocessed train/dev/test data. If there is any questions, feel free to contact us.

## Requirements

python 3
To install all the packages needed, please use
```
pip install -r requirements.txt 
``` 

## Download Data

We released our pre-processed train/dev/test data for you to start training.

- processed_data/labeled_data.pkl,train_unlabeled_data.pkl,test_unlabeled_data.pkl

## Data Augmentation

We released our code for data augmentation. The models we use for back translation are ''transformer.wmt19.en-de'' and ''transformer.wmt19.de-en'' from [fairseq](https://github.com/pytorch/fairseq/tree/master/examples/translation).

To run back translation, please use

```
python back_tanslation.py   --gpu gpu_number \
                            --data_path path_for_data
``` 

or

```
bash bt.sh
``` 

We also provided our augmented unlabeled data via back translation.

- processed_data/train_unlabeled_data_bt_69000.pkl

## Training

To train our SMDA model, please first make sure you have processed data as required and then training with default parameters:
```
python train.py --epochs 20 \
                --batch_size 256 \
                --batch_size_u 64 \
                --max_seq_length 64 \
                --lrmain 3e-6 \
                --lrlast 1e-3 \
                --gpu gpu_number \
                --output_dir path_for_output \
                --data_path path_for_data \
                --uda \
                --weight_decay 0.0 \
                --adam_epsilon 1e-8 \
                --average 'macro' \
                --warmup_steps 100 \
                --lambda_u 0.1 \
                --T 1.0 \
                --no_class 0 \
```

or

```
bash train.sh
```

You may change the no_class to change the classification task as you want.

## Testing

We also provided scripts for running prediction with trained model. You can use:
```
python predict.py   --batch_size 512 \
                    --max_seq_length 64 \
                    --gpu 0,1,2,3 \
                    --output_dir path_for_output \
                    --data_path path_for_data \
                    --average 'macro' \
                    --no_class 0 \
                    --model_path path_for_trained_model
```

or

```
bash predict.sh
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
 
