# Few-Shot Learning with Graph Neural Networks
Implementation of [Few-Shot Learning with Graph Neural Networks](https://arxiv.org/pdf/1711.04043.pdf) on Python3, Pytorch 0.3.1


## Mini-Imagenet

### Download the dataset
Create **images.zip** file and copy it inside ```mini_imagenet``` directory:

    
    .
    ├── ...
    └── datasets                    
       └── compressed                
          └── mini_imagenet
             └── images.zip

The **images.zip** file must contain the splits and images in the following format:

    ── images.zip
       ├── test.csv                
       ├── train.csv 
       ├── val.csv 
       └── images
          ├── n0153282900000006.jpg
          ├── ...
          └── n1313361300001299.jpg

The splits *{test.csv, train.csv, val.csv}* can be downloaded from [Ravi and Larochelle - splits](https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet). For more information on how to obtain the images check the original source [Ravi and Larochelle - github](https://github.com/twitter/meta-learning-lstm)


### Training

```
# 5-Way 1-shot | Few-shot 
EXPNAME=minimagenet_N5_S1
python3 main.py --exp_name $EXPNAME --dataset mini_imagenet --test_N_way 5 --train_N_way 5 --train_N_shots 1 --test_N_shots 1 --batch_size 100 --dec_lr=15000 --iterations 80000

# 5-Way 5-shot | Few-shot 
EXPNAME=minimagenet_N5_S5
python3 main.py --exp_name $EXPNAME --dataset mini_imagenet --test_N_way 5 --train_N_way 5 --train_N_shots 5 --test_N_shots 5 --batch_size 40 --dec_lr=15000 --iterations 90000

# 5-Way 5-shot 20%-labeled | Semi-supervised  
EXPNAME=minimagenet_N5_S1_U4
python3 main.py --exp_name $EXPNAME --dataset mini_imagenet --test_N_way 5 --train_N_way 5 --train_N_shots 5 --test_N_shots 5  --unlabeled_extra 4 --batch_size 40 --dec_lr=15000 --iterations 100000
```


## Omniglot

### Download the dataset
Download **images_background.zip** and **images_evaluation.zip** files from [brendenlake/omniglot](https://github.com/brendenlake/omniglot/tree/master/python) and copy it inside the ```omniglot``` directory:

    .
    ├── ...
    └── datasets                    
       └── compressed                
          └── omniglot
             ├── images_background.zip
             └── images_evaluation.zip
             
### Training
```
# 5-Way 1-shot | Few-shot 
EXPNAME=omniglot_N5_S1_v2
python3 main.py --exp_name $EXPNAME --dataset omniglot --test_N_way 5 --train_N_way 5 --train_N_shots 1 --test_N_shots 1 --batch_size 300  --dec_lr=10000  --iterations 100000

# 5-Way 5-shot | Few-shot 
EXPNAME=omniglot_N5_S5
python3 main.py --exp_name $EXPNAME --dataset omniglot --test_N_way 5 --train_N_way 5 --train_N_shots 5 --test_N_shots 5 --batch_size 100  --dec_lr=10000  --iterations 80000

# 20-Way 1-shot | Few-shot 
EXPNAME=omniglot_N20_S1
python3 main.py --exp_name $EXPNAME --dataset omniglot --test_N_way 20 --train_N_way 20 --train_N_shots 1 --test_N_shots 1 --batch_size 100  --dec_lr=10000  --iterations 80000

# 5-Way 5-shot 20%-labeled | Semi-supervised  
EXPNAME=omniglot_N5_S1_U4
python3 main.py --exp_name $EXPNAME --dataset omniglot --test_N_way 5 --train_N_way 5 --train_N_shots 5 --test_N_shots 5 --unlabeled_extra 4 --batch_size 100  --dec_lr=10000  --iterations 80000
```

## Citation
If you find this code useful you can cite us using the following bibTex:
```
@article{garcia2017few,
  title={Few-Shot Learning with Graph Neural Networks},
  author={Garcia, Victor and Bruna, Joan},
  journal={arXiv preprint arXiv:1711.04043},
  year={2017}
}
```
