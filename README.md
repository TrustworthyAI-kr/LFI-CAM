# LFI-CAM


# Quick Runs
- Visualization
```
# activate your virtual environment, then..
tensorboard --logdir='board' --port=6006 --host='localhost'

# If you want to change the directory for the tensorboard visualizatoin,
# select where-ever you want(ex: /xxx/yyy/board), and put argument "--board-path /xxx/yyy/board"
```

- Catdog
```
python imagenet.py -a resnet18 --data ./data/catdog --epochs 300 --checkpoint checkpoints/catdog/abn --gpu-id 0,1,2,3 --lr 0.1 --schedule 150 225 --num_classes 2
```

- STL10
```
python stl.py -a resnet18 --epochs 300 --checkpoint checkpoints/stl/abn --gpu-id 0,1,2,3 --lr 0.1 --schedule 150 225
```


## Using CIFAR-10 or CIFAR-100
- Training & Evaluating from Scratch
```
python cifar.py -a resnet --dataset cifar10 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110-mod --gpu-id 0,1,2,3
```

## Using ImageNet or custom dataset
- Training & Evaluating from Scratch
```
python abn.py -a resnet50 --data ./data/neu_split --epochs 300 --checkpoint checkpoints/neu/res50_mix --gpu-id 0,1,2
```

- Training & Evaluating through Finetuning and Selective Freezing
```
python abn.py -a resnet50 --data ./data/neu_split --epochs 300 --checkpoint checkpoints/neu/res50_mix --gpu-id 0,1,2 --fine_tune --freeze_layer 4
```

- ONLY running Eval mode for Model Trained from Scratch
```
python abn.py -a resnet50 --data ./data/neu_split --epochs 1 --checkpoint checkpoints/neu/res50_mix --gpu-id 0,1,2
 --evaluate --resume checkpoints/neu/res50_mix/best_checkpoint.pth.tar
```

- ONLY running Eval mode for Model Trained through Finetuning and Selective Freezing
```
python abn.py -a resnet50 --data ./data/neu_split --epochs 1 --checkpoint checkpoints/neu/res50_mix --gpu-id 0,1,2
--evaluate --resume checkpoints/neu/res50_mix/best_checkpoint.pth.tar --freeze_layer 4
```



## Enviroment
Our source code is based on [https://github.com/bearpaw/pytorch-classification/](https://github.com/bearpaw/pytorch-classification/) implemented with PyTorch. We are grateful for the author!
Requirements of PyTorch version are as follows:
- PyTorch : 0.4.0
- PyTorch vision : 0.2.1


## Execution
Example of run command is as follows:

#### Training
```bash
python3 cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-110 --gpu-id 0,1

python3 imagenet.py -a resnet152 --data ../../dataset/imagenet_data/ --epochs 90 --schedule 31 61 --gamma 0.1 -c checkpoints/imagenet/resnet152 --gpu-id 4,5,6,7 --test-batch 100
```

#### Evaluation
```bash
python3 cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-110 --gpu-id 0,1 --evaluate --resume checkpoints/cifar100/resnet-110/model_best.pth.tar

python3 imagenet.py -a resnet152 --data ../../../../dataset/imagenet_data/ --epochs 90 --schedule 31 61 --gamma 0.1 -c checkpoints/imagenet/resnet152 --gpu-id 4,5,6 --test-batch 10 --evaluate --resume checkpoints/imagenet/resnet152/model_best.pth.tar
```

## Performances
### CIFAR100
|  | top-1 error (LFI-CAM) | top-1 error (ABN) | top-1 error ([original](https://github.com/bearpaw/pytorch-classification)) |
|:------------|------------:|------------:|------------:|
| ResNet110   |        --.- |        22.5 |        24.1 |
| DenseNet    |        --.- |        21.6 |        22.5 |
| Wide ResNet |        --.- |        18.1 |        18.9 |
| ResNeXt     |        --.- |        17.7 |        18.3 |

### ImageNet2012
|  | top-1 error (LFI-CAM) | top-1 error (ABN) | top-1 error ([original](https://github.com/bearpaw/pytorch-classification)) |
|:------------|------------:|------------:|
| ResNet50    |        --.- |        23.1 |        24.1 |
| ResNet101   |        --.- |        21.8 |        22.5 |
| ResNet152   |        --.- |        21.4 |        22.2 |

### Examples of attention map
![overview image](./example.jpeg)



