# LFI-CAM: Learning Feature Importance for Better Visual Explanation-Official Pytorch Implementation

## Abstract 

This repository contains source code for the **LFI-CAM(Learning of Feature Importance CAM, 
[arXiv paper](https://arxiv.org/pdf/2105.00937.pdf))** paper.

<p align="justify">
Class Activation Mapping (CAM) is a powerful technique used to understand the decision making of Convolutional Neural Network (CNN) in computer vision. Recently, there have been attempts not only to generate better visual explanations, but also to improve classification performance using visual explanations. However, the previous works still have their own drawbacks. In this paper, we propose a novel architecture, LFI-CAM, which is trainable for image classification and visual explanation in an end-to-end manner. LFI-CAM generates an attention map for visual explanation during forward propagation, at the same time, leverages the attention map to improve the classification performance through the attention mechanism. Our Feature Importance Network (FIN) focuses on learning the feature importance instead of directly learning the attention map to obtain a more reliable and consistent attention map. We confirmed that LFI-CAM model is optimized not only by learning the feature importance but also by enhancing the backbone feature representation to focus more on important features of the input image. Experimental results show that LFI-CAM outperforms the baseline models’s accuracy on the classification tasks as well as significantly improves on the previous works in terms of attention map quality and stability over different hyper-parameters.
</p>


***Authors: [Kwang Hee Lee](https://github.com/kh22l22)<sup>1,\*,\*\*</sup>, [Chaewon Park](https://github.com/emilypark0418)<sup>1,\*</sup>, [Junghyun Oh](https://github.com/jhvics1)<sup>1,2,*</sup>, and Nojun Kwak<sup>2</sup>***

> <sup>1</sup> Boeing Korea Engineering and Technology Center(BKETC), <sup>2</sup> Seoul National University 

> <sup>\*</sup> indicates equal contribution, <sup>\*\*</sup> indicates corresponding author


## Quick Guides
- Using Tensorboard
```
# activate your virtual environment, then..
tensorboard --logdir='board' --port=6006 --host='localhost'

# If you want to change the directory for the tensorboard visualization,
# select wherever you want(ex: /xxx/yyy/board), and put argument "--board-path /xxx/yyy/board"
```

## Training + Evaluation
### Using Catdog dataset
```
python imagenet.py -a resnet18 --data ./data/catdog --epochs 300 --checkpoint checkpoints/catdog/resnet18 \
 --gpu-id 0,1,2,3 --lr 0.1 --schedule 150 225 --num_classes 2
```

### Using STL10 dataset
```
python stl.py -a resnet18 --epochs 300 --checkpoint checkpoints/stl/resnet18 \
--gpu-id 0,1,2,3 --lr 0.1 --schedule 150 225
```

### Using CIFAR-10 or CIFAR-100 dataset
```
python cifar.py -a resnet --dataset cifar10 --depth 110 --epochs 300 \
--schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet --gpu-id 0,1,2,3
```

### Using ImageNet dataset
```
python imagenet.py -a resnet50 --data ./data/imagenet --epochs 300 \
--checkpoint checkpoints/imagenet/resnet50 --gpu-id 0,1,2
```

## Evaluation Only
```
python imagenet.py -a resnet50 --data ./data/imagenet --epochs 1 --checkpoint checkpoints/imagenet/resnet50 \ 
--gpu-id 0,1,2,3 --evaluate --resume checkpoints/neu/res50_mix/best_checkpoint.pth.tar
```


## Environment
```
pip install -r requirements.txt
```

Our source code is based on [https://github.com/bearpaw/pytorch-classification/](https://github.com/bearpaw/pytorch-classification/) and [https://github.com/machine-perception-robotics-group/attention_branch_network/](https://github.com/machine-perception-robotics-group/attention_branch_network) implemented with PyTorch. 
Requirements of PyTorch version are as follows:
- PyTorch : 0.4.0
- PyTorch vision : 0.2.1

## Attention Map Example and Stability Analysis
![overview image](./asset/Stability_Test.png)

> Examples of stability test on visual explanation. Each row displays CAM results of ABN or LFI-CAM models that were trained with various (5) hyper-parameters. As illustrated, ABN’s CAM results are unreliable and inconsistent even for same test images despite the similar accuracies of the models. On the other
hand, LFI-CAM results in much more consistent CAM images with better visual quality. (a)(c) ABN on STL10 (a) and Cat&Dog (c), (b)(d) LFI-CAM on STL10 (b) and Cat&Dog (d). 

![overview image](./asset/IOU_Test.png)

> Stability evaluation of visual explanation. (a) IoU between models per dataset, (b) Average IoU per dataset.

## IoU Calculation
1. Assuming you have trained two models to compare, create attention npy files
```
 python cifar.py -a resnet --depth 110 --dataset cifar100 --epochs 1 \
 --evaluate --resume /base/path/checkpoint.pth.tar --gpu-id 0
```
2. Compare attention values
```
python  utils/metrics.py  --att-base /base/path/att --att-target /target/path/att --threshold 100

```

