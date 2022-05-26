# ATP-Net
Adaptive Trilinear Pooling Network for Fine-Grained Visual Categorization



## Requirements
- python 3.8
- pytorch 1.7.0

## Train

Step 1. 
- Download the resnet-50 pre-training parameters.
[resnet50-parameters-download](https://download.pytorch.org/models/resnet50-19c8e357.pth)


- Download the CUB-200-2011 dataset.
[CUB-download](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)

Step 2. 
- Set the path to the dataset and resnet parameters in the code.

Step 3. Train the fc_layer and proj-layer only.
- python train_firststep.py

Step 4. Fine-tune all layers. It gets an accuracy of around 87% on CUB-200-2011 when using resnet-34.
- python train_finetune.py

