<p align="center">
  <h1 align="center">
   Multi-Label Self-Knowledge Distillation 
  </h1> 
</p>

## What's new
- [2025.03] Add implementation of several other methods to our codebase. 
- [2024.12] Paper accepted to AAAI25.

## Requirements

The repo is tested with:
 
> - numpy==1.21.2
> - Pillow==9.2.0
> - randaugment==1.0.2
> - scikit_learn==1.1.2
> - timm==0.6.7
> - torch==1.8.1
> - torchvision==0.9.1

But it should be runnable with other PyTorch versions.

To install requirements:

```
pip install -r requirements.txt
```

## Quick start

We provide several bash command in [run.sh](run.sh). Simply pick up your expected lines to execute.

You can also try your own distillers and other options by making your own configuration files under the guidance of [Configuration files](#configuration-files).

## Dataset preparation

Your Pascal VOC 2007 dataset folder should be like this:

```
[Any name you want]
  |--VOCtrainval2007
    |--VOCdevkit
      |--VOC2007
        |--JPEGImages
          |--000005.jpg
          |--...
        |--ImageSets
          |--Main
            |--trainval.txt
  |--VOCtest2007
    |--VOCdevkit
      |--VOC2007
        |--JPEGImages
          |--000001.jpg
          |--...
        |--ImageSets
          |--Main
            |--test.txt
```

Your MS-COCO 2014 dataset folder should be like this:

```
[Any name you want]
  |--train2014
    |--COCO_train2014_000000000009.jpg
    |--...
  |--val2014
    |--COCO_val2014_000000000042.jpg
    |--...
  |--train_anno.json
  |--val_anno.json
```

`train_anno.json` and `val_anno.json` are in the fold `./appendix` of [L2D](https://github.com/penghui-yang/L2D)'s repository.

Your NUS-WIDE dataset folder should be like this:

```
[Any name you want]
  |--Flickr
    |--actor
      |--0001_2124494179.jpg
      |--0002_174174086.jpg
      |--...
    |--administrative_assistant
      |--0001_534152430.jpg
      |--0002_258761806.jpg
      |--...
    |--...
  |--ImageList
    |--Imagelist.txt
    |--TrainImagelist.txt
    |--TestImagelist.txt
  |--TrainTestLabels
    |--Labels_airport_Train.txt
    |--Labels_airport_Test.txt
    |--...
  |--Concepts81.txt
```

All codes of the data processing part are in the fold `./data`, and you can replace them with your own code.

## Configuration files

We use configuration files to pass parameters to the program. An example in the fold `./configs` is shown below:

```python
dataset = "coco"
teacher_pretrained = False
img_size = 224
batch_size = 64

model_t = "resnet101"
lr_t = 1e-4
max_epoch_t = 80
stop_epoch_t = 30

model_s = "resnet34"
lr_s = 1e-4
max_epoch_s = 80
stop_epoch_s = 80

criterion_t2s_para = dict(
    name="L2D",
    para=dict(
        lambda_ft=0.0,
        ft_dis=None,
        lambda_le=1.0,
        le_dis=dict(
            name="LED",
            para=dict(
                lambda_cd=100.0,
                lambda_id=1000.0
            )
        ),
        lambda_logits=10.0,
        logits_dis=dict(
            name="MLD",
            para=dict()
        )
    )
)
```

We split a distiller into three parts: feature-based part, label-wise embedding part and logits-based part. Each part has a balancing parameter lambda and corresponding parameters.

It is worth noting that you can set `teacher_pretrained = True` after you have already trained a teacher model and stored its weight parameters in order to avoid repetitive training and save your time.


## Citation
```
@article{Wang_Wang_Zhang_Fang_Wang_2025, 
    title={Multi-label Self Knowledge Distillation}, volume={39}, 
    url={https://ojs.aaai.org/index.php/AAAI/article/view/35433}, 
    DOI={10.1609/aaai.v39i20.35433},  
    author={Wang, Xucong and Wang, Pengkun and Zhang, Shurui and Fang, Miao and Wang, Yang}, 
    year={2025}, 
    month={Apr.}, 
    pages={21330-21338}}
```

## Acknowledgement

This repo is partly based on the following repos, thank the authors a lot.

1. [Penghui-yang/L2D](https://github.com/penghui-yang/L2D)
2. [Alibaba-MIIL/ASL](https://github.com/Alibaba-MIIL/ASL)
3. [Alibaba-MIIL/ML-Decoder](https://github.com/Alibaba-MIIL/ML_Decoder)
4. [wutong16/DistributionBalancedLoss](https://github.com/wutong16/DistributionBalancedLoss)
5. [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
6. [DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
