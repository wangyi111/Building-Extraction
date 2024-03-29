# Building -Extraction

This repository contains several implementation demos of deep learning based building extraction with tensorflow.

1. [**Single-pixel based segmentation**](./0-old/): simple CNN & Residual Network.

2. [**Fully convolutional network**](./1-FCN/), based on the paper [Fully Convolutional Networks for Semantic Segmentation](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html).

3. [**U-Net**](./2-U-Net/), based on the paper [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).


### Data
A small example data from [ISPRS](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx) is provided in `data/vaihingen-trial-data-raw`. A cropping script is provided as `data/crop_tile.py`, which can crop a big tile to small patches.
