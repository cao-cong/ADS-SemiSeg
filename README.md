# ADS-SemiSeg
This repository contains official implementation of Adversarial Dual-Student with Differentiable Spatial Warping for Semi-Supervised Semantic Segmentation in TCSVT 2022, by Cong Cao, Tianwei Lin, Dongliang He, Fu Li, Huanjing Yue, Jingyu Yang, and Errui Ding. [[arxiv]](https://arxiv.org/abs/2203.02792)

<p align="center">
  <img width="800" src="https://github.com/cao-cong/ADS-SemiSeg/blob/main/images/framework.png">
</p>


## Code

### Dependencies

- Python >= 3.5
- Pytorch >= 1.1

### Test

You can download pretrained weights from [here](https://drive.google.com/drive/folders/1Ch9bUbqToN2hisl3afnCW32qhP12p9SB?usp=sharing) (ADS-DGW_Dataset_SemiRatio_iterXXXXX.pth), then run:
```
bash run_scripts/test_VOC2012.sh
```
### Train
