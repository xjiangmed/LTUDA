# LTUDA

This repository contains the official PyTorch implementation of the following paper:

**Labeled-to-Unlabeled Distribution Alignment for Partially-Supervised Multi-Organ Medical Image Segmentation**

Xixi Jiang, Dong Zhang, Xiang Li, Kangyi Liu, Kwang-Ting Cheng, Xin Yang

## Abstract
<p align="justify">
Partially-supervised multi-organ medical image segmentation aims to learn a unified semantic segmentation model using multiple partially labeled datasets, where each dataset provides labels for a single class of organs. This task is challenging due to the scarcity of labeled foreground organs and the lack of supervision for distinguishing unlabeled foreground organs from background, which leads to distribution mismatch between labeled and unlabeled organs. However, directly applying pseudo-labels may result in the problem of performance degradation, since they rely on the assumption that labeled and unlabeled organs come from the same distribution. In this paper, we propose a labeled-to-unlabeled distribution alignment framework to address distribution mismatch by aligning feature distributions and enhancing discriminative capability. Specifically, we propose a cross-set data augmentation strategy to perform region-level mixing between labeled and unlabeled organs, enriching the training set and reducing distribution discrepancy. Moreover, we propose a prototype-based distribution alignment method that implicitly reduces intra-class variation and increases the separation between unlabeled foreground and background, by encouraging consistency between the outputs of two prototype classifiers and a linear classifier. Experimental results on AbdomenCT-1K demonstrate that our proposed method surpasses fully-supervised method. Besides, results on a union of four benchmark datasets (i.e., LiTS, MSD-Spleen, KiTS and NIH82) validate that our method outperforms state-of-the-art partially-supervised methods by a large margin.

## The overall architecture
![image](https://github.com/xjiangmed/LTUDA/blob/main/imgs/framework.png)

## Installation

- Create conda environment and activate it:
```
conda create -n ltuda python=3.6
conda activate ltuda
```
- Clone this repo:
```
git clone https://github.com/xjiangmed/LTUDA.git
cd LTUDA
```
- Install requirements:
```
pip install -r requirements.txt
```

## Usage
### Data preparation
- Toy dataset: partially labeled images are sampled from [AbdomenCT-1K](https://zenodo.org/record/7860267#.ZFEMBnZBy3A).
- Partially labeled dataset: a union of four benchmark datasets (LiTS, MSD-Spleen, KiTS and NIH82)

Dataset | source
--- | :---:
LiTS | [data](https://competitions.codalab.org/competitions/17094)
MSD-Spleen | [data](http://medicaldecathlon.com/)
KiTS | [data](https://kits19.grand-challenge.org/data/)
NIH82 | [data](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT)

### Train 
- Stage1: cross-set data augmentation:
```
python train_CDA.py --save_model_path ./checkpoint/CDA --model unet 
```
- Stage2: prototype-based distribution alignment
```
python train_CDA_PDA.py --save_model_path ./checkpoint/CDA_PDA --model unet_proto --reload_path './checkpoint/CDA/model_best.pth' 
```
### Test
- To test the model, please run:
```
python test.py --model unet_proto --reload_path './checkpoint/CDA_PDA/model_best.pth'
```

## Qualitative results
TSNE feature visualization
![image](https://github.com/xjiangmed/LTUDA/blob/main/imgs/tsne.png)
## Quantitative results
**Toy dataset**: partially labeled images are sampled from the AbdomenCT-1K dataset.
![image](https://github.com/xjiangmed/LTUDA/blob/main/imgs/toy_results.jpg)

## Citation
If our paper is useful for your research, please cite:
```
@article{ ,
      title={Labeled-to-Unlabeled Distribution Alignment for Partially-Supervised Multi-Organ Medical Image Segmentation}, 
      author={Xixi Jiang and Dong Zhang and Xiang Li and Kangyi Liu and Kwang-Ting Cheng and Xin Yang},
      journal={arXiv},
      year={2023}
}
```

## Implementation reference
- [ProtoSeg](https://github.com/tfzhou/ProtoSeg)
- [DoDNet](https://github.com/jianpengz/DoDNet)
- [FixMatch_pytorch](https://github.com/valencebond/FixMatch_pytorch)
