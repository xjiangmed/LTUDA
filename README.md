# LTUDA

This repository contains the official PyTorch implementation of the following paper:

**Labeled-to-Unlabeled Distribution Alignment for Partially-Supervised Multi-Organ Medical Image Segmentation**

Xixi Jiang, Dong Zhang, Xiang Li, Kangyi Liu, Kwang-Ting Cheng, Xin Yang

## Abstract
<p align="justify">
Partially-supervised multi-organ medical image segmentation aims to develop a unified semantic segmentation model by utilizing multiple partially-labeled datasets, with each dataset providing labels for a single class of organs. However, the limited availability of labeled foreground organs and the absence of supervision to distinguish unlabeled foreground organs from the background pose a significant challenge, which leads to a distribution mismatch between labeled and unlabeled pixels. Although existing pseudo-labeling methods can be employed to learn from both labeled and unlabeled pixels, they are prone to performance degradation in this task, as they rely on the assumption that labeled and unlabeled pixels have the same distribution. In this paper, to address the problem of distribution mismatch, we propose a labeled-to-unlabeled distribution alignment (LTUDA) framework that aligns feature distributions and enhances discriminative capability. Specifically, we introduce a cross-set data augmentation strategy, which performs region-level mixing between labeled and unlabeled organs to reduce distribution discrepancy and enrich the training set. Besides, we propose a prototype-based distribution alignment method that implicitly reduces intra-class variation and increases the separation between the unlabeled foreground and background. This can be achieved by encouraging consistency between the outputs of two prototype classifiers and a linear classifier.Extensive experimental results on the AbdomenCT-1K dataset and a union of four benchmark datasets (including LiTS, MSD-Spleen, KiTS, and NIH82) demonstrate that our method outperforms the state-of-the-art partially-supervised methods by a considerable margin, and even surpasses the fully-supervised methods

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
bash env_setup.sh
```

## Usage
### Data preparation
- Toy dataset: partially labeled images are sampled from AbdomenCT-1K.
- Partially labeled dataset: a union of four benchmark datasets (LiTS, MSD-Spleen, KiTS and NIH82).
We evaluate the performance of the multi-organ segmentation model trained on partially labeled dataset on two external datasets, BTCV and AbdomenCT-1K.

Dataset | source
--- | :---:
LiTS | [data](https://competitions.codalab.org/competitions/17094)
MSD-Spleen | [data](http://medicaldecathlon.com/)
KiTS | [data](https://kits19.grand-challenge.org/data/)
NIH82 | [data](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT)
BTCV  | [data](https://www.synapse.org/#!Synapse:syn3193805/wiki/217753)
AbdomenCT-1K | [data](https://zenodo.org/record/7860267#.ZFEMBnZBy3A)
- For data preprocessing, the 3D CT volume is first sliced along the z-axis direction, and irrelevant non-abdominal regions are removed, and finally the axial slice is adjusted to a size of 256×256 pixels.
- A preprocessed toy dataset can be downloaded from the [link](https://drive.google.com/file/d/1d9Y6zJoYXG8Anzug3oeERYLK-4mu69XN/view?usp=sharing). 
- Partially labeled datasets can be processed using the same steps, we do not provide the processed images due to large data size. Here, we do not consider the segmentation of the tumor region, so the tumor in the LiTS dataset is treated as liver, and the tumor in the KiTS dataset is treated as the background. This treatment is based on the consideration that the tumors in the liver are all inside, while the tumors in the kidney have largely destroyed the integrity and continuity of the entire organ.

### Train 
- Stage1: cross-set data augmentation:
```
python train_CDA.py --save_model_path ../checkpoint/CDA --model unet 
```
- Stage2: prototype-based distribution alignment
```
python train_CDA_PDA.py --save_model_path ../checkpoint/CDA_PDA --model unet_proto --reload_path '../checkpoint/CDA/model_best.pth' 
```
The models trained on toy dataset are available [here](https://drive.google.com/file/d/1YGoWS8bFAUYmxjP9cRjMq8C37C0npuyA/view?usp=sharing).
The models trained on Partially labeled dataset are available [here](https://drive.google.com/file/d/1nsz5GEJtkwcw0FYw8x7NpmL2KX-D2-5l/view?usp=sharing).


### Test
- To test the model, please run:
```
python test.py --model unet_proto --reload_path '../checkpoint/CDA_PDA/ema_model_best.pth'
```
When testing, you can choose linear classifier, labeled prototype classifier or unlabeled prototype classifier.

## Qualitative results
TSNE feature visualization
![image](https://github.com/xjiangmed/LTUDA/blob/main/imgs/tsne.png)
## Quantitative results
Quantitative results on the toy dataset
![image](https://github.com/xjiangmed/LTUDA/blob/main/imgs/toy_results.jpg)

<!-- ## Citation
If our paper is useful for your research, please cite:
```
@article{ ,
      title={Labeled-to-Unlabeled Distribution Alignment for Partially-Supervised Multi-Organ Medical Image Segmentation}, 
      author={Xixi Jiang and Dong Zhang and Xiang Li and Kangyi Liu and Kwang-Ting Cheng and Xin Yang},
      journal={arXiv},
      year={2023}
}
``` -->

## Implementation reference
- [ProtoSeg](https://github.com/tfzhou/ProtoSeg)
- [DoDNet](https://github.com/jianpengz/DoDNet)
- [FixMatch_pytorch](https://github.com/valencebond/FixMatch_pytorch)



