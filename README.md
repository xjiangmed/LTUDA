# LTUDA

This repository contains the official PyTorch implementation of the following paper:

**Labeled-to-Unlabeled Distribution Alignment for Partially-Supervised Multi-Organ Medical Image Segmentation**

Xixi Jiang, Dong Zhang, Xiang Li, Kangyi Liu, Kwang-Ting Cheng, Xin Yang

## Abstract
<p align="justify">
Partially-supervised multi-organ medical image segmentation aims to learn a unified semantic segmentation model using multiple partially labeled datasets, where each dataset provides labels for a single class of organs. This task is challenging due to the scarcity of labeled foreground organs and the lack of supervision for distinguishing unlabeled foreground organs from background, which leads to distribution mismatch between labeled and unlabeled organs. However, directly applying pseudo-labels may result in the problem of performance degradation, since they rely on the assumption that labeled and unlabeled organs come from the same distribution. In this paper, we propose a labeled-to-unlabeled distribution alignment framework to address distribution mismatch by aligning feature distributions and enhancing discriminative capability. Specifically, we propose a cross-set data augmentation strategy to perform region-level mixing between labeled and unlabeled organs, enriching the training set and reducing distribution discrepancy. Moreover, we propose a prototype-based distribution alignment method that implicitly reduces intra-class variation and increases the separation between unlabeled foreground and background, by encouraging consistency between the outputs of two prototype classifiers and a linear classifier. Experimental results on AbdomenCT-1K demonstrate that our proposed method surpasses fully-supervised method. Besides, results on a union of four benchmark datasets (\ie, LiTS, MSD-Spleen, KiTS and NIH82) validate that our method outperforms state-of-the-art partially-supervised methods by a large margin.

## The overall architecture
![image](https://github.com/xjiang/LTUDA/blob/main/imgs/framework.png)
## Qualitative results
![image](https://github.com/xjiang/LTUDA/blob/main/imgs/figure3.png)
## Quantitative results
![image](https://github.com/xjiang/LTUDA/blob/main/imgs/result.png)

## Installation

- Create conda environment and activate it:
```
conda create -n ltuda python=3.6
conda activate ltuda
```
- Clone this repo:
```
git clone https://github.com/xjiang/LTUDA
cd LTUDA
```
- Install requirements:
```
pip install -r requirements.txt
```

## Usage
### Data preparation
You can download LiTS, MSD-Spleen, KiTS, NIH82 and AbdomenCT-1K from 


### Train 
- To view training results and loss plots, please run:
```
python -m visdom.server -p 6031
```
and click the URL http://localhost:6031

- To train the model, please run:
```
python train3d.py --dataroot ./octa-500/OCT2OCTA3M_3D --name transpro_3M --model TransPro --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch --pool_size 0 --load_size 304 --input_nc 1 --output_nc 1 --display_port 6031 --gpu_ids 0 --no_flip
```

### Test
- To test the model, please run:
```
python test3d.py --dataroot ./octa-500/OCT2OCTA3M_3D --name transpro_3M --test_name transpro_3M --model TransPro --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch --input_nc 1 --output_nc 1 --gpu_ids 0 --num_test 15200 --which_epoch 164 --load_iter 164
```

## Citation
If our paper is useful for your research, please cite:
```
@article{li2023vesselpromoted,
      title={Vessel-Promoted OCT to OCTA Image Translation by Heuristic Contextual Constraints}, 
      author={Shuhan Li and Dong Zhang and Xiaomeng Li and Chubin Ou and Lin An and Yanwu Xu and Kwang-Ting Cheng},
      journal={arXiv},
      year={2023}
}
```

## Implementation reference
[CycleGAN and pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)