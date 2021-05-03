# CFPNet: Channel-Wise Feature Pyramid for Real-Time Semantic Segmentation
This project contains the Pytorch implementation for the proposed CFPNet: [**paper**] (https://arxiv.org/abs/2103.12212).

<div align=center><img src="https://github.com/AngeLouCN/CFPNet/blob/main/figures/seg_model.png" width="2000" height="250" alt="Result"/></div>
Real-time semantic segmentation is playing a more important role in computer vision, due to the growing demand for mobile devices and autonomous driving. Therefore, it is very important to achieve a good trade-off among performance, model size and inference speed. In this paper, we propose a Channel-wise Feature Pyramid (CFP) module to balance those factors. Based on the CFP module, we built CFPNet for real-time semantic segmentation which applied a series of dilated convolution channels to extract effective features. Experiments on Cityscapes and CamVid datasets show that the proposed CFPNet achieves an effective combination of those factors. For the Cityscapes test dataset, CFPNet achievse 70.1% class-wise mIoU with only 0.55 million parameters and 2.5 MB memory. The inference speed can reach 30 FPS on a single RTX 2080Ti GPU (GPU usage 60%) with a 1024×2048-pixel image.

## Installation
-Enviroment: Python 3.6; Pytorch 1.0; CUDA 9.0; cuDNN V7
-Install some packages:
```
pip install opencv-python pillow numpy matplotlib
```
-Clone this repository
```
git clone https://github.com/AngeLouCN/CFPNet
```
-One GPU with 11GB memory is needed
<div align=center><img src="https://github.com/AngeLouCN/CFPNet/blob/main/figures/sample_result.png" width="784" height="462" alt="Result"/></div>
This repository contains the implementation of a new version U-Net (DC-UNet) used to segment different types of biomedical images. This is a binary classification task: the neural network predicts if each pixel in the biomedical images is either a region of interests (ROI) or not. The neural network structure is described in this 
