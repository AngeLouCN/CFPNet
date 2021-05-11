# CFPNet: Channel-Wise Feature Pyramid for Real-Time Semantic Segmentation
This project contains the Pytorch implementation for the proposed CFPNet: [**paper**](https://arxiv.org/ftp/arxiv/papers/2103/2103.12212.pdf)

<div align=center><img src="https://github.com/AngeLouCN/CFPNet/blob/main/figures/seg_model.png" width="2000" height="250" alt="Result"/></div>
<div align=center><img src="https://github.com/AngeLouCN/CFPNet/blob/main/figures/cfp module.png" width="800" height="300" alt="Result"/></div>
Real-time semantic segmentation is playing a more important role in computer vision, due to the growing demand for mobile devices and autonomous driving. Therefore, it is very important to achieve a good trade-off among performance, model size and inference speed. In this paper, we propose a Channel-wise Feature Pyramid (CFP) module to balance those factors. Based on the CFP module, we built CFPNet for real-time semantic segmentation which applied a series of dilated convolution channels to extract effective features. Experiments on Cityscapes and CamVid datasets show that the proposed CFPNet achieves an effective combination of those factors. For the Cityscapes test dataset, CFPNet achievse 70.1% class-wise mIoU with only 0.55 million parameters and 2.5 MB memory. The inference speed can reach 30 FPS on a single RTX 2080Ti GPU (GPU usage 60%) with a 1024×2048-pixel image.

## Installation
- Enviroment: Python 3.6; Pytorch 1.0; CUDA 9.0; cuDNN V7
- Install some packages:
```
pip install opencv-python pillow numpy matplotlib
```
- Clone this repository
```
git clone https://github.com/AngeLouCN/CFPNet
```
- One GPU with 11GB memory is needed

## Dataset
You need to download the two dataset——CamVid and Cityscapes, and put the files in the ```dataset```folder with following structure.
```
|—— camvid
|    ├── train
|    ├── test
|    ├── val 
|    ├── trainannot
|    ├── testannot
|    ├── valannot
|    ├── camvid_trainval_list.txt
|    ├── camvid_train_list.txt
|    ├── camvid_test_list.txt
|    └── camvid_val_list.txt
├── cityscapes
|    ├── gtCoarse
|    ├── gtFine
|    ├── leftImg8bit
|    ├── cityscapes_trainval_list.txt
|    ├── cityscapes_train_list.txt
|    ├── cityscapes_test_list.txt
|    └── cityscapes_val_list.txt  
```
## Training
- You can run: ```python train.py -h```to check the detail of optional arguments. In the ```train.py```, you can set the dataset, train type, epochs and batch size, etc.
- training on Cityscapes train set.  
```
python train.py --dataset cityscapes
```
- training on Camvid train and val set.  
```
python train.py --dataset camvid --train_type trainval --max_epochs 1000 --lr 1e-3 --batch_size 16
```
- During training course, every 50 epochs, we will record the mean IoU of train set, validation set and training loss to draw a plot, so you can check whether the training process is normal.  

| Val mIoU vs Epochs | Train loss vs Epochs |
| :---: | :---: |
|<div align=center><img src="https://github.com/AngeLouCN/CFPNet/blob/main/figures/iou_vs_epochs.png" width="600" height="300" alt="Result"/></div>|<div align=center><img src="https://github.com/AngeLouCN/CFPNet/blob/main/figures/loss_vs_epochs.png" width="600" height="300" alt="Result"/></div>|

## Testing
- After training, the checkpoint will be saved at ```checkpoint```folder, you can use ```test.py```to predict the result.
```
python test.py --dataset ${camvid, cityscapes} --checkpoint ${CHECKPOINT_FILE}
```

## Evalution
- For those dataset that do not provide label on the test set (e.g. Cityscapes), you can use ```predict.py``` to save all the output images, then submit to official webpage for evaluation.
```
python test.py --dataset ${camvid, cityscapes} --checkpoint ${CHECKPOINT_FILE}
```

## Inference Speed
- You can run the ```eval_fps.py``` to test the model inference speed, input the image size such as ```1024,2048```.
```
python eval_fps.py 1024,2048
```
## Results
- Results for CFPNet-V1, CFPNet-V2 and CFPNet-v3:

| Dataset | Model | mIoU |
| :---: | :---: | :---: |
| Cityscapes | CFPNet-V1 | 60.4% |
| Cityscapes | CFPNet-V2 | 66.5% |
| Cityscapes | CFPNet-V3 | 70.1% |

- Sample results: (from top to bottom is Original, CFPNet-V1, CFPNet-V2 and CFPNet-v3)
<div align=center><img src="https://github.com/AngeLouCN/CFPNet/blob/main/figures/sample_result.png" width="784" height="462" alt="Result"/></div>

| Category_acc vs size | Class_acc vs size |
| :---: | :---: |
|<div align=center><img src="https://github.com/AngeLouCN/CFPNet/blob/main/figures/category_acc_size.png" width="600" height="275" alt="Result"/></div>|<div align=center><img src="https://github.com/AngeLouCN/CFPNet/blob/main/figures/class_acc_size.png" width="600" height="275" alt="Result"/></div>|
| Class_acc vs parameter| Class_acc vs speed|
|<div align=center><img src="https://github.com/AngeLouCN/CFPNet/blob/main/figures/class_acc_param.png" width="600" height="275" alt="Result"/></div>|<div align=center><img src="https://github.com/AngeLouCN/CFPNet/blob/main/figures/acc_speed.png" width="600" height="275" alt="Result"/></div>|

## Comparsion
- Results of Cityscapes
<div align=center><img src="https://github.com/AngeLouCN/CFPNet/blob/main/figures/seg_result.png" width="1000" height="350" alt="Result"/></div>

- Results of CamVid
<div align=center><img src="https://github.com/AngeLouCN/CFPNet/blob/main/figures/seg_results_camvid.png" width="500" height="200" alt="Result"/></div>

## Citation
If you think our work is helpful, please consider to cite:

```
@article{lou2021cfpnet,
  title={CFPNet: Channel-wise Feature Pyramid for Real-Time Semantic Segmentation},
  author={Lou, Ange and Loew, Murray},
  journal={arXiv preprint arXiv:2103.12212},
  year={2021}
}
```
