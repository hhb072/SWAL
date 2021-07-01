# SWAL
Code for Paper ["Selective Wavelet Attention Learning for Single Image Deraining"](https://link.springer.com/article/10.1007/s11263-020-01421-z)

## Prerequisites
* Python 3
* PyTorch

## Models

We provide the models trained on DDN, DID, Rain100H, Rain100L, and AGAN datasets in the following links:

* [Google Driver](https://drive.google.com/drive/folders/1rOuxUmOEHf_6t7-ZhNfrvbwRj-Se_oFA?usp=sharing) 
* [Jianguo Yun](https://www.jianguoyun.com/p/DbB0gXUQiaCuBxi37v0D)

Download them into the *model* folder before testing. 

## Dataset

1. Download the rain datasets.
2. Arrange the images and generate a list file, just like the rain12 set in the *data* folder.

You can also modify the data_loader code in your manner.

## Run

Train SWAL on a single GPU:

	 CUDA_VISIBLE_DEVICES=0 python main.py --ngf=16 --ndf=64  --output_height=320  --trainroot=YOURPATH --trainfiles='YOUR_FILELIST'  --save_iter=1 --batchSize=8 --nrow=8 --lr_d=1e-4 --lr_g=1e-4  --cuda  --nEpochs=500
	 
Train SWAL on multiple GPUs:

	 CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --ngf=16 --ndf=64  --output_height=320  --trainroot=YOURPATH --trainfiles='YOUR_FILELIST'  --save_iter=1 --batchSize=32 --nrow=8 --lr_d=1e-4 --lr_g=1e-4  --cuda  --nEpochs=500	 

Test SWAL:

	 CUDA_VISIBLE_DEVICES=0 python test.py --ngf=16  --outf='test' --testroot='data/rain12_test' --testfiles='data/rain12_test.list' --pretrained='model/rain100l_best.pth'  --cuda

Adjust the parameters according to your own settings. 

## Citation

If you use our codes, please cite the following paper:

	 @article{huang2021selective,
	   title={Selective Wavelet Attention Learning for Single Image Deraining},
	   author={Huang, Huaibo and Yu, Aijing and Chai, Zhenhua and He, Ran and Tan, Tieniu},
	   journal={International Journal of Computer Vision},
	   volume={129},
	   number={4},
	   pages={1282--1300},
	   year={2021},
	  }
 
**The released codes are only allowed for non-commercial use.**
