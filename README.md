# SWAL
A pytorch implementation of Paper ["Selective Wavelet Attention Learning for Single Image Deraining"](https://link.springer.com/article/10.1007/s11263-020-01421-z)

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
2. Arrange the images and generate a list file, just like in the *test* folder.

You can also modify the data_loader code in your manner.

## Run

Use run.sh to train SWAL and test.sh for evalation.

Adjust *--trainroot*, *--testroot*, *--trainfiles*, and *--testfiles* according to your own settings. 

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
