# SWAL
[Selective Wavelet Attention Learning for Single Image Deraining](https://link.springer.com/article/10.1007/s11263-020-01421-z)

We propose a Selective Wavelet Attention Learning (SWAL) method by learning a series of wavelet attention maps to guide the separation of rain and background information in both spatial and frequency domains. The key aspect of our method is utilizing wavelet transform to learn the content and structure of rainy features because the high-frequency features are more sensitive to rain degradations, whereas the low-frequency features preserve more of the background content. To begin with, we develop a selective wavelet attention encoder-decoder  network to learn wavelet attention maps guiding the separation of rainy and background features at multiple scales. Meanwhile, we introduce wavelet pooling and unpooling to the encoder-decoder network, which shows superiority in learning increasingly abstract representations while preserving the background details. In addition, we propose latent alignment learning to supervise the background features as well as augment the training data to further improve the accuracy of deraining. Finally, we employ a hierarchical discriminator network based on selective wavelet attention to adversarially improve the visual fidelity of the generated results both globally and locally. Extensive experiments on synthetic and real datasets demonstrate that the proposed approach achieves more appealing results both quantitatively and qualitatively than the recent state-of-the-art methods.

## Prerequisites
* Python 3
* PyTorch

## Run



## Results


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
