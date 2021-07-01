from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from dataset import *
import time
import numpy as np
import torchvision.utils as vutils
from torch.autograd import Variable
from networks import *
import math
from math import log10
import torchvision
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import torchvision.transforms as transforms

import skimage
from skimage import io,transform
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

parser = argparse.ArgumentParser()
parser.add_argument("--ngf", type=int, default=64, help="dim of the latent code, Default=512")
parser.add_argument("--ndf", type=int, default=64, help="dim of the latent code, Default=512")
parser.add_argument('--nrow', type=int, help='the number of images in each row', default=8)
parser.add_argument('--trainfiles', default="celeba_hq_attr.list", type=str, help='the list of training files')
parser.add_argument('--trainroot', default="/home/huaibo.huang/data/celeba-hq/celeba-hq-wx-256", type=str, help='path to dataset')
parser.add_argument('--testfiles', default="celeba_hq_attr.list", type=str, help='the list of training files')
parser.add_argument('--testroot', default="/home/huaibo.huang/data/celeba-hq/celeba-hq-wx-256", type=str, help='path to dataset')
parser.add_argument('--trainsize', type=int, help='number of training data', default=28000)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--input_height', type=int, default=128, help='the height  of the input image to network')
parser.add_argument('--input_width', type=int, default=None, help='the width  of the input image to network')
parser.add_argument('--output_height', type=int, default=128, help='the height  of the output image to network')
parser.add_argument('--output_width', type=int, default=None, help='the width  of the output image to network')
parser.add_argument('--crop_height', type=int, default=None, help='the width  of the output image to network')
parser.add_argument('--crop_width', type=int, default=None, help='the width  of the output image to network')
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--start_epoch", default=0, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--cur_iter", default=0, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='results/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--tensorboard', action='store_true', help='enables tensorboard')
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")

str_to_list = lambda x: [int(xi) for xi in x.split(',')]

def readlinesFromFile(path):
    print("Load from file %s" % path)        
    f=open(path)
    data = []
    for line in f:
      content = line.split()        
      data.append(content[0])      
          
    f.close()  
    return data
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    

def time2str(t):
    t = int(t)
    day = t // 86400
    hour = t % 86400 // 3600
    minute = t % 3600 // 60
    second = t % 60
    return "{:02d}/{:02d}/{:02d}/{:02d}".format(day, hour, minute, second)    

def compute_metrics(img, gt):
    img = img.numpy().transpose((0, 2, 3, 1))
    gt = gt.numpy().transpose((0, 2, 3, 1)) 
    img = img[0,:,:,:] * 255.
    gt = gt[0,:,:,:] * 255.
    img = np.array(img, dtype = 'uint8')
    gt = np.array(gt, dtype = 'uint8')
    gt = skimage.color.rgb2ycbcr(gt)[:,:,0]
    img = skimage.color.rgb2ycbcr(img)[:,:,0] 
    cur_psnr = compare_psnr(img, gt, data_range=255)
    cur_ssim = compare_ssim(img, gt, data_range=255)
    return cur_psnr, cur_ssim     

def main():
    
    global opt, model
    opt = parser.parse_args()
    print(opt)    

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        
    is_scale_back = False
    
    #--------------build models -------------------------
    model = WaveletModel(ngf=opt.ngf, ndf=opt.ndf).cuda()    
    if opt.pretrained:
        opt.start_epoch, opt.cur_iter = load_model(model, opt.pretrained)
    print(model)    
    
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    
    #-----------------load dataset--------------------------    
    test_list = readlinesFromFile(opt.testfiles)   
    assert len(test_list) > 0
               
    test_set = ImageDatasetFromFile(test_list, opt.testroot, crop_height=None, output_height=None, is_random_crop=False, is_mirror=False, normalize=None)     
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
        
    start_time = time.time()
        
    def test():        
        psnrs = AverageMeter()
        ssims = AverageMeter()
        
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(test_data_loader, 0):
                
                data, label = batch
                if len(data.size()) == 3:
                    data, label = data.unsqueeze(0), label.unsqueeze(0)
                                 
                data, label = Variable(data).cuda(), Variable(label).cuda()
               
                fake, _, _ = model.generate(data)
                
                cur_psnr, cur_ssim = compute_metrics(fake.data.cpu(), label.data.cpu())
                psnrs.update(cur_psnr)
                ssims.update(cur_ssim)
                               
               
                info = "\n====> Test: time: {}: Iter: {}, ".format(time2str(time.time()-start_time), iteration)                    
                info += 'PSNR: {:.5f}({:.5f}), SSIM: {:.5f}({:.5f}), '.format(psnrs.val, psnrs.avg, ssims.val, ssims.avg)
                print(info, flush=True)
                
                vutils.save_image(fake.data.cpu(), '{}/fake-{:03d}.png'.format(opt.outf, iteration))
                                       
              
        print(' * Total PSNR: {:.5f}, SSIM: {:.5f}, '
          .format(psnrs.avg, ssims.avg))   
        
    test()
           
def load_model(model, pretrained, strict=False):
    state = torch.load(pretrained)
    model.G.load_state_dict(state['G'], strict=strict)    
    return state['epoch'], state['iter']

if __name__ == "__main__":
    main()    