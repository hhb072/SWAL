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
from math import log10
import torchvision
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import torchvision.transforms as transforms
import skimage
from skimage import io,transform
# from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
# from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--lr_d', type=float, default=0.0001, help='learning rate of the encoder, default=0.0002')
parser.add_argument('--lr_g', type=float, default=0.0001, help='learning rate of the generator, default=0.0002')
parser.add_argument("--weight_align", type=float, default=1e-5, help="Default=1.0")
parser.add_argument("--weight_aug", type=float, default=1e-2, help="Default=1.0")
parser.add_argument("--weight_adv", type=float, default=1e-3, help="Default=1.0")
parser.add_argument("--ngf", type=int, default=16, help="dim of the latent code, Default=512")
parser.add_argument("--ndf", type=int, default=64, help="dim of the latent code, Default=512")
parser.add_argument("--save_iter", type=int, default=1, help="Default=1")
parser.add_argument("--test_iter", type=int, default=1, help="Default=1")
parser.add_argument('--num_mse', type=int, help='the number of images in each row', default=100000)
parser.add_argument('--nrow', type=int, help='the number of images in each row', default=8)
parser.add_argument('--trainfiles', default="train.list", type=str, help='the list of training files')
parser.add_argument('--trainroot', default="trainroot", type=str, help='path to dataset')
parser.add_argument('--testfiles', default="test.list", type=str, help='the list of training files')
parser.add_argument('--testroot', default="testroot", type=str, help='path to dataset')
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

def main():
    
    global opt, model
    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

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

       
    opt.cur_iter = 0
        
    #--------------build models -------------------------
    model = WaveletModel(ngf=opt.ngf, ndf=opt.ndf).cuda()    
    print(model)
    if opt.pretrained:
        opt.start_epoch, opt.cur_iter = load_model(model, opt.pretrained)
    
    optimizerD = optim.Adam(filter(lambda p: p.requires_grad, model.D.parameters()), lr=opt.lr_d)
    optimizerG = optim.Adam(filter(lambda p: p.requires_grad, model.G.parameters()), lr=opt.lr_g)
    
    #-----------------load dataset--------------------------
    train_list = readlinesFromFile(opt.trainfiles)   
    assert len(train_list) > 0
              
    train_set = ImageDatasetFromFile(train_list, opt.trainroot, crop_height=opt.output_height, output_height=opt.output_height, is_random_crop=True, is_mirror=True, normalize=None)    
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
            
    start_time = time.time()
       
    def train_mse(epoch): 
        model.train()
        lossesR = AverageMeter()
           
        
        for iteration, batch in enumerate(train_data_loader, 0):    
            data, label = batch
                
            batch_size = data.shape[0]
                        
            data, label = Variable(data).cuda(), Variable(label).cuda()  
                          
            #=========== Update batchSize ==================  
            fake, att, res = model.generate(data)
                                            
            lossR = model.get_l1_loss(fake, label)                        
           
            
            lossG = lossR 
            
            #=========== Backward && Update =================
            lossesR.update(lossR.data, batch_size)
                                   
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
            
            #========= Print loss =================
            if iteration % 10 == 0:
                info = "\n====> Cur_iter: [{}]: Epoch[{}]({}/{}): time: {}: ".format(opt.cur_iter, epoch, iteration, len(train_data_loader), time2str(time.time()-start_time))
                info += 'L1: {:.5f}({:.5f}), '.format(lossesR.val, lossesR.avg)                           
                print(info, flush=True)
                        
            opt.cur_iter += 1         
    
    def train_gan(epoch): 
        model.train()
        lossesR = AverageMeter()
        lossesR_ = AverageMeter()
        lossesF = AverageMeter()   
        
        for iteration, batch in enumerate(train_data_loader, 0):    
            data, label = batch
                
            batch_size = data.shape[0]
            
            data, label = Variable(data).cuda(), Variable(label).cuda()  
            
            fake, att, res = model.generate(data)
            
            alpha = random.random() * 0.5
            rec, _, res_ = model.generate(label*alpha+data*(1-alpha))
                                  
            lossR = model.get_l1_loss(fake, label)                        
            lossR_ = model.get_l1_loss(rec, label)
            
            lossF = 0
            for r0, r1 in zip(res, res_):
                lossF += model.get_l1_loss(r0,r1.detach())    
            
            lossG = lossR + lossR_ * opt.weight_aug + lossF * opt.weight_align #
            
            #=========== Update D ================== 
            realD = model.discriminate(torch.cat((data, label), dim=1))
            fakeD = model.discriminate(torch.cat((data, fake.detach()), dim=1))
            
            lossD_real = model.get_ls_loss(realD, 1.0)
            lossD_fake = model.get_ls_loss(fakeD, 0.0)
            
            lossD = lossD_real + lossD_fake
            
            optimizerD.zero_grad()
            lossD.backward()
            optimizerD.step()
            
            #========== Update G ===================
            fakeG = model.discriminate(torch.cat((data, fake), dim=1))
            
            lossG_fake = model.get_ls_loss(fakeG, 1.0)
                                    
            lossG += lossG_fake * opt.weight_adv
            
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
            
            lossesR.update(lossR.data, batch_size)
            lossesR_.update(lossR_.data, batch_size)
            lossesF.update(lossF.data, batch_size)
            
            #========= Print loss =================
            if iteration % 10 == 0:
                info = "\n====> Cur_iter: [{}]: Epoch[{}]({}/{}): time: {}: ".format(opt.cur_iter, epoch, iteration, len(train_data_loader), time2str(time.time()-start_time))
                info += 'L1: {:.5f}({:.5f}), '.format(lossesR.val, lossesR.avg)
                info += 'L1R: {:.5f}({:.5f}), '.format(lossesR_.val, lossesR_.avg)
                info += 'F: {:.5f}({:.5f}), '.format(lossesF.val, lossesF.avg)        
                info += 'Adv: {:.3f}, {:.3f}, {:.4f}, '.format(lossD_real.data, lossD_fake.data, lossG_fake.data)
                print(info, flush=True)
                                   
            opt.cur_iter += 1         
       
    #----------------Train by epochs--------------------------    
    model.train()
    for epoch in range(opt.start_epoch, opt.nEpochs + 1): 
        if opt.cur_iter < opt.num_mse:
            train_mse(epoch)
        else:
            train_gan(epoch)
        
        if epoch % opt.save_iter == 0:
            save_epoch = (epoch//opt.save_iter)*opt.save_iter   
            save_checkpoint(model, save_epoch, opt.cur_iter, '', opt.manualSeed)
                  
def load_model(model, pretrained, strict=False):
    state = torch.load(pretrained)
    model.G.load_state_dict(state['G'], strict=strict)   
    return state['epoch'], state['iter']
            
def save_checkpoint(model, epoch, iteration, prefix="", manualSeed=0):
    model_out_path = "model/" + prefix +"model_epoch_{}_iter_{}_seed_{}.pth".format(epoch, iteration, manualSeed)
    state = {"epoch": epoch, "iter": iteration, "G": model.G.state_dict(), "D": model.D.state_dict()}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    main()    
