import torch
import torch.nn as nn
import functools
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models
import math
import time
import torch.multiprocessing as multiprocessing
from torch.nn.parallel import data_parallel
import torch.nn.utils.spectral_norm as spectral_norm
import scipy.io as sio
import copy
from functools import wraps


class WaveletTransform(nn.Module): 
    def __init__(self, scale=1, dec=True, params_path='model/wavelet.mat', transpose=True, cdim=3):
        super(WaveletTransform, self).__init__()
        
        self.scale = scale
        self.dec = dec
        self.transpose = transpose
        if scale == 0:
            return
            
        self.cdim = cdim
        
        ks = int(2**self.scale)
        nc = cdim * ks * ks
        
        if dec:
          self.conv = nn.Conv2d(in_channels=cdim, out_channels=nc, kernel_size=ks, stride=ks, padding=0, groups=cdim, bias=False)
        else:
          self.conv = nn.ConvTranspose2d(in_channels=nc, out_channels=cdim, kernel_size=ks, stride=ks, padding=0, groups=cdim, bias=False)
        
        # init
        dct = sio.loadmat(params_path)        
        self.conv.weight.data = torch.from_numpy(dct['rec%d' % ks])[:ks*ks].repeat(cdim, 1, 1, 1)       
        self.conv.weight.requires_grad = False
                           
    def forward(self, x): 
        if self.scale == 0:
            return x
        
        if self.dec:
          output = self.conv(x)          
          if self.transpose:
            osz = output.size()
            output = output.view(osz[0], self.cdim, -1, osz[2], osz[3]).transpose(1,2).contiguous().view(osz)            
        else:
          if self.transpose:
            xsz = x.size()
            x = x.view(xsz[0], -1, self.cdim, xsz[2], xsz[3]).transpose(1,2).contiguous().view(xsz)             
          output = self.conv(x)        
        return output 

class _BN_Residual_Block(nn.Module): 
    def __init__(self, in_channels, out_channels, groups=1, act=nn.ReLU(True), bn=False, wider_channel=True, last=True):
        super().__init__()
        
        self.last = last
        mid_channels = out_channels if out_channels > in_channels else in_channels
        
        if in_channels is not out_channels:
            self.expand_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, groups=groups, bias=False)
            self.expand_bn = nn.BatchNorm2d(out_channels) if bn else None
        else:
            self.expand_conv, self.expand_bn = None, None
            
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels) if bn else None
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, 1, 1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if bn else None  
        if not last:
            self.act = act
        
    def forward(self, x):
        if self.expand_conv is not None:
            id = self.expand_conv(x) 
            if (not self.last) and (self.expand_bn is not None):
                id = self.expand_bn(id)
        else:
            id = x
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)        
        if (not self.last) and (self.bn2 is not None):
            x = self.bn2(x) 
        x = x + id
        if not self.last:
            x = self.act(x)  
        return x
        
class WaveletPooling(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True, scale=1, reduction=16, hdim=32):
        super().__init__()
        
        num_wavelets = 4**scale
        self.wavelet_dec = WaveletTransform(scale, dec=True, cdim=in_channels, transpose=True)
        
        self.process = _BN_Residual_Block(in_channels*num_wavelets, out_channels*num_wavelets, groups=num_wavelets)
        
        self.att0 = nn.Sequential(
                        nn.Conv2d(in_channels, hdim, 3, 2, 1, bias=False),                        
                        nn.ReLU(True),
                      )
        for i in range(scale - 1):
            self.att0.add_module('conv_{}'.format(i), nn.Conv2d(out_channels, out_channels, 3, 2, 1, bias=False))
            self.att0.add_module('relu_{}'.format(i), nn.ReLU(True))
            
        self.att1 = nn.Sequential(
                        nn.Conv2d(out_channels*num_wavelets+hdim, hdim, 1, 1, 0, bias=False),                        
                        nn.ReLU(True),
                        nn.Conv2d(hdim, 4, 1, 1, 0, bias=False), 
                        nn.Sigmoid(),                      
                      )
       
                      
        if downsample:
            self.pool = nn.Conv2d(out_channels*num_wavelets, out_channels, 1, 1, 0, bias=False)
        else:
            assert in_channels == out_channels
            self.pool = WaveletTransform(scale, dec=False, cdim=in_channels, transpose=True)
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_() 
        
        eps = torch.cuda.FloatTensor(std.size()).normal_()       
        eps = Variable(eps)
        
        return eps.mul(std).add_(mu)
    
    def forward(self, x, prior_att=None):
        att = self.att0(x)
        x = self.wavelet_dec(x)        
        x = self.process(x)
        if prior_att is None:
            att = self.att1(torch.cat((att, x), dim=1))  
            
        else:
            att = prior_att
        
        out_att = att
        att = att.unsqueeze(2).repeat(1, 1, x.shape[1]//att.shape[1], 1, 1).flatten(1,2)
        x_att = x * att
        x_res = x - x_att
        x_forward = self.pool(x_att)
        return x_forward, x_res, out_att

class WaveletUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True, scale=1, last=False):
        super().__init__()
        
        num_wavelets = 4**scale
        wavelet_channels = out_channels * num_wavelets
        if upsample:
            self.up = nn.Conv2d(in_channels, in_channels*num_wavelets, 1, 1, 0, bias=False)
        else:
            assert in_channels == out_channels
            self.up = WaveletTransform(scale, dec=True, cdim=in_channels)
        self.process = _BN_Residual_Block(in_channels*num_wavelets, out_channels*num_wavelets, groups=num_wavelets, last=last)
        self.wavelet_rec = WaveletTransform(scale, dec=False, cdim=out_channels)
        
        if last:
            self.uncertainty = nn.ConvTranspose2d(in_channels*num_wavelets, 1, 4, 2, 0)
        else:
            self.uncertainty = None
            
        
    def forward(self, x, res):
        x = self.up(x)
        x = x + res
        x = self.process(x)
        x = self.wavelet_rec(x)        
        return x
        
class ResPool(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, num_layers=1, scale=1, reduction=8):
        super().__init__()
        
        self.pool = WaveletPooling(in_channels, out_channels, downsample, reduction=reduction, scale=scale)
        self.res = nn.ModuleList([_BN_Residual_Block(out_channels, out_channels) for i in range(num_layers)])
        
    def forward(self, x, prior_att=None):
        x, res, att = self.pool(x, prior_att)
        for block in self.res:
            x = block(x)
            
        return x, res, att

class ResUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, upsample, num_layers=1, scale=1, last=True):
        super().__init__()
                
        self.res = nn.ModuleList([_BN_Residual_Block(in_channels, in_channels) for i in range(num_layers)])
        self.up = WaveletUpsample(in_channels, out_channels, upsample, last=last, scale=scale)
        
    def forward(self, x, res):        
        for block in self.res:
            x = block(x)
        x = self.up(x, res)    
        return x
        
class Generator(nn.Module):
    def __init__(self, ngf=64):
        super().__init__()
            
        self.down, self.up = [], []
        self.down += [ResPool(3, ngf, True, 16, reduction=2)] # 128
        self.up += [ResUpsample(ngf, 3, True, 16, last=True)] 
                     
        self.down += [ResPool(ngf, ngf*2, True, 16, reduction=4)] # 64
        self.up += [ResUpsample(ngf*2, ngf, True, 16)] 
               
        self.down += [ResPool(ngf*2, ngf*4, True, 16, reduction=8)] # 32
        self.up += [ResUpsample(ngf*4, ngf*2, True, 16)]
               
        self.down += [ResPool(ngf*4, ngf*4, True, 16, reduction=8)] # 16
        self.up += [ResUpsample(ngf*4, ngf*4, True, 16)]
              
        self.down += [ResPool(ngf*4, ngf*4, True, 16, reduction=8)] # 8
        self.up += [ResUpsample(ngf*4, ngf*4, True, 16)]
                         
        self.down, self.up = nn.ModuleList(self.down), nn.ModuleList(self.up[::-1])
                
    def forward(self, x):
        x = x * 2 - 1       
        res = []
        att = []
       
        for down in self.down:
            x, r, a = down(x)
            res.append(r)
            att.append(a.detach())                
                    
        feats = []
        for up, r in zip(self.up, res[::-1]):            
            feats.append(x)
            x = up(x, r)
            
        x = torch.tanh(x) * 0.5 + 0.5  
       
        return x, att, feats
        
class Discriminator(nn.Module):
    def __init__(self, ngf=64):
        super().__init__()
                
        self.down, self.fc = [], []
        self.down += [ResPool(6, ngf, True, 1, reduction=2)] # 128 
        self.fc += [nn.Conv2d(ngf*4, 1, 5, 2, 0)]
        
        self.down += [ResPool(ngf, ngf, True, 1, reduction=2)] # 64
        self.fc += [nn.Conv2d(ngf*4, 1, 5, 2, 0)]
        
        self.down += [ResPool(ngf, ngf, True, 1, reduction=2)] # 32
        self.fc += [nn.Conv2d(ngf*4, 1, 5, 2, 0)]
        
        self.down += [ResPool(ngf, ngf, True, 1, reduction=2)] # 16 
        self.fc += [nn.Conv2d(ngf*4, 1, 5, 2, 0)]
        
        self.down += [ResPool(ngf, ngf, True, 1, reduction=2)] # 8
        self.fc += [nn.Conv2d(ngf*4, 1, 5, 2, 0)]
                           
        self.down = nn.ModuleList(self.down)
        self.fc = nn.ModuleList(self.fc)
                
    def forward(self, x): 
        x = x * 2 - 1
       
        res = []
        out = []
        for down, fc in zip(self.down, self.fc):
            x, r, _ = down(x)
            o = fc(r)
            res.append(r.flatten(1))
            out.append(o.flatten(1))
            
        res = torch.cat(res, dim=1)        
        out = torch.cat(out, dim=1)
        out = torch.sigmoid(out)
        return out

class WaveletModel(nn.Module):
    def __init__(self, ngf=64, ndf=64):
        super().__init__()
        
        self.G = Generator(ngf)
        self.D = Discriminator(ndf)
        
        self.L1 = nn.L1Loss()    
               
    def forward(self, x):
        pass
            
    def generate(self, x):
        return data_parallel(self.G, x)
        
    def discriminate(self, x):
        return data_parallel(self.D, x).mean()
    
    def get_l1_loss(self, x, y):
        return data_parallel(self.L1, (x, y)).mean()
        
    def get_ls_loss(self, x, y):
        return ((x - y)**2).mean()
       

