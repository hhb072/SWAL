import torch
import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
      

    
def load_image(file_path, input_height=None, input_width=None, output_height=None, output_width=None,
              crop_height=None, crop_width=None, is_random_crop=True, is_mirror=True, is_gray=False, align=32):
    
    if input_width is None:
      input_width = input_height
    if output_width is None:
      output_width = output_height
    if crop_width is None:
      crop_width = crop_height
    
    img = Image.open(file_path)
    if is_gray is False and img.mode != 'RGB':
      img = img.convert('RGB')
    if is_gray and img.mode != 'L':
      img = img.convert('L')
    
        
    [w, h] = img.size
    imgR = ImageOps.crop(img, (0, 0, w//2, 0))
    imgL = ImageOps.crop(img, (w//2, 0, 0, 0))
       
    if is_mirror and random.randint(0,1) == 0:
      imgR = ImageOps.mirror(imgR)
      imgL = ImageOps.mirror(imgL)  
      
    if input_height is not None:
      imgR = imgR.resize((input_width, input_height),Image.BICUBIC)
      imgL = imgL.resize((input_width, input_height),Image.BICUBIC)
    
    [w, h] = imgR.size     
    if crop_height is not None:         
      if is_random_crop:
        #print([w,cropSize])        
        cx1 = random.randint(0, w-crop_width) if crop_width < w else 0
        cx2 = w - crop_width - cx1
        cy1 = random.randint(0, h-crop_height) if crop_height < h else 0
        cy2 = h - crop_height - cy1        
      else:
        cx2 = cx1 = int(round((w-crop_width)/2.))
        cy2 = cy1 = int(round((h-crop_height)/2.))
      imgR = ImageOps.crop(imgR, (cx1, cy1, cx2, cy2))
      imgL = ImageOps.crop(imgL, (cx1, cy1, cx2, cy2))          
    if output_height is not None:
      imgR = imgR.resize((output_width, output_height),Image.BICUBIC)
      imgL = imgL.resize((output_width, output_height),Image.BICUBIC)
    
    [w, h] = imgR.size
    h1 = h // align * align
    w1 = w // align * align
    if h1 != h or w1 != w:
        imgR = imgR.resize((w1, h1),Image.BILINEAR)
        imgL = imgL.resize((w1, h1),Image.BILINEAR)
    return imgR, imgL
         
class ImageDatasetFromFile(data.Dataset):
    def __init__(self, image_list, root_path, 
                input_height=None, input_width=None, output_height=None, output_width=None,
                crop_height=None, crop_width=None, is_random_crop=False, is_mirror=True, is_gray=False, normalize=None):
        super(ImageDatasetFromFile, self).__init__()
                
        self.image_filenames = image_list 
        self.is_random_crop = is_random_crop
        self.is_mirror = is_mirror
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.root_path = root_path
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.is_gray = is_gray      
        
        if normalize is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), normalize])

    def __getitem__(self, index):          
          
        imgR, imgL = load_image(join(self.root_path, self.image_filenames[index]), 
                                  self.input_height, self.input_width, self.output_height, self.output_width,
                                  self.crop_height, self.crop_width, self.is_random_crop, self.is_mirror, self.is_gray)
        
        imgR = self.transform(imgR)   
        imgL = self.transform(imgL)        
        
        return imgR, imgL

    def __len__(self):
        return len(self.image_filenames)
      
