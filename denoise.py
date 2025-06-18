import os
import torch

import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from argparse import ArgumentParser
from torchvision.transforms.functional import to_grayscale, to_tensor, to_pil_image

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.Mish(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.Mish(inplace=True)
    )   

class UNet(nn.Module):

    def __init__(self):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, 1, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--unet", dest="unet", type=str, help="Path to U-Net weights for denoising A4C echocardiographic images.")
    parser.add_argument("--input", dest="input", type=str, help="Path to folder containing noisy A4C echocardiographic images.")
    parser.add_argument("--output", dest="output", type=str, help="Path to folder for storing denoised A4C echocardiographic images.")
    parser.add_argument("--device", dest="device", default="cuda:0", type=str, help="Device to use for inference.")


    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    unet = UNet().to(args.device)
    unet.load_state_dict(torch.load(args.unet, weights_only=True))
    unet.eval()

    for img_uid in os.listdir(args.input):
        with torch.no_grad():
            img = Image.open(f"{args.input}/{img_uid}")
            img_gray = to_grayscale(img, num_output_channels=1) 
            img_tnsr = to_tensor(img_gray)
            img_batch = img_tnsr.unsqueeze(dim=0)

            out_batch = unet(img_batch.to(args.device))
            out_batch = torch.clip(out_batch, 0, 1).cpu().squeeze(dim=0) * 255
            out_batch = out_batch.to(dtype=torch.uint8)

            out_img = to_pil_image(out_batch, mode='L')
            out_img.save(f"{args.output}/{img_uid}")
