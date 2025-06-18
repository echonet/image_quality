import os
import sys
import torch
from PIL import Image

from argparse import ArgumentParser
from torchvision.transforms.functional import to_grayscale, to_tensor, to_pil_image, normalize

sys.path.append("stylespace_analysis")
from stylespace_analysis.manipulate import LoadModel as Load_G
from stylespace_analysis.manipulate import S2List, SL2D, GetSName

sys.path.append("encoder4editing")
from encoder4editing.utils.common import tensor2im
from encoder4editing.utils.model_utils import load_e4e_standalone as Load_E

def get_file_uid(path):
    components = path.split('/')
    file_uid = components[-1]
    return file_uid

def img2tensor(path):
    img = Image.open(args.input)
    img_gray = to_grayscale(img, num_output_channels=3)
    img_tnsr = to_tensor(img_gray)
    img_norm = normalize(img_tnsr, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    img_batch = img_norm.unsqueeze(dim=0)
    return img_batch

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--global", dest="_global", type=int, help="Manipulation of global noise in A4C echocardiographic image.")
    parser.add_argument("--center_field", dest="center_field", type=int, help="Manipulation of center-field noise in A4C echocardiographic image.")
    parser.add_argument("--near_field", dest="near_field", type=int, help="Manipulation of near-field noise in A4C echocardiographic image.")
    parser.add_argument("--input", dest="input", type=str, help="Path to file of a clean A4C echocardiographic image.")
    parser.add_argument("--output", dest="output", type=str, help="Path to folder for storing noise simulation results.")
    parser.add_argument("--encoder", dest="encoder", type=str, help="Path to encoder weights (.pt).")
    parser.add_argument("--gan", dest="gan", type=str, help="Path to StyleGAN weights (.pkl).")
    parser.add_argument("--device", dest="device", default="cuda:0", type=str, help="Device to use for inference.")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(f"{args.output}/clean", exist_ok=True)
    os.makedirs(f"{args.output}/noisy", exist_ok=True)

    G = Load_G(args.gan, args.device) 
    E = Load_E(args.encoder, args.device)

    clean_img_tnsr = img2tensor(args.input).to(args.device)

    with torch.no_grad():
        clean_img_w = E(clean_img_tnsr)
        clean_img_s = G.synthesis.W2S(clean_img_w) 

    clean_slist = S2List(clean_img_s)
    noisy_slist = S2List(clean_img_s)

    noisy_slist[12][:,247] -= args._global
    noisy_slist[9][:,416] -= args.center_field
    noisy_slist[12][:,100] += args.near_field

    clean_sdict = SL2D(clean_slist, GetSName(G), args.device)
    noisy_sdict = SL2D(noisy_slist, GetSName(G), args.device)

    clean_reconstruct = G.synthesis.forward(ws=None, encoded_styles=clean_sdict, noise_mode="const", force_fp32=True)
    noisy_reconstruct = G.synthesis.forward(ws=None, encoded_styles=noisy_sdict, noise_mode="const", force_fp32=True)
    
    clean_img = tensor2im(clean_reconstruct.squeeze())
    noisy_img = tensor2im(noisy_reconstruct.squeeze())

    clean_img.save(f"{args.output}/clean/{get_file_uid(args.input)}")
    noisy_img.save(f"{args.output}/noisy/{get_file_uid(args.input)}")


