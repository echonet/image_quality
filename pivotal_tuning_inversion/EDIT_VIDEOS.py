
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import cv2
import os
import time
from tqdm import tqdm
from configs import paths_config, global_config, hyperparameters
import pickle
from PIL import Image
import warnings
from argparse import ArgumentParser
warnings.filterwarnings("ignore")

def convert_to_grayscale(frame_tnsr):
    frame_arr = frame_tnsr.cpu().numpy().squeeze()
    frame_arr = np.clip(frame_arr, -1.0, 1.0)
    frame_arr = ((frame_arr + 1) * 127.5).astype(np.uint8)
    return frame_arr 

def edit_video(video_uid, edit_strength, pivot_tuned_G, trajectory, dest):
    mp4 = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dest, mp4, 30, (256, 256), False)

    for f in range(hyperparameters.max_images_to_invert):
        frame_w = torch.load(f"{paths_config.experiments_dir}/{video_uid}/embeddings/{video_uid}_F{f}.pt")
        edited_frame_w = frame_w + edit_strength * trajectory 
        edited_frame_tnsr = pivot_tuned_G.synthesis(edited_frame_w, noise_mode="const", force_fp32=True).detach()
        edited_frame = convert_to_grayscale(edited_frame_tnsr)

        writer.write(edited_frame)
    writer.release()

if __name__ == "__main__":
    # ARGUMENTS
    parser = ArgumentParser()
    parser.add_argument("--experiments_dir", dest="experiments_dir", type=str, help="Path to experiments directory")
    args = parser.parse_args()

    # CONFIGURATION
    device = torch.device(global_config.device)
    paths_config.experiments_dir = args.experiments_dir
    video_uids = os.listdir(paths_config.experiments_dir)

    trajectory = torch.load(f"pretrained_models/PAIRED_SYNTHETIC_TRAJECTORY.pt")

    for video_uid in tqdm(video_uids):
        if os.path.exists(f"{paths_config.experiments_dir}/{video_uid}/pivot_tuned_G.pt"):
            with open(f"{paths_config.experiments_dir}/{video_uid}/pivot_tuned_G.pt", 'rb') as pivot_tuned_weights:
                pivot_tuned_G = torch.load(pivot_tuned_weights, weights_only=False).eval().cuda()

            # MAKE FOLDER FOR SPECIFIC TRAJECTORY TYPE
            if not os.path.exists(f"{paths_config.experiments_dir}/{video_uid}/PAIRED_SYNTHETIC_TRAJECTORY"):
                os.makedirs(f"{paths_config.experiments_dir}/{video_uid}/PAIRED_SYNTHETIC_TRAJECTORY")

            for edit_strength in np.arange(0, 1.51, 0.1):
                edit_strength = round(edit_strength, 2)

                edit_video(
                    video_uid,
                    edit_strength,
                    pivot_tuned_G,
                    trajectory,
                    f"{paths_config.experiments_dir}/{video_uid}/PAIRED_SYNTHETIC_TRAJECTORY/edited_{edit_strength}_.mp4"
                )
