
from scripts.run_pti import run_PTI
from configs import paths_config, hyperparameters, global_config
import pandas as pd
import shutil
import torchvision
import torch
import cv2
import os
from tqdm import tqdm 
from argparse import ArgumentParser

def process_tnsr_for_display(tnsr):
  return ((tnsr + 1) * 127.5).squeeze().clamp(0, 255).to(torch.uint8).detach().cpu().numpy()

if __name__ == "__main__":
    # ARGUMENTS
    parser = ArgumentParser()
    parser.add_argument("--experiments_dir", dest="experiments_dir", type=str, help="Path to experiments directory")
    parser.add_argument("--source_dir", dest="source_dir", type=str, help="Path to original videos")
    args = parser.parse_args()

    # CONFIGURATIONS
    paths_config.experiments_dir = args.experiments_dir 
    device = torch.device(global_config.device)
                            
    # VIDEO LOCATION + NAME(S)
    SRC_DIR = args.source_dir 
    VIDEOS = os.listdir(SRC_DIR)

    # TUNING
    for video in tqdm(VIDEOS):
        video_uid = video[:-4]

        # VIDEO CONFIGURATION
        video_dir = f"{paths_config.experiments_dir}/{video_uid}"
        embeddings_dir = f"{video_dir}/embeddings"

        paths_config.input_data_path = f"{video_dir}/frames"
        paths_config.embedding_dir = f"{video_dir}/embeddings"
        paths_config.checkpoint_dir = video_dir

        # FOLDERS
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(paths_config.input_data_path, exist_ok=True)
        os.makedirs(paths_config.embedding_dir, exist_ok=True)

        # TRANSFER ORIGINAL VIDEO
        shutil.copyfile(f"{SRC_DIR}/{video}", f"{video_dir}/video.mp4")

        # EXTRACT FRAMES FROM ORIGINAL VIDEO (PTI WILL BUILD CORRESPONDING LATENT EMBEDDINGS)
        frames, audio, metadata = torchvision.io.read_video(f"{video_dir}/video.mp4", pts_unit="sec", output_format="TCHW")
        frames = frames[:,0,:,:].cpu().numpy()

        for f in range(hyperparameters.max_images_to_invert):
            frame = frames[f]
            cv2.imwrite(f"{paths_config.input_data_path}/{video_uid}_F{f}.png", frame)

        # PIVOT TUNE
        model_id = run_PTI(use_multi_id_training=True)
    
        # BUILD RECONSTRUCTION VIDEO
        with open(f"{paths_config.checkpoint_dir}/pivot_tuned_G.pt", 'rb') as pivot_tuned_weights:
            pivot_tuned_G = torch.load(pivot_tuned_weights).to(device)

        mp4 = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(f"{video_dir}/reconstructed_video.mp4", mp4, 30, (256, 256), False)

        for f in range(hyperparameters.max_images_to_invert):
            frame_w = torch.load(f"{paths_config.embedding_dir}/{video_uid}_F{f}.pt")
            frame_im = pivot_tuned_G.synthesis(frame_w, noise_mode="const", force_fp32=True)
            frame_im = process_tnsr_for_display(frame_im)

            writer.write(frame_im)
        writer.release()



