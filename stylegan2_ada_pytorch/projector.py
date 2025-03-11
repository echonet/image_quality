# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from time import perf_counter
from tqdm import tqdm 
import click
import time
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy

def project_video(
    G,
    target: torch.Tensor, # (F, 1, 256, 256) and dynamic range [0,255]
    n_frames,
    *,
    num_steps                  = 2500,
    initial_learning_rate      = 0.3,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (n_frames, G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    z_samples = np.random.RandomState(123).randn(n_frames, G.z_dim)
    # (F, 512)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)
    # (F, 14, 512)
    w_samples = w_samples[:, :1, :]
    # (F, 1, 512)

    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    target_images = target.repeat(1, 3, 1, 1).to(device).to(torch.float32)
    # (F, 3, 256, 256)
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)
    # (F, 7995392)
    w_opt = torch.tensor(w_samples, dtype=torch.float32, device=device, requires_grad=True) 
    # (F, 1, 512)

    optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)
    lr = initial_learning_rate

    for step in range(num_steps):
        if step >= 750:
            if step % 10 == 0:
                lr = lr * 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr 

        w_s = w_opt.repeat(1, G.mapping.num_ws, 1)
        # (F, 14, 512)

        w = w_s[:1,:,:]
        # (1, 14, 512)
        synth_image = G.synthesis(w, noise_mode="const").repeat(1, 3, 1, 1)
        synth_image = (synth_image + 1) * (255/2)
        # (1, 3, 256, 256)
        synth_images = synth_image
        # (F, 3, 256, 256)

        for frame in range(1, n_frames):
            w = w_s[frame:frame+1, :, :]
            synth_image = G.synthesis(w, noise_mode="const").repeat(1, 3, 1, 1)
            synth_image = (synth_image + 1) * (255/2)
            synth_images = torch.cat((synth_images, synth_image))

        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        # (F, 7995392)

        _target_images = target_images[:, :1, :, :].flatten(start_dim=1)
        # (F, 65536)
        _synth_images = synth_images[:, :1, :, :].flatten(start_dim=1)
        # (F, 65536)

        # 1, 2.5e7
        reconstruction_loss = 1 * (_target_images - _synth_images).flatten().square().mean()
        perceptual_loss = 2e7  * (target_features - synth_features).flatten().square().mean()

        temporal_reconstruction_loss = 1.5 * ((_target_images[1:, :] - _target_images[:-1, :]) - (_synth_images[1:, :] - _synth_images[:-1, :])).flatten().square().mean()

        temporal_perceptual_loss = 0 * ((target_features[1:, :] - target_features[:-1, :]) 
                                        - (synth_features[1:, :] - synth_features[:-1, :])).flatten().square().mean()

        loss = reconstruction_loss + perceptual_loss + temporal_reconstruction_loss + temporal_perceptual_loss

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f"step {step+1:>4d}/{num_steps}: loss {loss:.2f}, l_r {reconstruction_loss:.2f}, l_p {perceptual_loss:.2f}, l_tr {temporal_reconstruction_loss:.2f}, l_tp {temporal_perceptual_loss:.2f}, lr {lr:.2f}")

    return w_opt.detach().repeat(1, G.mapping.num_ws, 1)































def project(
    G,
    target: torch.Tensor, # (F, 1, 256, 256)
    *,
    num_steps                  = 2500,
    w_avg_samples              = 1,
    initial_learning_rate      = 0.3,
    verbose                    = False,
    device: torch.device
):

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.repeat(3, 1, 1).unsqueeze(0).to(device).to(torch.float32)
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)
    lr = initial_learning_rate

    for step in range(num_steps):
        # Learning rate schedule.
        if step >= 2250:
            if step % 10 == 0:
                lr = lr * 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr 

        # Synth images from opt_w.
        ws = (w_opt).repeat([1, G.mapping.num_ws, 1])

        synth_images = G.synthesis(ws, noise_mode='const').repeat(1, 3, 1, 1)
        synth_images = (synth_images + 1) * (255/2)

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)

        # p4.75, c1 - 2
        # p2.75, c1 - 1 
        # p1, c5 - 3

        perceptual_weight = 0.25
        perceptual_loss = (target_features - synth_features).square().sum()
        weighted_perceptual_loss = perceptual_weight * perceptual_loss

        reconstruction_weight = 6
        reconstruction_loss = torch.nn.functional.l1_loss(synth_images, target_images)
        weighted_reconstruction_loss = reconstruction_weight * reconstruction_loss

        loss = weighted_perceptual_loss + weighted_reconstruction_loss

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: loss {loss:<4.2f}, weighted_perceptual_loss {weighted_perceptual_loss:<4.2}, weighted_reconstruction_loss {weighted_reconstruction_loss:<4.2f}, lr {lr:<4.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

    return w_out.repeat([1, G.mapping.num_ws, 1])

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
def run_projection(
    network_pkl: str,
    target_fname: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps:
            synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()

    # Save final projected frame and W vector.
    target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
