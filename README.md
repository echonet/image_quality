# EchoNet-Quality: Denoising Echocardiograms via Deep Generative Modeling of Ultrasound Noise

This is the official repository for the following paper (ICCV 2025 Workshop CVAMD):

> **EchoNet-Quality: Denoising Echocardiograms via Deep Generative Modeling of Ultrasound Noise**<br>
> https://arxiv.org/abs/2505.00043
>
> **Abstract:** *Echocardiography (echo), or cardiac ultrasound, is the most widely used imaging modality for cardiac form and function due to its relatively low cost, rapid acquisition time, and non-invasive nature. However, ultrasound acquisitions are often limited by artifacts and noise that hinder diagnostic interpretation in clinical settings. Existing methodologies for denoising echos consist solely of traditional filtering-based algorithms or deep learning methods developed on radio-frequency (RF) signals which prevents clinical applicability and scalability. To address these limitations, we introduce the first deep generative model capable of simulating ultrasound noise developed on B-mode data. Using this generative model, we develop a synthetic dataset of paired clean and noisy echo images to train a downstream model for real-world image denoising and demonstrate state-of-the-art performance in both internal and external experiments. In both held-out test sets, our method results in echo images with higher gCNR in comparison to noisy image counterparts and images derived from a comparable method which is consistent with provided visual comparisons. Our experiments showcase the potential of our method for future clinical use to improve the quality of echo acquisitions..*


## Noise Simulation
![](figures/noise_simulation.png)

## Image Denoising
| Physician-Labeled Noisy           | Denoised                            |
|--------------------------------- |------------------------------------    |
| ![](figures/EXAMPLE_1_NOISY.png) | ![](figures/EXAMPLE_1_DENOISED.png) |
| ![](figures/EXAMPLE_2_NOISY.png) | ![](figures/EXAMPLE_2_DENOISED.png) |
| ![](figures/EXAMPLE_3_NOISY.png) | ![](figures/EXAMPLE_3_DENOISED.png) |

## Inference
1. Download model weights from release.

2. For denoising, run the following command:
```
python3 denoise.py --unet [...] --input [...] --output [...]
```
- `unet`: path to pretrained weights for U-Net (.pt)
- `input`: path to folder containing noisy A4C echo images (.png)
- `output`: path to folder that will store denoised A4C echo images (.png)

3. For noise simulation, run the following command:
```
python3 simulate_noise.py --gan [...] --encoder [...] --input [...] --output [...] --global [...] --center_field [...] --near_field [...]
```
- `gan`: path to pretrained weights for StyleGAN (.pkl)
- `encoder`: path to pretrained weights for Encoder4Editing (.pt)
- `input`: path to file containing a clean A4C echo image (.png)
- `output`: path to folder that will store noise simulation results (.png)
- `global`: integer that controls the extent of global noise
- `center_field`: integer that controls the extent of center-field noise
- `near_field`: integer that controls the extent near-field noise

## Acknowledgements
This repository builds upon [StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch) and [e4e](https://github.com/omertov/encoder4editing).
