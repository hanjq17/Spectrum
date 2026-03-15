<div align=center>
  
# [CVPR 2026] Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration

Jiaqi Han* $^1$, Juntong Shi* $^1$, Puheng Li $^1$, Haotian Ye $^1$, Qiushan Guo $^2$, Stefano Ermon $^1$

**$^1$ Stanford University**   **$^2$ ByteDance**   

<p>
<a href='https://arxiv.org/abs/2603.01623'><img src='https://img.shields.io/static/v1?&logo=arxiv&label=Paper&message=Arxiv:Spectrum&color=B31B1B'></a>
<a href='https://hanjq17.github.io/Spectrum/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
</p>

</div>


## 🎯 Overview

![Illustration](assets/illustration.png "Illustration")


We propose *Spectrum*, a training-free *spectral* diffusion feature forecaster that enables global, long-range feature reuse with tightly controlled error. We view the latent features of the denoiser as functions over time and approximate them with *Chebyshev polynomials*. Specifically, we fit the coefficient for each basis via ridge regression, which is then leveraged to forecast features at multiple future diffusion steps. We theoretically reveal that our approach admits more favorable long-horizon behavior and yields an error bound that does not compound with the step size.

Extensive experiments on various state-of-the-art image and video diffusion models consistently verify the superiority of our approach. Notably, we achieve up to $4.79\times$ speedup on FLUX.1 and $4.67\times$ speedup on Wan2.1-14B, while maintaining much higher sample quality compared with the baselines. See more demos on our [Project Page](https://github.com/hanjq17/Spectrum).

Please give us a star ⭐ if you find our work interesting!

![Overview](assets/img_qualitative_study.png "Overview")

🚀 Also checkout our previous work [CHORDS](https://github.com/hanjq17/CHORDS) on multi-core diffusion sampling acceleration, accepted at ICCV 2025!

## 🛠 Dependencies

Our code relies on the following core packages:
```
torch
transformers
diffusers
hydra-core
imageio
imageio-ffmpeg
```
For the specific versions of these packages that have been verified as well as some optional dependencies, please refer to `requirements.txt`. We recommend creating a new virual environment via the following procedure:
```bash
conda create -n spectrum python=3.10
conda activate spectrum
pip install -r requirements.txt
```

## 🚀 Running Inference

Prior to running inference pipeline, please make sure that the models have been downloaded from 🤗 huggingface. We provide the download script for some example models for image and video generation in `download.py`.


We use hydra to organize different hyperparameters for the image/video diffusion model as well as the sampling algorithm. The default configurations can be found under `configs` folder. The entries to launch the sampling for image and video generation are `src/text_to_image.py` and `src/text_to_video.py`, respectively. For SDXL, please refer to `src/text_to_image_sdxl.py`.


### ⭐ Text-to-Image (T2I)

The command below is an example to perform image generation on Flux using 1 GPU.

```bash
CUDA_VISIBLE_DEVICES=0 \
python src/text_to_image.py \
    model=flux \
    algo=spectrum \
    algo.w=0.5 \
    algo.lam=0.1 \
    algo.m=4 \
    window_size=2 \
    flex_window=0.75 \
    exp_name=temp \
    ngpu=1 \
    total_prompt_num=1000 \
    output_base_path=output_samples_image \
    prompt_file=prompts/DrawBench200.txt
```

For `model` we currently support:
- `flux`: [Flux](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- `sd3-5`: [Stable Diffusion 3.5-Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
- `sdxl`: [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (please use `python src/text_to_image_sdxl.py` to launch)

`algo.w` is by default set to 1.0, which recovers our Chebyshev predictor. Post publication, we also find that a convex mixture of our spectral predictor with linear interpolation slightly enhances robustness across a wider range of acceleration ratios. We recommend setting `algo.w` between 0.5 and 1.0, with a relatively larger value of `algo.w` when enabling more aggressive speedups (see `flex_window`).

`algo.lam` refers to the regularization strength $\lambda$ in the paper. By default set to 0.1.

`algo.m` refers to the number of Chebyshev bases. By default set to 4.

`window_size` refers to the initial window size $\mathcal{N}$ in the paper.

`flex_window` refers to the hyperparameter $\alpha$ in the paper. Notably, $\mathcal{N}$ and $\alpha$ defines the sequence of diffusion steps to perform actual forward pass of the denoiser. More details are in Appendix B.1 and Table 6 in the paper. A larger value of $\alpha$ corresponds to fewer actual network forwards, leading to larger speedup.

 `ngpu` corresponds to the number of GPUs to use in parallel. We split all prompts equally to several gpus to speedup the benchmark for all methods. Note that it should match `CUDA_VISIBLE_DEVICES`.
 
 `output_base_path` is the directory to save the generated samples.
 
 `prompt_file` stores the list of prompts, each per line, that will be sequentially employed to generate each image.

For full functionality of the script, please refer to the arguments and their default values (such as the number of inference steps, the resolution of the image, etc.) under the `configs` folder, which is parsed by hydra.

**Remarks:** `window_size=2` and `flex_window=0.75` recovers the $\alpha=0.75$ setting in the paper with 14 full network passes ($\approx 3.5\times$ speedup). For more aggressive acceleration, use `window_size=2` and `flex_window=3.0`, which corresponds to the $\alpha=3.0$ setting in the paper with 10 network passes ($\approx 5\times$ speedup).

We also provide a boilerplate script to launch the inference:
```bash
# For Flux and Stable Diffusion 3.5-Large
bash scripts/run_mp_image.sh
# For SDXL
bash scripts/run_mp_image_sdxl.sh
```

### ⭐ Text-to-Video (T2V)

Similarly, the following script can be used for video generation with *Spectrum*:

```bash
CUDA_VISIBLE_DEVICES=0 \
python src/text_to_video.py \
    model=hunyuan \
    algo=spectrum \
    algo.w=0.5 \
    algo.lam=0.1 \
    algo.m=4 \
    window_size=2 \
    flex_window=0.75 \
    exp_name=temp \
    ngpu=1 \
    total_prompt_num=1000 \
    output_base_path=output_samples_video \
    prompt_file=prompts/video_demo.txt
```
where for `model` we currently support:
- `hunyuan`: [HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo)
- `wan14b`: [Wan2.1-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers)

We also provide a boilerplate script to launch the inference:
```bash
# For HunyuanVideo and Wan2.1-14B
bash scripts/run_mp_video.sh
```

**Remark:** For high-resolution video generation, change `model.width`, `model.height`, and `model.num_frames` to your specific choice. For exmaple, we use `1080x720x129f` setting with HunyuanVideo for the qualitative examples.

## 📌 Citation

Please consider citing our work if you find it useful:

```
@article{han2026adaptive,
  title={Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration},
  author={Han, Jiaqi and Shi, Juntong and Li, Puheng and Ye, Haotian and Guo, Qiushan and Ermon, Stefano},
  journal={arXiv preprint arXiv:2603.01623},
  year={2026}
}
```

## 🗒️ Acknowledgments

Part of the code was inspired by [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer). We thank the authors for open-sourcing the codebase.

## 🧩 Contact and Community Contribution

If you have any question, welcome to contact me at:

Jiaqi Han: jiaqihan@stanford.edu

🔥 We warmly welcome **community contributions** for e.g. supporting more models! Please open/submit a PR if you are interested!

