from diffusers import (
    DiffusionPipeline,
    HunyuanVideoPipeline,
    WanPipeline,
)

# Image diffusion models
model_path = "black-forest-labs/FLUX.1-dev"
pipe = DiffusionPipeline.from_pretrained(model_path)

model_path = "stabilityai/stable-diffusion-3.5-large"
pipe = DiffusionPipeline.from_pretrained(model_path)

model_path = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(model_path)

# Video diffusion models
model_path = "hunyuanvideo-community/HunyuanVideo"
pipe = HunyuanVideoPipeline.from_pretrained(model_path)

model_path = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
pipe = WanPipeline.from_pretrained(model_path)

print('done')