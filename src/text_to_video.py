import hydra
import torch
from pipelines.hunyuan_forward import our_hunyuan_forward
from pipelines.wan_forward import our_wan_forward
from pipelines.hunyuan_pipeline import our_hunyuan_call
from pipelines.wan_pipeline import our_wan_call
import torch.multiprocessing as mp

from diffusers import HunyuanVideoPipeline, WanPipeline, AutoencoderKLWan
from diffusers.utils import logging
from diffusers.utils import export_to_video
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from utils import set_method, set_w, set_lam, set_m
import json
import os
from tqdm import tqdm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def worker(
    rank: int,
    prompts: list,
    output_path,
    config,
):
    dtype = torch.float16 if config.model.dtype == "float16" else torch.bfloat16
    model_path = config.model.model_path
    num_inference_steps = config.num_inference_steps
    seed = config.seed
    method = config.algo.algo_name

    if config.model.model_name == 'hunyuan':
        pipeline = HunyuanVideoPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
        )
    elif config.model.model_name == 'wan14b':
        vae = AutoencoderKLWan.from_pretrained(
            config.model.model_path, subfolder="vae", torch_dtype=torch.float32)
        flow_shift = 3.0 # 5.0 for 720P, 3.0 for 480P
        scheduler = UniPCMultistepScheduler(
            prediction_type='flow_prediction', use_flow_sigmas=True,
            num_train_timesteps=1000, flow_shift=flow_shift)
        pipeline = WanPipeline.from_pretrained(
            config.model.model_path, vae=vae, torch_dtype=dtype)
        pipeline.scheduler = scheduler
    else:
        raise NotImplementedError(f"Model {config.model.model_name} not implemented.")
    
    set_method(method)
    pipeline.transformer.flex_w = False
    if method in ['spectrum']:
        set_w(config.algo.w)
        set_lam(config.algo.lam)
        set_m(config.algo.m)

    # Overwrite the functions
    pipeline.transformer.__class__.num_steps = num_inference_steps
    if method == 'nocache':
        pass
    elif method in ['spectrum']:
        if config.model.model_name == 'hunyuan':
            pipeline.__class__.__call__ = our_hunyuan_call
            pipeline.transformer.__class__.forward = our_hunyuan_forward
        elif config.model.model_name == 'wan14b':
            pipeline.__class__.__call__ = our_wan_call
            pipeline.transformer.__class__.forward = our_wan_forward
        else:
            raise NotImplementedError(f"Model {config.model.model_name} not implemented.")
        pipeline.transformer.cnt = 0
        pipeline.transformer.num_consecutive_cached_steps = 0
        pipeline.transformer.num_steps = num_inference_steps

        pipeline.transformer.pre_firstblock_hidden_states = None
        pipeline.transformer.previous_residual = None
        pipeline.transformer.pre_compute_hidden = None
        pipeline.transformer.predict_loss = None
        pipeline.transformer.predict_hidden_states = None
        pipeline.transformer.warmup_steps = config.warmup_steps
        if config.window_size is not None:
            pipeline.transformer.window_size = config.window_size
            pipeline.transformer.curr_ws = config.window_size
            pipeline.transformer.flex_window = config.flex_window
        
        if config.model.model_name == 'wan14b':
            pipeline.transformer.cnt_uncond = 0
            pipeline.transformer.num_consecutive_cached_steps_uncond = 0
            pipeline.transformer.pre_firstblock_hidden_states_uncond = None
            pipeline.transformer.previous_residual_uncond = None
            pipeline.transformer.pre_compute_hidden_uncond = None
            pipeline.transformer.predict_loss_uncond = None
            pipeline.transformer.predict_hidden_states_uncond = None
            if config.window_size is not None:
                pipeline.transformer.curr_ws_uncond = config.window_size
    else:
        raise NotImplementedError(f"Method {method} not implemented.")

    pipeline.to(f"cuda:{rank}")

    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()

    for idx, prompt in tqdm(
        enumerate(prompts), total=len(prompts), desc=f"Rank {rank}"
    ):
        pipeline.transformer.actual_forward_counter = 0
        pipeline.transformer.actual_forward_counter_uncond = 0

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        video = pipeline(
            prompt,
            height=config.model.height,
            width=config.model.width,
            num_frames=config.model.num_frames,
            guidance_scale=config.model.guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator().manual_seed(seed)
            ).frames[0]

        end.record()
        torch.cuda.synchronize()

        elapsed_time = start.elapsed_time(end) * 1e-3

        cur_name = prompt.replace(" ", "_").replace(".", "_")
        cur_name = cur_name[:50]  # limit length
        # assert os.path.exists(output_path), f"Output path {output_path} does not exist."
        cur_output_path = os.path.join(output_path, cur_name)

        export_to_video(video, f"{cur_output_path}.mp4", fps=config.model.fps)

        stats = {
            "time": elapsed_time,
            "prompt": prompt,
            "actual_forward_count": pipeline.transformer.actual_forward_counter if hasattr(
                pipeline.transformer, 'actual_forward_counter') else None,
            "window_size": pipeline.transformer.window_size if hasattr(
                pipeline.transformer, 'window_size') else None,
            "flex_window": pipeline.transformer.flex_window if hasattr(
                pipeline.transformer, 'flex_window') else None,
            "w": config.algo.w if hasattr(config.algo, 'w') else None,
            "lam": config.algo.lam if hasattr(config.algo, 'lam') else None,
            "method": method,
        }
        with open(f"{cur_output_path}.json", "w") as f:
            json.dump(stats, f, indent=4)
        print(cur_output_path)
        print(stats)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(config):
    mp.set_start_method("spawn", force=True)
    processes = []
    num_processes = config.ngpu
    prompt_file = config.prompt_file
    total_prompt_num = config.total_prompt_num
    with open(prompt_file, 'r') as f:
        prompts = f.read().splitlines()
        start = 0
        prompts = prompts[start:start+total_prompt_num]
    # split prompts into num_processes parts
    prompts_splitted = [prompts[i::num_processes] for i in range(num_processes)]

    output_base_path = config.output_base_path
    exp_name = config.exp_name
    output_path = os.path.join(output_base_path, exp_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for rank in range(num_processes):
        p = mp.Process(
            target=worker,
            args=(
                rank,
                prompts_splitted[rank],
                output_path,
                config,
            ),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()

