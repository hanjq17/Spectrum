import hydra
import torch
from pipelines.sdxl_forward import our_sdxl_forward
from pipelines.sdxl_pipeline import our_sdxl_call
import torch.multiprocessing as mp


from diffusers import DiffusionPipeline
from diffusers.utils import logging

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

    pipeline = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
    )
    
    set_method(method)
    if method in ['spectrum']:
        set_w(config.algo.w)
        set_lam(config.algo.lam)
        set_m(config.algo.m)

    # Overwrite the functions
    pipeline.unet.__class__.num_steps = num_inference_steps
    if method == 'nocache':
        pass
    elif method in ['spectrum']:
        if config.model.model_name == 'sdxl':
            pipeline.__class__.__call__ = our_sdxl_call
            pipeline.unet.__class__.forward = our_sdxl_forward
        else:
            raise NotImplementedError(f"Model {config.model.model_name} not implemented.")
        pipeline.unet.cnt = 0
        pipeline.unet.num_consecutive_cached_steps = 0
        pipeline.unet.num_steps = num_inference_steps

        pipeline.unet.warmup_steps = config.warmup_steps
        if config.window_size is not None:
            pipeline.unet.window_size = config.window_size
            pipeline.unet.curr_ws = config.window_size
            pipeline.unet.flex_window = config.flex_window
    else:
        raise NotImplementedError(f"Method {method} not implemented.")

    pipeline.to(f"cuda:{rank}")

    for idx, prompt in tqdm(
        enumerate(prompts), total=len(prompts), desc=f"Rank {rank}"
    ):
        pipeline.unet.actual_forward_counter = 0

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        img = pipeline(
            prompt, 
            num_inference_steps=num_inference_steps,
            generator=torch.Generator().manual_seed(seed),
            ).images[0]

        end.record()
        torch.cuda.synchronize()

        elapsed_time = start.elapsed_time(end) * 1e-3

        cur_name = prompt.replace(" ", "_").replace(".", "_")
        cur_name = cur_name[:50]  # limit length
        cur_output_path = os.path.join(output_path, cur_name)
        if not os.path.exists(cur_output_path):
            os.makedirs(cur_output_path)

        img.save(f"{cur_output_path}/image.png")
        stats = {
            "time": elapsed_time,
            "prompt": prompt,
            "actual_forward_count": pipeline.unet.actual_forward_counter if hasattr(
                pipeline.unet, 'actual_forward_counter') else None,
            "window_size": pipeline.unet.window_size if hasattr(
                pipeline.unet, 'window_size') else None,
            "flex_window": pipeline.unet.flex_window if hasattr(
                pipeline.unet, 'flex_window') else None,
            "w": config.algo.w if hasattr(config.algo, 'w') else None,
            "lam": config.algo.lam if hasattr(config.algo, 'lam') else None,
            "method": method,
        }
        with open(f"{cur_output_path}/stats.json", "w") as f:
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

