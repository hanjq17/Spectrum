import math
from typing import Any, Dict, Optional, Tuple, Union

import torch

from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from cache_functions.cache_init import cache_init
from utils import (
    step_derivative_approximation,
    step_taylor_formula,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def our_wan_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    temp_dict = {},
    is_uncond: bool = False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    ####### Here ######
    if temp_dict is None:
        temp_dict = {}
    if temp_dict.get("cache_dic", None) is None:
        temp_dict["cache_dic"], temp_dict["current"] = cache_init(self)

    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    rotary_emb = self.rope(hidden_states)

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    ########## Here ##########
    cache_dic = temp_dict["cache_dic"]
    current = temp_dict["current"]

    if not is_uncond:
        cur_cnt = self.cnt
        cur_num_consecutive_cached_steps = self.num_consecutive_cached_steps
        cur_curr_ws = self.curr_ws
        cur_actual_forward_counter = self.actual_forward_counter
    else:
        cur_cnt = self.cnt_uncond
        cur_num_consecutive_cached_steps = self.num_consecutive_cached_steps_uncond
        cur_curr_ws = self.curr_ws_uncond
        cur_actual_forward_counter = self.actual_forward_counter_uncond
    
    actual_forward = True
    if cur_cnt >= self.warmup_steps:
        # can_use_cache = ((self.cnt + 1 - self.warmup_steps) % self.window_size != 0)
        actual_forward = (cur_num_consecutive_cached_steps + 1) % math.floor(cur_curr_ws) == 0
        if actual_forward:
            cur_curr_ws += self.flex_window
            cur_curr_ws = round(cur_curr_ws, 3)
            current["block_activated_steps"].append(current["step"])

    cur_cnt += 1
    if actual_forward:
        cur_num_consecutive_cached_steps = 0
    else:
        cur_num_consecutive_cached_steps += 1
    
    if cur_cnt == self.num_steps:
        # Reset counters
        cur_cnt = 0
        cur_num_consecutive_cached_steps = 0
        cur_curr_ws = self.window_size if hasattr(self, 'window_size') and self.window_size is not None else None
    
    if actual_forward:
        cur_actual_forward_counter += 1
    
    if not actual_forward:
        out = step_taylor_formula(cache_dic=cache_dic, current=current)
        hidden_states = out.reshape(batch_size, -1, out.shape[-1])  # [2, A, B]
    else:
        current["activated_steps"].append(current["step"])

        # 4. Transformer blocks
        for index_block, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
        
        feat = hidden_states.reshape(-1, hidden_states.shape[-1]).unsqueeze(0)  # [1, 2A, B]
        step_derivative_approximation(cache_dic=cache_dic, current=current, feature=feat)


    # 5. Output norm, projection & unpatchify
    shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

    # Move the shift and scale tensors to the same device as hidden_states.
    # When using multi-GPU inference via accelerate these will be on the
    # first device rather than the last device, which hidden_states ends up
    # on.
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)
    
    ######## HERE!!!!!!
    temp_dict["current"]["step"] += 1

    ### set the variables back
    if not is_uncond:
        self.cnt = cur_cnt
        self.num_consecutive_cached_steps = cur_num_consecutive_cached_steps
        self.curr_ws = cur_curr_ws
        self.actual_forward_counter = cur_actual_forward_counter
    else:
        self.cnt_uncond = cur_cnt
        self.num_consecutive_cached_steps_uncond = cur_num_consecutive_cached_steps
        self.curr_ws_uncond = cur_curr_ws
        self.actual_forward_counter_uncond = cur_actual_forward_counter

    if not return_dict:
        return (output, temp_dict)

    return Transformer2DModelOutput(sample=output)