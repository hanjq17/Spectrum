from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from cache_functions.cache_init import cache_init
from utils import (
    step_derivative_approximation,
    step_taylor_formula,
)
import math

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def our_hunyuan_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    pooled_projections: torch.Tensor,
    guidance: torch.Tensor = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
    temp_dict = {},
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
    p, p_t = self.config.patch_size, self.config.patch_size_t
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p
    post_patch_width = width // p
    first_frame_num_tokens = 1 * post_patch_height * post_patch_width

    # 1. RoPE
    image_rotary_emb = self.rope(hidden_states)

    # 2. Conditional embeddings
    temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)

    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

    # 3. Attention mask preparation
    latent_sequence_length = hidden_states.shape[1]
    condition_sequence_length = encoder_hidden_states.shape[1]
    sequence_length = latent_sequence_length + condition_sequence_length
    attention_mask = torch.ones(
        batch_size, sequence_length, device=hidden_states.device, dtype=torch.bool
    )  # [B, N]
    effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)  # [B,]
    effective_sequence_length = latent_sequence_length + effective_condition_sequence_length
    indices = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)  # [1, N]
    mask_indices = indices >= effective_sequence_length.unsqueeze(1)  # [B, N]
    attention_mask = attention_mask.masked_fill(mask_indices, False)
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]

    ########## Here ##########
    cache_dic = temp_dict["cache_dic"]
    current = temp_dict["current"]
    
    actual_forward = True
    if self.cnt >= self.warmup_steps:
        actual_forward = (self.num_consecutive_cached_steps + 1) % math.floor(self.curr_ws) == 0
        if actual_forward:
            self.curr_ws += self.flex_window
            self.curr_ws = round(self.curr_ws, 3)  # counter for floating point errors
            current["block_activated_steps"].append(current["step"])

    self.cnt += 1
    if actual_forward:
        self.num_consecutive_cached_steps = 0
    else:
        self.num_consecutive_cached_steps += 1
    
    if self.cnt == self.num_steps:
        # Reset counters
        self.cnt = 0
        self.num_consecutive_cached_steps = 0
        self.curr_ws = self.window_size if hasattr(self, 'window_size') and self.window_size is not None else None

    if actual_forward:
        self.actual_forward_counter += 1
    
    if not actual_forward:
        out = step_taylor_formula(cache_dic=cache_dic, current=current)
        hidden_states = out.reshape(batch_size, -1, out.shape[-1])  # [2, A, B]
    else:
        current["activated_steps"].append(current["step"])
    
        # 4. Transformer blocks
        for index_block, block in enumerate(self.transformer_blocks):
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )

        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )
        feat = hidden_states.reshape(-1, hidden_states.shape[-1]).unsqueeze(0)  # [1, 2A, B]
        step_derivative_approximation(cache_dic=cache_dic, current=current, feature=feat)

    # 5. Output projection
    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
    hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)
    
    ######## HERE!!!!!!
    temp_dict["current"]["step"] += 1

    if not return_dict:
        return (hidden_states, temp_dict)

    return Transformer2DModelOutput(sample=hidden_states)
