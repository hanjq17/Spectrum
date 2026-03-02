"""
Inherited from TaylorSeer:
https://github.com/Shenyi-Z/TaylorSeer/blob/main/TaylorSeers-Diffusers/taylorseer_flux/cache_functions/cache_init.py
"""

def cache_init(self):
    """
    Initialization for cache.
    """
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1] = {}
    cache["hidden"] = {}
    cache["firstblock_hidden"] = {}
    cache_index[-1] = {}
    cache_index["layer_index"] = {}

    cache_dic["cache_counter"] = 0

    cache_dic["taylor_cache"] = False
    cache_dic["Delta-DiT"] = False

    mode = self.mode if hasattr(self, "mode") and self.mode is not None else "Taylor"

    if mode == "original":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 1
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 3

    elif mode == "ToCa":
        cache_dic["cache_type"] = "attention"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.1
        cache_dic["fresh_threshold"] = 5
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 3

    elif mode == "Taylor":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 2
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["taylor_cache"] = True
        cache_dic["max_order"] = self.order if hasattr(self, "order") and self.order is not None else 1
        cache_dic["firstblock_max_order"] = 1
        cache_dic["first_enhance"] = 3

    elif mode == "Delta":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 3
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["Delta-DiT"] = True
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 1

    current = {}
    current["block_activated_steps"] = [0]
    current["activated_steps"] = [0]
    current["step"] = 0
    current["num_steps"] = self.num_steps

    return cache_dic, current
