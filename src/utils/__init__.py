from typing import Dict

import torch
from utils.basis_utils import (
    ChebyshevForecaster,
    Spectrum,
)


cur_method = None
W = 0.5
M = 4
LAM = 0.1
DTYPE = torch.bfloat16

def set_method(method):
    global cur_method
    cur_method = method
    
def set_w(w_value):
    global W
    W = w_value
    
def set_m(m_value):
    global M
    M = m_value

def set_lam(lam_value):
    global LAM
    LAM = lam_value


def step_derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if cur_method in ['spectrum']:
        cur_step = current['activated_steps'][-1]
        if 'forecaster' not in cache_dic['cache']:
            # print(f'init forecaster at step {cur_step}')
        # if cur_step == 0:
            feature_info = feature.shape
            forecaster = ChebyshevForecaster(
                M=M, K=100, lam=LAM, device=feature.device, feature_shape=feature.shape[1:])
            if cur_method == 'spectrum':
                forecaster = Spectrum(
                    forecaster,
                    taylor_order=1,
                    enable_blend=True,
                    prefer='cheb',
                    w=W,
                )
        else:
            feature_info, forecaster = cache_dic['cache']['forecaster']
        # fit basis based on feature_info
        t = torch.ones(1).to(feature) * cur_step
        H = feature  # [K, F] K = num_points, F = feature_dim
        H = H.reshape(-1)
        forecaster.update(t[-1], H)

        info = (feature_info, forecaster)
        cache_dic['cache']['forecaster'] = info
    else:
        raise NotImplementedError(f"Method {cur_method} not implemented.")


def step_taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Compute Taylor expansion error.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if cur_method in ['spectrum']:
        feature_info, forecaster = cache_dic['cache']['forecaster']
        output = forecaster.predict(current['step'])
        x_shape = feature_info
        output = output.reshape(x_shape)
        output = output.to(DTYPE)
    else:
        raise NotImplementedError(f"Method {cur_method} not implemented.")

    return output
