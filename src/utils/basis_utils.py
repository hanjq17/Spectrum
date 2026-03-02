import torch
from typing import Optional, Tuple
import torch.nn as nn

DTYPE = torch.bfloat16

def _flatten(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    shape = x.shape
    return x.reshape(1, -1) if x.ndim == 1 else x.reshape(1, -1), shape

def _unflatten(x_flat: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return x_flat.reshape(shape)


class BaseForecaster(nn.Module):
    def __init__(self, M: int = 3, K: int = 10, lam: float = 1e-3, device: Optional[torch.device] = None, feature_shape = None):
        super().__init__()
        assert K >= M + 2, "K should exceed basis size for stability"
        self.M = M
        self.K = K
        self.lam = lam
        self.register_buffer("t_buf", torch.empty(0))       # (<=K,)
        self._H_buf: Optional[torch.Tensor] = None           # (<=K, F)
        self._shape: Optional[torch.Size] = None
        self._coef: Optional[torch.Tensor] = None            # (P, F)
        self._XtX_fac: Optional[torch.Tensor] = None         # Cholesky factor of (X^T X + lam I)
        self._tau_cache: Optional[torch.Tensor] = None       # (<=K,)
        self._X_cache: Optional[torch.Tensor] = None         # (<=K, P)
        self._last_delta_norm: Optional[torch.Tensor] = None
        self.device_ref = device
        self.feature_shape = feature_shape

    # ---- abstract bits ---- #
    def _taus(self, t: torch.Tensor) -> torch.Tensor:
        """Map scalar times to τ ∈ [-1, 1] using current window endpoints.
        Uses an affine map based on (t_min, t_max) of the buffer for stability.
        """
        assert self.t_buf.numel() >= 1
        # t_min = self.t_buf.min()
        # t_max = self.t_buf.max()
        t_min = (torch.ones(1) * 0).to(t)
        t_max = (torch.ones(1) * 50).to(t)
        if torch.isclose(t_max, t_min):
            return torch.zeros_like(t)
        mid = 0.5 * (t_min + t_max)
        rng = (t_max - t_min)
        return (t - mid) * 2.0 / rng

    def _build_design(self, taus: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def P(self) -> int:
        raise NotImplementedError

    # ---- core methods ---- #
    def update(self, t: float | torch.Tensor, h: torch.Tensor) -> None:
        """Append (t, h) to the window and update EMA of step delta.
        h can be any shape; we flatten internally.
        """
        device = self.device_ref or h.device
        t = torch.as_tensor(t, dtype=DTYPE, device=device)
        h_flat, shape = _flatten(h)
        h_flat = h_flat.to(device)
        if self._shape is None:
            self._shape = shape
        else:
            assert shape == self._shape, "Feature shape must remain constant"

        # update buffers
        if self.t_buf.numel() == 0:
            self.t_buf = t[None]
            self._H_buf = h_flat
        else:
            # maintain last-delta norm EMA
            delta = (h_flat - self._H_buf[-1])
            self._last_delta_norm = delta.norm(p=2)
            self.t_buf = torch.cat([self.t_buf, t[None]], dim=0)
            self._H_buf = torch.cat([self._H_buf, h_flat], dim=0)
            # trim
            if self.t_buf.numel() > self.K:
                self.t_buf = self.t_buf[-self.K:]
                self._H_buf = self._H_buf[-self.K:]
        # invalidate caches & coef when window changed
        self._coef = None
        self._XtX_fac = None
        self._tau_cache = None
        self._X_cache = None
        self._row_mean_cache = None
        self._row_var_cache = None

    def last_delta(self) -> torch.Tensor:
        if self._last_delta_norm is None:
            # small nonzero to avoid division by zero in TR checks
            return torch.tensor(1e-6, device=self.t_buf.device if self.t_buf.numel() else 'cpu')
        return self._last_delta_norm

    def ready(self) -> bool:
        return True
        # return self.t_buf.numel() >= min(self.K, self.M + 2)
    
    def _unflatten(self, H):
        return H.reshape(-1, *self.feature_shape)

    def _flatten(self, H):
        return H.reshape(H.shape[0], -1)

    def _fit_if_needed(self) -> None:
        if self._coef is not None:
            return
        assert self.ready()
        taus = self._taus(self.t_buf)
        X = self._build_design(taus).to(torch.float32)                 # (K, P)
        H = self._H_buf.to(torch.float32)                           # (K, F)
        K_, P = X.shape
        F = H.shape[1]

        assert P == self.P
        lamI = self.lam * torch.eye(P, device=X.device, dtype=X.dtype)
        Xt = X.transpose(0, 1)                       # (P, K)
        XtX = Xt @ X + lamI                          # (P, P)
        # Cholesky solve for many RHS (P x F)
        try:
            L = torch.linalg.cholesky(XtX.to(torch.float32))
        except:
            # add jitter if ill-conditioned
            jitter = 1e-6 * XtX.diag().mean()
            L = torch.linalg.cholesky(XtX + jitter * torch.eye(P, device=X.device))
        
        XtH = Xt @ H                                 # (P, F)
        # Solve (L L^T) C = XtH
        C = torch.cholesky_solve(XtH.to(torch.float32), L).to(DTYPE)             # (P, F)
        
        self._coef = C
        self._XtX_fac = L
        self._tau_cache = taus
        self._X_cache = X.to(DTYPE)

    @torch.no_grad()
    def predict(self, t_star: float | torch.Tensor) -> torch.Tensor:
        assert self._shape is not None
        device = self.t_buf.device
        t_star = torch.as_tensor(t_star, dtype=DTYPE, device=device)
        self._fit_if_needed()

        tau_star = self._taus(t_star)
        x_star = self._build_design(tau_star[None])  # (1, P)
        h_flat = x_star @ self._coef                 # (1, F)
        
        return _unflatten(h_flat, self._shape)


class ChebyshevForecaster(BaseForecaster):
    """Chebyshev T-polynomials on τ ∈ [-1, 1]: T_0..T_M via recurrence.
    Columns: [T0, T1, ..., TM] → P = M + 1
    """
    def __init__(self, M: int = 4, K: int = 10, lam: float = 1e-3, device: Optional[torch.device] = None, feature_shape = None):
        super().__init__(M, K, lam, device, feature_shape)

    @property
    def P(self) -> int:
        return self.M + 1

    def _build_design(self, taus: torch.Tensor) -> torch.Tensor:
        taus = taus.reshape(-1, 1)                   # (K, 1)
        K = taus.shape[0]
        T0 = torch.ones((K, 1), device=taus.device, dtype=taus.dtype)
        if self.M == 0:
            return T0
        T1 = taus
        cols = [T0, T1]
        for m in range(2, self.M + 1):
            Tm = 2 * taus * cols[-1] - cols[-2]
            cols.append(Tm)
        return torch.cat(cols[: self.M + 1], dim=1)  # (K, P)


class Spectrum(nn.Module):
    def __init__(self,
                 cheb_like,
                 taylor_order: int = 2,
                 enable_blend: bool = True,
                 prefer: str = 'auto',
                 w: float = None,
                 alpha: float = 6.0,
                 ema_beta: float = 0.9):
        super().__init__()
        assert taylor_order in (1, 2, 3)
        assert prefer in ('auto', 'taylor', 'cheb')
        self.cheb = cheb_like
        self.taylor_order = taylor_order
        self.enable_blend = enable_blend
        self.prefer = prefer
        self.alpha = alpha
        self.ema_beta = ema_beta
        self._delta_ref = None  # EMA of |Δt|
        self.w = w

    @torch.no_grad()
    def _local_taylor_discrete(self, t_star: torch.Tensor) -> torch.Tensor:
        H = self.cheb._H_buf
        t = self.cheb.t_buf
        h_i = H[-1]; t_i = t[-1]
        if t.numel() < 2:
            return h_i.clone()
        h_im1 = H[-2]; t_im1 = t[-2]
        # Unit-step forward difference Δh_i
        dh1 = (h_i - h_im1)
        # Fractional step in units of the last spacing (NO rounding)
        dt_last = (t_i - t_im1).clamp_min(1e-8)
        k = ((t_star - t_i) / dt_last).to(h_i.dtype)
        # First two terms
        out = h_i + k * dh1
        if self.taylor_order >= 2 and t.numel() >= 3:
            h_im2 = H[-3]
            d2 = (h_i - 2 * h_im1 + h_im2)              # Δ²h_i
            out = out + 0.5 * k * (k - 1.0) * d2        # C(k,2)
        if self.taylor_order >= 3 and t.numel() >= 4:
            h_im3 = H[-4]
            d3 = (h_i - 3*h_im1 + 3*h_im2 - h_im3)      # Δ³h_i
            out = out + (k * (k - 1.0) * (k - 2.0) / 6.0) * d3  # C(k,3)
        return out

    @torch.no_grad()
    def predict(self, t_star: float | torch.Tensor, return_weight: bool = False):
        device = self.cheb.t_buf.device
        t_star = torch.as_tensor(t_star, dtype=DTYPE, device=device)
        # Chebyshev (maybe with discrete Hermite constraint)
        h_cheb = self.cheb.predict(t_star)
        # Discrete Taylor (Newton forward with fractional k)
        h_taylor = self._local_taylor_discrete(t_star)

        assert self.w is not None
        w = self.w
        
        h_mix = (1 - w) * h_taylor + w * h_cheb
        return (h_mix, float(w)) if return_weight else h_mix

    def update_w(self, new_w: float):
        self.w = new_w
    
    # passthroughs
    def update(self, t, h):
        return self.cheb.update(t, h)
    def last_delta(self):
        return self.cheb.last_delta()
    def ready(self):
        return self.cheb.ready()


