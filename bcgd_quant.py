# bcgd_quant.py
import torch


def quantized_relu(x: torch.Tensor, alpha: float, bits: int):
    """
    Exact Eq.(2) from Yin et al.
    """
    L = (2 ** bits) - 1

    out = torch.zeros_like(x)

    # middle region
    mask_mid = (x > 0) & (x <= L * alpha)
    k = torch.floor(x[mask_mid] / alpha)
    out[mask_mid] = k * alpha

    # saturation
    mask_sat = x > L * alpha
    out[mask_sat] = L * alpha

    return out


def grad_mask_x(x: torch.Tensor, alpha: float, bits: int):
    """
    Proxy dσ/dx (clipped ReLU derivative)
    """
    L = (2 ** bits) - 1
    return ((x > 0) & (x <= L * alpha)).float()


def grad_alpha(x: torch.Tensor, alpha: float, bits: int):
    """
    3-valued proxy derivative dσ/dα from the paper
    """
    L = (2 ** bits) - 1
    g = torch.zeros_like(x)

    g[(x > 0) & (x <= L * alpha)] = 2 ** (bits - 1)
    g[x > L * alpha] = L

    return g
