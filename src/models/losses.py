import torch
from torch.distributions.normal import Normal
import torch.nn.functional as F
import math

def gaussian_nll(mu, sigma, target):
    return 0.5 * torch.log(2 * torch.pi * sigma**2) + (target - mu)**2 / (2 * sigma**2)

def sigma_penalized_gaussian_nll(mu, sigma, target, sigma_weight):
    return gaussian_nll(mu, sigma, target) + sigma_weight * torch.mean(sigma)

def gaussian_crps(mu, sigma, target, eps=1e-6):
    """
    Computes the CRPS for a Gaussian distribution.
    
    Args:
        mu: Tensor of predicted means, shape [B]
        sigma: Tensor of predicted std deviations, shape [B]
        target: Tensor of true targets, shape [B]
    Returns:
        Tensor of CRPS values, shape [B]
    """
    sigma = sigma.clamp(min=eps)
    z = (target - mu) / sigma
    normal = Normal(0, 1)

    crps = sigma * (
        z * (2 * normal.cdf(z) - 1) +
        2 * normal.log_prob(z).exp() -
        1 / math.sqrt(math.pi)
    )
    return crps

def mse_loss(pred, target):
    return F.mse_loss(pred, target)
