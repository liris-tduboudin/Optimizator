import torch
import torch.nn as nn

def domain_sampler(nb_points, dims):
    magnitude = 1.0
    return magnitude * 2.0 * (torch.rand(nb_points, dims)-0.5)
