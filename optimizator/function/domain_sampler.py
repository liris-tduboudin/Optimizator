import torch
import torch.nn as nn

class DomainSampler(nn.Module):
    def __init__(self, nb_points, dims) -> None:
        super().__init__()

        self.nb_points = nb_points
        self.dims = dims
        weight_init = torch.zeros(self.nb_points, self.dims)
        for harmonic_idx in range(self.dims):
            # magnitude = 10*10**(-0.21*harmonic_idx - 1.6)
            magnitude = 1.0
            weight_init[:, harmonic_idx] = magnitude*2.0*(torch.rand(self.nb_points)-0.5)
        self.weight = nn.Parameter(weight_init)

    def forward(self, _):

        return self.weight