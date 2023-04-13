import torch
import torch.nn as nn

class DomainSampler(nn.Module):
    def __init__(self, nb_points, dims) -> None:
        super().__init__()

        self.nb_points = nb_points
        self.dims = dims
        weight_init = 2 * (torch.rand(self.nb_points, self.dims)-0.5)
        for candidate_idx in range(self.nb_points):
            for harmonic_idx in range(self.dims):
                magnitude = 10**(-0.21*harmonic_idx - 1.6)
                weight_init[candidate_idx, harmonic_idx] *= magnitude
        
        self.weight = nn.Parameter(weight_init)

    def forward(self, _):

        return self.weight