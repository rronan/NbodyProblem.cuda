import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def step(space: torch.Tensor, G: float, dt: float, damping: float, softening: float):
    n, d, _ = space.shape
    dist = torch.norm(space[..., 0].view(n, 1, d), space[..., 0].view(1, n, d))
    dist = torch.tril(dist, diagonal=-1)
    dist_flat = dist[torch.tril_indices(n, n, -1)]
    assert len(dist_flat) == n * (n - 1) / 2
    dist[torch.triu_indices(n, n, 1)] = -dist_flat
    force = G / dist_flat
    pass


def run(
    space: np.array,
    nsteps: int,
    G: float,
    dt: float,
    damping: float,
    softening: float,
    write_interval: int,
):
    space = torch.tensor(space).to(device)
    print(space.shape)
