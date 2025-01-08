import time
import torch
from torch import nn
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


# Use nn.Module so we can compile
class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def update_velocity(
        self, space: torch.Tensor, G: float, dt: float, softening: float
    ):
        n, d, _ = space.shape
        x = space[..., 0].view(n, 1, d) - space[..., 0].view(1, n, d)
        x *= (
            G / (x.pow(2) + softening).sum(-1).pow(1.5).fill_diagonal_(float("inf"))
        ).unsqueeze(-1)
        space[..., 1] -= dt * x.sum(1)
        return space

    def update_position(self, space: torch.Tensor, dt: float):
        space[..., 0] += dt * space[..., 1]
        return space

    def forward(self, space: torch.Tensor, G: float, dt: float, softening: float):
        space = self.update_velocity(space=space, G=G, dt=dt, softening=softening)
        space = self.update_position(space=space, dt=dt)


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
    t0 = time.time()
    total_time = 0
    model = Model()
    with open("trajectories/results.data", "w") as f:
        f.write("")
    for i in range(nsteps):
        t = time.time()
        model.forward(space=space, G=G, dt=dt, softening=softening)
        total_time += time.time() - t
        if i % write_interval == 0:
            with open("trajectories/results.data", "a") as f:
                for body in space:
                    pos = " ".join([str(x.item()) for x in body[:, 0]])
                    f.write(pos + "\n")
                f.write("\n")
    print(f"CPU time: {total_time} seconds")
    print(f"Total time: {time.time() - t0} seconds")
