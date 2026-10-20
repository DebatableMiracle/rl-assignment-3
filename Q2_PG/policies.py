"""Policy representations (both output the MEAN of a Gaussian with fixed sigma = 0.3).

a. TablePolicy : each of the ds observation dims is cut into B = 8 bins -> a bin-index tuple -> ONE unique row of a
                 trainable table theta in R^{B^ds x da}  (4096 x 1 for InvertedPendulum).  Initialised to zero.
b. NeuralPolicy: ds -> 64 -> 64 -> da, ReLU, linear output.  PyTorch default init in hidden layers, final layer
                 weights AND bias zero.

Binning (documented as required)
--------------------------------
Observation = (cart position, pole angle, cart velocity, pole angular velocity).  Ranges used:
    cart pos  [-0.5, 0.5]    pole angle [-0.2, 0.2]  (episode terminates at |angle| > 0.2)
    cart vel  [-1.0, 1.0]    pole ang.vel [-2.0, 2.0]
Each range is split into B = 8 equal-width bins with np.digitize on the 7 interior edges.  Values outside the range
are NOT dropped: they fall into the first / last bin of that dimension (digitize returns 0 / B-1), i.e. the range is
effectively extended to -inf / +inf in the two outermost bins.
Index = sum_d bin_d * B^(d_last - d)  (row-major / base-B number) -> every one of the 8^4 tuples has its own row.
"""
import numpy as np
import torch
import torch.nn as nn

OBS_LOW = np.array([-0.5, -0.2, -1.0, -2.0])
OBS_HIGH = -OBS_LOW
BINS = 8


class TablePolicy(nn.Module):
    kind = "table"

    def __init__(self, ds, da, bins=BINS, low=OBS_LOW, high=OBS_HIGH):
        super().__init__()
        self.ds, self.da, self.bins = ds, da, bins
        self.theta = nn.Parameter(torch.zeros(bins ** ds, da))                 # zero-initialised table
        self.edges = [np.linspace(low[d], high[d], bins + 1)[1:-1] for d in range(ds)]   # 7 interior edges
        self.mult = bins ** np.arange(ds - 1, -1, -1)

    def index(self, obs):
        obs = np.asarray(obs, dtype=np.float64).reshape(-1, self.ds)
        idx = np.zeros(len(obs), dtype=np.int64)
        for d in range(self.ds):
            idx += np.digitize(obs[:, d], self.edges[d]) * self.mult[d]       # out-of-range -> end bins
        return idx

    def mean(self, obs):
        return self.theta[torch.as_tensor(self.index(obs))]


class NeuralPolicy(nn.Module):
    kind = "neural"

    def __init__(self, ds, da, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ds, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, da))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def mean(self, obs):
        return self.net(torch.as_tensor(np.asarray(obs), dtype=torch.float32))


def make_policy(kind, seed, ds, da):
    """For a given seed every neural method gets IDENTICAL initial parameters (global torch seed set here)."""
    torch.manual_seed(seed)
    return TablePolicy(ds, da) if kind == "table" else NeuralPolicy(ds, da)


def param_norm(policy):
    return float(torch.sqrt(sum((p.detach() ** 2).sum() for p in policy.parameters())))
