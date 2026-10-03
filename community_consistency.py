"""
Consistency analysis of SINC's community detection with respect to t and g.

t = tau            -> threshold on H^2 used when building the interaction network
g = resolution      -> resolution parameter used by Leiden community detection

Two independent 1D sweeps are run:
  - vary t, holding g fixed
  - vary g, holding t fixed

Key idea: the marginal PDs and pairwise H-statistics are independent of both
t and g, so we compute them ONCE and reuse them across both sweeps instead of
recomputing them from scratch for every value (as the original script did).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os

_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
if _here not in sys.path:
    sys.path.insert(0, _here)

from sincV2 import (
    PyTorchWrapper,
    get_background_data,
    compute_marginal_pds,
    compute_pairwise_interactions,
    build_interaction_network,
    InteractionGraph,
)

# ---------------------------------------------------------
# Model (same architecture as parameter_test.py)
# ---------------------------------------------------------
class SmallNet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 16),         nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def train_nn(model, X_train_np, y_train_np, epochs=150, lr=3e-3, batch_size=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    X_t = torch.from_numpy(X_train_np).float().to(device)
    y_t = torch.from_numpy(y_train_np).float().to(device)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    opt, crit = torch.optim.Adam(model.parameters(), lr=lr), nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
    model.eval()
    return model


# ---------------------------------------------------------
# Core sweeps
# ---------------------------------------------------------
def _detect(W, g, algorithm, method, seed):
    ig_obj = InteractionGraph(W)
    communities = ig_obj.detect_communities(
        resolution=g, algorithm=algorithm, method=method,
        seed=seed, n_iterations=-1
    )
    sizes = sorted((len(c) for c in communities), reverse=True)
    return communities, sizes


def run_t_sweep(interactions, feature_names, t_values, g_fixed,
                 algorithm="leiden", method="cpm", seed=42):
    """Vary t (tau), holding g (resolution) fixed."""
    records = []
    for t in t_values:
        W = build_interaction_network(interactions, feature_names, tau=t)
        communities, sizes = _detect(W, g_fixed, algorithm, method, seed)
        records.append({
            "param": "t", "value": t, "g_fixed": g_fixed,
            "n_communities": len(communities),
            "community_sizes": sizes,
            "largest_community": sizes[0] if sizes else 0,
            "n_singletons": sum(1 for s in sizes if s == 1),
        })
        print(f"  [t sweep] t={t:.2f} (g={g_fixed}) -> {len(communities)} communities  sizes={sizes}")
    return pd.DataFrame(records)


def run_g_sweep(interactions, feature_names, g_values, t_fixed,
                 algorithm="leiden", method="cpm", seed=42):
    """Vary g (resolution), holding t (tau) fixed."""
    W = build_interaction_network(interactions, feature_names, tau=t_fixed)
    records = []
    for g in g_values:
        communities, sizes = _detect(W, g, algorithm, method, seed)
        records.append({
            "param": "g", "value": g, "t_fixed": t_fixed,
            "n_communities": len(communities),
            "community_sizes": sizes,
            "largest_community": sizes[0] if sizes else 0,
            "n_singletons": sum(1 for s in sizes if s == 1),
        })
        print(f"  [g sweep] g={g:.2f} (t={t_fixed}) -> {len(communities)} communities  sizes={sizes}")
    return pd.DataFrame(records)


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------
def plot_sweeps(df_t, df_g, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(df_t["value"], df_t["n_communities"], marker="o")
    axes[0].set_xlabel("tau (t)")
    axes[0].set_ylabel("# communities")
    axes[0].set_title(f"Communities vs t (g={df_t['g_fixed'].iloc[0]})")

    axes[1].plot(df_g["value"], df_g["n_communities"], marker="o", color="darkorange")
    axes[1].set_xlabel("resolution (g)")
    axes[1].set_ylabel("# communities")
    axes[1].set_title(f"Communities vs g (t={df_g['t_fixed'].iloc[0]})")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(1000, 10)
    y = (20 * np.sin(np.pi * X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3])
         + 10 * X[:, 4] - 5 * X[:, 5]
         + np.random.normal(0, 0.1, 1000))
    feats = [f"feat_{i}" for i in range(10)]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_bg = get_background_data(X_train, subsample_size=100)

    print("Training model...")
    model_strong = train_nn(SmallNet(10), X_train, y_train, epochs=150)

    wrapper = PyTorchWrapper(model_strong, is_classification=False)
    print("Computing marginal PDs (once, reused for both sweeps)...")
    pds = compute_marginal_pds(wrapper, X_bg, grid_resolution=30)
    print("Computing pairwise interactions (once, reused for both sweeps)...")
    interactions = compute_pairwise_interactions(wrapper, X_bg, pds, batch_size=512)

    # build_interaction_network squares the raw H values (H^2) before thresholding/
    # using them as edge weights, so that's the actual scale g (resolution) and t (tau)
    # operate on. Cap g's sweep range at the largest H^2 present in the network -
    # beyond that, CPM resolution has nothing left to bite into (every edge would
    # already be "too weak" relative to the resolution) so higher values are wasted.
    max_h2 = max((h_val ** 2 for h_val in interactions.values()), default=1.0)
    print(f"Max H^2 across all pairs: {max_h2:.6f} -> using as g's upper bound")

    t_values = list(np.linspace(0, 1.0, 100))        # centered on 0.05
    g_values = list(np.linspace(0, max_h2, 1000))      # capped at the largest observed H^2
    t_fixed = 0.000000000001  # used while sweeping g
    g_fixed = 0.001   # used while sweeping t

    print("\nSweeping t (g fixed)...")
    df_t = run_t_sweep(interactions, feats, t_values, g_fixed=g_fixed)

    print("\nSweeping g (t fixed)...")
    df_g = run_g_sweep(interactions, feats, g_values, t_fixed=t_fixed)

    df_t.to_csv(os.path.join(_here, "community_t_sweep.csv"), index=False)
    df_g.to_csv(os.path.join(_here, "community_g_sweep.csv"), index=False)
    print("\nSaved results to community_t_sweep.csv and community_g_sweep.csv")

    plot_sweeps(df_t, df_g, save_path=os.path.join(_here, "community_sweeps.png"))
