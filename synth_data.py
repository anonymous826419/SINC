import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import sys, os
# Remove 'import shapiq' if it is there, and use this exact path:
from shapiq.explainer import TabularExplainer
import shap
import shapiq

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
    calculate_synergy_score
)

# ---------------------------------------------------------
# Neural Networks
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
# Wrappers
# ---------------------------------------------------------
class PyTorchBlackBoxWrapper:
    """A simple function wrapper required by shapiq to query the PyTorch model."""
    def __init__(self, model):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        
    def __call__(self, X):
        self.model.eval()
        with torch.no_grad():
            t_data = torch.from_numpy(X).float().to(self.device)
            preds = self.model(t_data).cpu().numpy()
        return preds

class PyTorchArchipelagoWrapper:
    def __init__(self, model):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        
    def __call__(self, batch_data):
        self.model.eval()
        with torch.no_grad():  # <-- This prevents the RuntimeError
            t_data = torch.from_numpy(np.vstack(batch_data)).float().to(self.device)
            preds = self.model(t_data).cpu().numpy()
            
        return preds.reshape(-1, 1) if preds.ndim == 1 else preds

class TabularXformer:
    def __init__(self, instance, baseline):
        self.instance = np.array(instance)
        self.baseline = np.array(baseline)
        self.num_features = len(self.instance)  # <-- Archipelago needs this
        
    def __call__(self, mask):
        x = np.copy(self.baseline)
        for i in np.argwhere(mask == True).flatten(): 
            x[i] = self.instance[i]
        return x

# ---------------------------------------------------------
# Threshold filtering (replaces the fixed "top 10")
# ---------------------------------------------------------
IMPORTANCE_THRESHOLD = 0.02  # keep items contributing >= 2% of the total importance

def filter_by_share(df, value_col, threshold=IMPORTANCE_THRESHOLD):
    """Add a '<value_col>_Share_%' column (each row's share of the column total)
    and keep only rows whose share is >= threshold. Sorted descending."""
    df = df.copy()
    total = df[value_col].sum()
    share = df[value_col] / total if total > 0 else 0.0
    df[f"{value_col}_Share_%"] = (share * 100).round(2)
    kept = df[share >= threshold].sort_values(by=value_col, ascending=False).reset_index(drop=True)
    return kept, total, len(df)

# ---------------------------------------------------------
# Pipelines
# ---------------------------------------------------------

def run_standard_shap(label, model, X_bg, X_val, feature_names):
    print(f"\n{'='*55}")
    print(f"  Standard SHAP Analysis — {label}")
    print(f"{'='*55}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Convert NumPy arrays to PyTorch tensors for DeepExplainer
    X_bg_tensor = torch.from_numpy(X_bg).float().to(device)
    X_val_tensor = torch.from_numpy(X_val).float().to(device)

    # Create a simple wrapper to ensure the output is 2D (batch_size, 1)
    class ShapDeepWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        def forward(self, x):
            out = self.base_model(x)
            # If the model squashed the output to 1D, expand it back to 2D
            if out.ndim == 1:
                out = out.unsqueeze(-1)
            return out

    wrapped_model = ShapDeepWrapper(model)

    # DeepExplainer is ideal for PyTorch models
    explainer = shap.DeepExplainer(wrapped_model, X_bg_tensor)
    
    # Generate SHAP values
    shap_values = explainer.shap_values(X_val_tensor)

    # DeepExplainer sometimes returns a list or an array with an extra dimension
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if shap_values.ndim == 3:
        shap_values = shap_values.squeeze(-1)

    # Global importance = mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap,
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    print(f"\n--- Global SHAP Feature Importance (mean |SHAP|) [{label}] ---")
    print(shap_importance.to_string(index=False))

def run_shapiq(label, model, X_bg, X_val, feature_names):
    print(f"\n{'='*55}")
    print(f"  SHAP-IQ ANALYSIS — {label}")
    print(f"{'='*55}")
    
    wrapper = PyTorchBlackBoxWrapper(model)
    
    # Initialize the SHAP-IQ Tabular Explainer
    from shapiq.explainer import TabularExplainer
    explainer = TabularExplainer(
        model=wrapper,
        data=X_bg,
        index="k-SII",  # k-Shapley Interaction Index
        max_order=2     # Stop at Pairwise interactions
    )
    
    X_sample = X_val[:50] 
    n_features = len(feature_names)
    
    # Arrays to track both main effects and interaction strengths
    main_effects = np.zeros(n_features)
    W_shapiq = np.zeros((n_features, n_features))
    
    print(f"Approximating SHAP-IQ values (Main + Pairwise) for {len(X_sample)} instances...")
    for i in range(len(X_sample)):
        interaction_vals = explainer.explain(X_sample[i], budget=1024)
        
        # Helper function to safely extract values from the shapiq object
        def get_val(key_tuple):
            if hasattr(interaction_vals, "dict_values"):
                return interaction_vals.dict_values.get(key_tuple, 0.0)
            elif hasattr(interaction_vals, "__getitem__"):
                try:
                    return interaction_vals[key_tuple]
                except KeyError:
                    return 0.0
            return 0.0

        # 1. Extract 1st-order Main Effects
        for f in range(n_features):
            main_effects[f] += abs(get_val((f,)))
            
        # 2. Extract 2nd-order Pairwise Interactions
        for f1 in range(n_features):
            for f2 in range(f1 + 1, n_features):
                W_shapiq[f1, f2] += abs(get_val((f1, f2)))
                
    # Average across the evaluated instances
    main_effects /= len(X_sample)
    W_shapiq /= len(X_sample) 
    
    # --- Format Main Effects Table ---
    main_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": main_effects
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    
    #print("\n--- Global SHAP Feature Importance (mean |SHAP|) ---")
    #print(main_df.to_string(index=False))
    
    # --- Format Pairwise Interactions Table ---
    pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            pairs.append({
                "Feature_Pair": f"{feature_names[i]} + {feature_names[j]}", 
                "SHAPIQ_Interaction_Strength": W_shapiq[i, j]
            })
            
    pairs_df, total, n_all = filter_by_share(pd.DataFrame(pairs), "SHAPIQ_Interaction_Strength")
    
    print("\n--- Global SHAP Pairwise Interactions ---")
    print(f"Global SHAP-IQ Pairwise Interactions contributing >= {IMPORTANCE_THRESHOLD:.0%} "
          f"of the total ({len(pairs_df)} of {n_all} pairs; total strength = {total:.4f}):")
    print(pairs_df.to_string(index=False))

def run_sinc(label, model, X_bg, X_val_df, y_val, feature_names, t, g):
    print(f"\n{'='*55}")
    print(f"  SINC PIPELINE — {label}")
    print(f"{'='*55}")

    print("\n--- Phase 1: Interaction Mapping ---")
    wrapper = PyTorchWrapper(model, is_classification=False)
    pds = compute_marginal_pds(model_wrapper=wrapper, X_background=X_bg, grid_resolution=30)
    interactions = compute_pairwise_interactions(
        model_wrapper=wrapper, X_background=X_bg, pds=pds, batch_size=512
    )
    W = build_interaction_network(interactions, feature_names, tau=t)
    
    # Optional: plot_interaction_heatmap(W, feature_names, title=f"[{label}] Interaction Network")

    print("\n--- Phase 2: Community Detection ---")
    ig = InteractionGraph(W)
    communities = ig.detect_communities(
        resolution=g, algorithm="leiden", method="cpm", seed=42, n_iterations=-1
    )
    # Optional: ig.plot_graph()
    
    print(f"Found {len(communities)} communities:")
    for i, c in enumerate(communities):
        print(f"  Group {i}: {c}")

    print("\n--- Phase 3: Synergy Validation ---")
    results = calculate_synergy_score(
        wrapped_model=wrapper, 
        X_val_df=X_val_df, 
        y_val=y_val,
        communities=communities, 
        metric_func=mean_squared_error, 
        n_repeats=5,
    )

    print("\n--- FINAL RESULTS ---")
    print(results.to_string(index=False))

    print("\n--- Visualizing Active Structures ---")
    try:
        W_meta = analyze_inter_group_interactions(W, communities)
        plot_meta_heatmap(W_meta, results_df=results, title=f"SINC Map [{label}]")
    except Exception as e:
        print(f"Meta-visualization failed: {e}")

    return results, communities, W

def run_archipelago(label, model, X_train, X_val_df, feature_names):
    print(f"\n{'='*55}")
    print(f"  ARCHIPELAGO GLOBAL BASELINE — {label}")
    print(f"{'='*55}")
    
    from src.explainer import Archipelago
    wrapper = PyTorchArchipelagoWrapper(model)
    baseline = X_train.mean(axis=0) 
    
    sample = X_val_df.sample(n=50, random_state=42)
    total_instances = len(sample)
    
    global_scores = defaultdict(float)
    interaction_counts = defaultdict(int)
    
    print(f"Running explanations for {total_instances} instances...")
    for i, (_, row) in enumerate(sample.iterrows()):
        xformer = TabularXformer(row.values, baseline)
        apgo = Archipelago(
            wrapper, 
            data_xformer=xformer, 
            output_indices=0  
        )
        
        explanations = apgo.explain(top_k=5)
        if isinstance(explanations, dict):
            explanations = explanations.items()
            
        for feature_set, score in explanations:
            community = tuple(sorted(feature_set))
            global_scores[community] += abs(score)
            interaction_counts[community] += 1
            
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{total_instances} instances")
            
    # Aggregate Results
    archipelago_results = []
    for community, total_score in global_scores.items():
        community_names = tuple(feature_names[idx] for idx in community)
        name_str = " + ".join(community_names)
            
        archipelago_results.append({
            "Feature_Community": name_str,
            "Community_Size": len(community),
            "Global_Importance": total_score / total_instances,
            "Appearance_Count": interaction_counts[community]
        })
        
    df_results, total, n_all = filter_by_share(pd.DataFrame(archipelago_results), "Global_Importance")
    
    print(f"\n=== GLOBAL FEATURE COMMUNITIES >= {IMPORTANCE_THRESHOLD:.0%} OF TOTAL [{label}] ===")
    print(f"({len(df_results)} of {n_all} communities; total importance = {total:.4f})")
    print(df_results.to_string())
    return df_results

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot

def choose_dataset(dataset_name, n_samples):
    np.random.seed(42)
    N = n_samples

    if dataset_name == "f1":
        X = np.zeros((N, 10))

        # Features that use the standard uniform [0.0, 1.0]
        standard_indices = [0, 1, 2, 5, 6, 8]
        X[:, standard_indices] = np.random.uniform(0.0, 1.0, size=(N, len(standard_indices)))

        # Features that use the offset uniform [0.6, 1.0]
        offset_indices = [3, 4, 7, 9]
        X[:, offset_indices] = np.random.uniform(0.6, 1.0, size=(N, len(offset_indices)))

        y = (np.pi ** (X[:, 0] * X[:, 1])) * np.sqrt(2 * X[:, 2]) \
            - np.arcsin(X[:, 3]) \
            + np.log(X[:, 2] + X[:, 4]) \
            - (X[:, 8] / X[:, 9]) * np.sqrt(X[:, 6] / X[:, 7]) \
            - X[:, 1] * X[:, 6]

        feats = [f"x_{i + 1}" for i in range(10)]
        return X, y, feats
    
    elif dataset_name == "f2":
        X = np.random.uniform(-1, 1, size=(N, 10))

        y = (np.pi ** (X[:, 0] * X[:, 1])) * np.sqrt(2 * np.abs(X[:, 2])) \
            - np.arcsin(0.5 * X[:, 3]) \
            + np.log(np.abs(X[:, 2] + X[:, 4]) + 1) \
            + (X[:, 8] / (1 + np.abs(X[:, 9]))) * np.sqrt(np.abs(X[:, 6]) / (1 + np.abs(X[:, 7]))) \
            - X[:, 1] * X[:, 6]

        feats = [f"x_{i + 1}" for i in range(10)]
        return X, y, feats

    elif dataset_name == "f3":
        X = np.random.uniform(-1, 1, size=(N, 10))
        
        y = np.exp(np.abs(X[:, 0] - X[:, 1])) \
            + np.abs(X[:, 1] * X[:, 2]) \
            - np.abs(X[:, 2]) ** (2 * np.abs(X[:, 3])) \
            + np.log(X[:, 3]**2 + X[:, 4]**2 + X[:, 6]**2 + X[:, 7]**2) \
            + X[:, 8] \
            + 1 / (1 + X[:, 9]**2)

        feats = [f"x_{i + 1}" for i in range(10)]
        return X, y, feats

    elif dataset_name == "f4":
        X = np.random.uniform(-1, 1, size=(N, 10))      
        
        y = np.exp(np.abs(X[:, 0] - X[:, 1])) \
            + np.abs(X[:, 1] * X[:, 2]) \
            - np.abs(X[:, 2]) ** (2 * np.abs(X[:, 3])) \
            + (X[:, 0] * X[:, 3])**2 \
            + np.log(X[:, 3]**2 + X[:, 4]**2 + X[:, 6]**2 + X[:, 7]**2) \
            + X[:, 8] \
            + 1 / (1 + X[:, 9]**2)
        
        feats = [f"x_{i + 1}" for i in range(10)]
        return X, y, feats

    elif dataset_name == "f5":
        X = np.random.uniform(-1, 1, size=(N, 10))

        y = 1 / (1 + X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2) \
            + np.sqrt(np.exp(X[:, 3] + X[:, 4])) \
            + np.abs(X[:, 5] + X[:, 6]) \
            + X[:, 7] * X[:, 8] * X[:, 9]

        feats = [f"x_{i + 1}" for i in range(10)]
        return X, y, feats

    elif dataset_name == "f6":
        X = np.random.uniform(-1, 1, size=(N, 10))

        y = np.exp(np.abs(X[:, 0] * X[:, 1]) + 1) \
            - np.exp(np.abs(X[:, 2] + X[:, 3]) + 1) \
            + np.cos(X[:, 4] + X[:, 5] - X[:, 7]) \
            + np.sqrt(X[:, 7]**2 + X[:, 8]**2 + X[:, 9]**2)

        feats = [f"x_{i + 1}" for i in range(10)]
        return X, y, feats

    elif dataset_name == "f7":
        X = np.random.uniform(-1, 1, size=(N, 10))

        y = (np.arctan(X[:, 0]) + np.arctan(X[:, 1]))**2 \
            + np.maximum(X[:, 2] * X[:, 3] + X[:, 5], 0) \
            - 1 / (1 + (X[:, 3] * X[:, 4] * X[:, 5] * X[:, 6] * X[:, 7])**2) \
            + (np.abs(X[:, 6]) / (1 + np.abs(X[:, 8])))**5 \
            + np.sum(X[:, 0:10], axis=1)

        feats = [f"x_{i + 1}" for i in range(10)]
        return X, y, feats

    elif dataset_name == "f8":
        X = np.random.uniform(-1, 1, size=(N, 10))
        
        y = X[:, 0] * X[:, 1] \
            + 2 ** (X[:, 2] + X[:, 4] + X[:, 5]) \
            + 2 ** (X[:, 2] + X[:, 3] + X[:, 4] + X[:, 6]) \
            + np.sin(X[:, 6] * np.sin(X[:, 7] + X[:, 8])) \
            + np.arccos(0.9 * X[:, 9])

        feats = [f"x_{i + 1}" for i in range(10)]
        return X, y, feats

    elif dataset_name == "f9":
        X = np.random.uniform(-1, 1, size=(N, 10))

        y = np.tanh(X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3]) * np.sqrt(np.abs(X[:, 4])) \
            + np.exp(X[:, 4] + X[:, 5]) \
            + np.log((X[:, 5] * X[:, 6] * X[:, 7])**2 + 1) \
            + X[:, 8] * X[:, 9] \
            + 1 / (1 + np.abs(X[:, 9]))

        feats = [f"x_{i + 1}" for i in range(10)]
        return X, y, feats

    elif dataset_name == "f10":
        X = np.random.uniform(-1, 1, size=(N, 10))

        y = np.sinh(X[:, 0] + X[:, 1]) \
            + np.arccos(np.tanh(X[:, 2] + X[:, 4] + X[:, 6])) \
            + np.cos(X[:, 3] + X[:, 4]) \
            + 1 / np.cos(X[:, 6] * X[:, 8])   
        
        feats = [f"x_{i + 1}" for i in range(10)]
        return X, y, feats

    raise ValueError(f"Unsupported dataset: {dataset_name}")


# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
if __name__ == "__main__":

    dataset_name = "f10"  # Change this to select different datasets (f1, f2, ..., f10)
    n_samples = 10000

    X, y, feats = choose_dataset(dataset_name, n_samples)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val_df = pd.DataFrame(X_val, columns=feats)
    X_bg = get_background_data(X_train, subsample_size=100)

    print("\n" + "="*55 + "\n  Model A: Strong (150 epochs)\n" + "="*55)
    model_strong = train_nn(SmallNet(10), X_train, y_train, epochs=150)

    with torch.no_grad():
            preds_strong = model_strong(torch.from_numpy(X_val).float()).numpy()
    print(f"  Model  R²: {r2_score(y_val, preds_strong):.4f}")

    run_sinc("Strong", model_strong, X_bg, X_val_df, y_val, feats, t=0.00001, g=0.009)
    #run_standard_shap("Strong", model_strong, X_bg, X_val, feats)
    
    run_shapiq("Strong", model_strong, X_bg, X_val, feats)
    
    run_archipelago("Strong", model_strong, X_train, X_val_df, feats)



