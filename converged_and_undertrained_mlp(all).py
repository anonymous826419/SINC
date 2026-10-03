import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import sys, os
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
        with torch.no_grad():
            t_data = torch.from_numpy(np.vstack(batch_data)).float().to(self.device)
            preds = self.model(t_data).cpu().numpy()
            
        return preds.reshape(-1, 1) if preds.ndim == 1 else preds

class TabularXformer:
    def __init__(self, instance, baseline):
        self.instance = np.array(instance)
        self.baseline = np.array(baseline)
        self.num_features = len(self.instance)
        
    def __call__(self, mask):
        x = np.copy(self.baseline)
        for i in np.argwhere(mask == True).flatten(): 
            x[i] = self.instance[i]
        return x

# ---------------------------------------------------------
# Pipelines
# ---------------------------------------------------------

def run_standard_shap(label, model, X_bg, X_val, feature_names):
    print(f"\n{'='*55}")
    print(f"  Standard SHAP Analysis — {label}")
    print(f"{'='*55}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    X_bg_tensor = torch.from_numpy(X_bg).float().to(device)
    X_val_tensor = torch.from_numpy(X_val).float().to(device)

    class ShapDeepWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        def forward(self, x):
            out = self.base_model(x)
            if out.ndim == 1:
                out = out.unsqueeze(-1)
            return out

    wrapped_model = ShapDeepWrapper(model)
    explainer = shap.DeepExplainer(wrapped_model, X_bg_tensor)
    shap_values = explainer.shap_values(X_val_tensor)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if shap_values.ndim == 3:
        shap_values = shap_values.squeeze(-1)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    total_shap = mean_abs_shap.sum()
    
    shap_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap,
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    shap_importance["Importance (%)"] = (shap_importance["Mean_Abs_SHAP"] / total_shap * 100) if total_shap > 0 else 0.0

    print(f"\n--- Global SHAP Feature Importance [{label}] ---")
    print(shap_importance.to_string(index=False))

def run_shapiq(label, model, X_bg, X_val, feature_names):
    print(f"\n{'='*55}")
    print(f"  SHAP-IQ ANALYSIS — {label}")
    print(f"{'='*55}")
    
    wrapper = PyTorchBlackBoxWrapper(model)
    explainer = TabularExplainer(
        model=wrapper,
        data=X_bg,
        index="k-SII",  
        max_order=2     
    )
    
    X_sample = X_val 
    n_features = len(feature_names)
    
    main_effects = np.zeros(n_features)
    W_shapiq = np.zeros((n_features, n_features))
    
    print(f"Approximating SHAP-IQ values (Main + Pairwise) for {len(X_sample)} instances...")
    for i in range(len(X_sample)):
        interaction_vals = explainer.explain(X_sample[i], budget=1024)
        
        def get_val(key_tuple):
            if hasattr(interaction_vals, "dict_values"):
                return interaction_vals.dict_values.get(key_tuple, 0.0)
            elif hasattr(interaction_vals, "__getitem__"):
                try:
                    return interaction_vals[key_tuple]
                except KeyError:
                    return 0.0
            return 0.0

        for f in range(n_features):
            main_effects[f] += abs(get_val((f,)))
            
        for f1 in range(n_features):
            for f2 in range(f1 + 1, n_features):
                W_shapiq[f1, f2] += abs(get_val((f1, f2)))
                
    main_effects /= len(X_sample)
    W_shapiq /= len(X_sample) 
    
    total_interaction_strength = W_shapiq.sum()
    
    pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            strength = W_shapiq[i, j]
            pct = (strength / total_interaction_strength * 100) if total_interaction_strength > 0 else 0.0
            pairs.append({
                "Feature_Pair": f"{feature_names[i]} + {feature_names[j]}", 
                "Interaction_Strength": strength,
                "Strength (%)": pct
            })
            
    pairs_df = pd.DataFrame(pairs).sort_values(by="Interaction_Strength", ascending=False).reset_index(drop=True)
    
    # --- Dynamic Grouping Logic (2% Threshold) ---
    threshold_pct = 2.0
    significant_pairs = pairs_df[pairs_df["Strength (%)"] >= threshold_pct].copy()
    noise_pairs = pairs_df[pairs_df["Strength (%)"] < threshold_pct]
    
    if not noise_pairs.empty:
        max_noise_val = noise_pairs["Interaction_Strength"].max()
        
        # Format significant values as strings to align perfectly with the "<" signs
        significant_pairs["Interaction_Strength"] = significant_pairs["Interaction_Strength"].apply(lambda x: f"{x:.6f}")
        significant_pairs["Strength (%)"] = significant_pairs["Strength (%)"].apply(lambda x: f"{x:.6f}")
        
        other_row = pd.DataFrame([{
            "Feature_Pair": f"Other {len(noise_pairs)} pairs (each)",
            "Interaction_Strength": f"< {max_noise_val:.6f}",
            "Strength (%)": "< 2.00"
        }])
        final_df = pd.concat([significant_pairs, other_row], ignore_index=True)
    else:
        significant_pairs["Interaction_Strength"] = significant_pairs["Interaction_Strength"].apply(lambda x: f"{x:.6f}")
        significant_pairs["Strength (%)"] = significant_pairs["Strength (%)"].apply(lambda x: f"{x:.6f}")
        final_df = significant_pairs
    
    print("\n--- Global SHAP-IQ Pairwise Interactions ---")
    print(final_df.to_string(index=False))

def run_sinc(label, model, X_bg, X_val_df, y_val, feature_names):
    print(f"\n{'='*55}")
    print(f"  SINC PIPELINE — {label}")
    print(f"{'='*55}")

    print("\n--- Phase 1: Interaction Mapping ---")
    wrapper = PyTorchWrapper(model, is_classification=False)
    pds = compute_marginal_pds(model_wrapper=wrapper, X_background=X_bg, grid_resolution=30)
    interactions = compute_pairwise_interactions(
        model_wrapper=wrapper, X_background=X_bg, pds=pds, batch_size=512
    )
    W = build_interaction_network(interactions, feature_names, tau=0.05)
    
    print("\n--- Phase 2: Community Detection ---")
    ig = InteractionGraph(W)
    communities = ig.detect_communities(
        resolution=0.001, algorithm="leiden", method="cpm", seed=42, n_iterations=-1
    )
    
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
    
    possible_score_cols = ["Impact", "Synergy", "Score", "MSE_Diff", "Synergy_Score"]
    score_col = next((col for col in possible_score_cols if col in results.columns), None)
    
    if not score_col:
        numeric_cols = results.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            score_col = numeric_cols[-1]
            
    if score_col:
        # Floor negative values to zero
        results[score_col] = results[score_col].clip(lower=0)
        
        total_impact = results[score_col].sum()
        results["Impact (%)"] = (results[score_col] / total_impact * 100) if total_impact > 0 else 0.0

    print("\n--- FINAL RESULTS ---")
    print(results.to_string(index=False))
    return results, communities, W

def run_archipelago(label, model, X_train, X_val_df, feature_names):
    print(f"\n{'='*55}")
    print(f"  ARCHIPELAGO GLOBAL BASELINE — {label}")
    print(f"{'='*55}")
    
    from src.explainer import Archipelago
    wrapper = PyTorchArchipelagoWrapper(model)
    baseline = X_train.mean(axis=0) 
    
    sample = X_val_df
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
            
    archipelago_results = []
    for community, total_score in global_scores.items():
        community_names = tuple(feature_names[idx] for idx in community)
        name_str = " + ".join(community_names)
            
        archipelago_results.append({
            "Feature_Community": name_str,
            "Community_Size": len(community),
            "Global_Attribution": total_score / total_instances,
            "Appearance_Count": interaction_counts[community]
        })
        
    df_results = pd.DataFrame(archipelago_results)
    
    total_attribution = df_results["Global_Attribution"].sum()
    df_results["Attribution (%)"] = (df_results["Global_Attribution"] / total_attribution * 100) if total_attribution > 0 else 0.0
    df_results = df_results.sort_values(by="Global_Attribution", ascending=False).reset_index(drop=True)
    
    # --- Dynamic Grouping Logic (2% Threshold) ---
    threshold_pct = 2.0
    significant_comms = df_results[df_results["Attribution (%)"] >= threshold_pct].copy()
    noise_comms = df_results[df_results["Attribution (%)"] < threshold_pct]
    
    if not noise_comms.empty:
        max_noise_val = noise_comms["Global_Attribution"].max()
        
        # Format strictly for presentation
        significant_comms["Global_Attribution"] = significant_comms["Global_Attribution"].apply(lambda x: f"{x:.6f}")
        significant_comms["Attribution (%)"] = significant_comms["Attribution (%)"].apply(lambda x: f"{x:.6f}")
        significant_comms["Community_Size"] = significant_comms["Community_Size"].astype(str)
        significant_comms["Appearance_Count"] = significant_comms["Appearance_Count"].astype(str)
        
        other_row = pd.DataFrame([{
            "Feature_Community": f"Other {len(noise_comms)} communities (each)",
            "Community_Size": "-",
            "Global_Attribution": f"< {max_noise_val:.6f}",
            "Appearance_Count": "-",
            "Attribution (%)": "< 2.00"
        }])
        final_df = pd.concat([significant_comms, other_row], ignore_index=True)
    else:
        significant_comms["Global_Attribution"] = significant_comms["Global_Attribution"].apply(lambda x: f"{x:.6f}")
        significant_comms["Attribution (%)"] = significant_comms["Attribution (%)"].apply(lambda x: f"{x:.6f}")
        significant_comms["Community_Size"] = significant_comms["Community_Size"].astype(str)
        significant_comms["Appearance_Count"] = significant_comms["Appearance_Count"].astype(str)
        final_df = significant_comms

    print(f"\n=== GLOBAL FEATURE COMMUNITIES [{label}] ===")
    print(final_df.to_string(index=False))
        
    return df_results # Return the raw dataframe for downstream analysis if needed, but print the formatted one

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(1000, 10)
    y = 20 * np.sin(np.pi * X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3]) + 10 * X[:, 4] - 5 * X[:, 5] + np.random.normal(0, 0.1, 1000)
    feats = [f"feat_{i}" for i in range(10)]
    
    X_train, X_val_full, y_train, y_val_full = train_test_split(X, y, test_size=0.2, random_state=42)
    
    NUM_TEST_SAMPLES = 100
    
    X_val = X_val_full[:NUM_TEST_SAMPLES]
    y_val = y_val_full[:NUM_TEST_SAMPLES]
    X_val_df = pd.DataFrame(X_val, columns=feats)

    X_bg = get_background_data(X_train, subsample_size=100)

    print("\n" + "="*55 + "\n  Model A: Converged (150 epochs)\n" + "="*55)
    model_converged = train_nn(SmallNet(10), X_train, y_train, epochs=150)
    
    run_standard_shap("Converged", model_converged, X_bg, X_val, feats)
    run_shapiq("Converged", model_converged, X_bg, X_val, feats)
    run_sinc("Converged", model_converged, X_bg, X_val_df, y_val, feats)
    run_archipelago("Converged", model_converged, X_train, X_val_df, feats)

    print("\n" + "="*55 + "\n  Model B: Undertrained (20 epochs)\n" + "="*55)
    model_undertrained = train_nn(SmallNet(10), X_train, y_train, epochs=20)
    
    run_standard_shap("Undertrained", model_undertrained, X_bg, X_val, feats)
    run_shapiq("Undertrained", model_undertrained, X_bg, X_val, feats)
    run_sinc("Undertrained", model_undertrained, X_bg, X_val_df, y_val, feats)
    run_archipelago("Undertrained", model_undertrained, X_train, X_val_df, feats)