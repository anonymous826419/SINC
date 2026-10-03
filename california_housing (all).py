import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import sys, os
from shapiq.explainer import TabularExplainer
import shapiq

_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
if _here not in sys.path:
    sys.path.insert(0, _here)

from sincV2 import (
    get_background_data,
    compute_marginal_pds,
    compute_pairwise_interactions,
    build_interaction_network,
    plot_interaction_heatmap,
    InteractionGraph,
    calculate_synergy_score,
    analyze_inter_group_interactions,
    plot_meta_heatmap,
)

# ---------------------------------------------------------
# SINC & Archipelago Wrappers for XGBoost
# ---------------------------------------------------------
class SincXGBWrapper:
    def __init__(self, model):
        self.model = model
        
    def predict(self, X):
        return self.model.predict(X)

class ArchipelagoXGBWrapper:
    def __init__(self, model):
        self.model = model
        
    def __call__(self, batch_data):
        batch_data = np.vstack(batch_data)
        preds = self.model.predict(batch_data)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds

class TabularXformer:
    def __init__(self, instance, baseline):
        self.instance = np.array(instance)
        self.baseline = np.array(baseline)
        self.num_features = len(self.instance) 
        
    def __call__(self, instance_repr):
        mask_indices = np.argwhere(instance_repr == True).flatten()
        xformed_data = np.copy(self.baseline)
        for i in mask_indices:
            xformed_data[i] = self.instance[i]
        return xformed_data 

# ---------------------------------------------------------
# Data
# ---------------------------------------------------------
def load_real_world_data():
    data = fetch_california_housing()
    X = data.data
    y = data.target
    feature_names = list(data.feature_names)

    print(f"Dataset Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {feature_names}")

    return X, y, feature_names

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Load & split
    print("Loading California Housing Data...")
    X, y, feature_names = load_real_world_data()

    X_train_np, X_val_full, y_train, y_val_full = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- UNIFORM TEST SAMPLES ---
    NUM_TEST_SAMPLES = 100
    np.random.seed(42)
    random_indices = np.random.choice(len(X_val_full), NUM_TEST_SAMPLES, replace=False)

    X_val_np = X_val_full[random_indices]
    y_val = y_val_full[random_indices]
    X_val_df = pd.DataFrame(X_val_np, columns=feature_names)
    X_bg = get_background_data(X_train_np, subsample_size=100)

    # 2. Train XGBoost
    print("\n" + "=" * 55)
    print("  Training XGBRegressor")
    print("=" * 55)
    
    model = xgb.XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.1, n_jobs=-1, tree_method="hist", random_state=42)
    model.fit(X_train_np, y_train)
    
    preds = model.predict(X_val_np)
    print(f"  Model R2 (on 100 samples): {r2_score(y_val, preds):.4f}")

    # 2b. Global SHAP Analysis
    print("\n" + "=" * 55)
    print("  Global SHAP Analysis")
    print("=" * 55)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val_np)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    total_shap = mean_abs_shap.sum()
    
    shap_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap,
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    
    shap_importance["Importance (%)"] = (shap_importance["Mean_Abs_SHAP"] / total_shap * 100) if total_shap > 0 else 0.0

    print("\n--- Global SHAP Feature Importance (mean |SHAP|) ---")
    print(shap_importance.to_string(index=False))

    # Bar chart: global feature importance
    plt.figure()
    shap.summary_plot(shap_values, X_val_df, plot_type="bar", show=False)
    plt.title("Global SHAP Feature Importance — California Housing (XGB)")
    plt.tight_layout()
    plt.savefig("shap_global_bar.png", dpi=150)
    plt.close()

    # Beeswarm: distribution + direction of each feature's effect
    plt.figure()
    shap.summary_plot(shap_values, X_val_df, show=False)
    plt.title("SHAP Summary (Beeswarm) — California Housing (XGB)")
    plt.tight_layout()
    plt.savefig("shap_summary_beeswarm.png", dpi=150)
    plt.close()

    print("\nSaved SHAP plots: shap_global_bar.png, shap_summary_beeswarm.png")

    # 2c. Global SHAP Pairwise Interactions
    print("\n--- Global SHAP Pairwise Interactions ---")

    n_feat = len(feature_names)
    shap_inter = explainer.shap_interaction_values(X_val_np) 

    shap_main_effect = np.abs(shap_inter[:, np.arange(n_feat), np.arange(n_feat)]).mean(axis=0)

    W_shap = np.abs(shap_inter).mean(axis=0) * 2.0 #Multiply by two since it both looks at A + B and B + A
    np.fill_diagonal(W_shap, 0.0)

    shap_pairs = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            shap_pairs.append({
                "Feature_Pair": f"{feature_names[i]} + {feature_names[j]}",
                "SHAP_Interaction_Strength": W_shap[i, j],
            })
            
    shap_pairs_df = pd.DataFrame(shap_pairs).sort_values(
        by="SHAP_Interaction_Strength", ascending=False
    ).reset_index(drop=True)
    
    total_interaction_strength = shap_pairs_df["SHAP_Interaction_Strength"].sum()
    shap_pairs_df["Strength (%)"] = (shap_pairs_df["SHAP_Interaction_Strength"] / total_interaction_strength * 100) if total_interaction_strength > 0 else 0.0

    # --- Dynamic Grouping Logic (2% Threshold) ---
    threshold_pct = 2.0
    significant_pairs = shap_pairs_df[shap_pairs_df["Strength (%)"] >= threshold_pct].copy()
    noise_pairs = shap_pairs_df[shap_pairs_df["Strength (%)"] < threshold_pct]
    
    if not noise_pairs.empty:
        max_noise_val = noise_pairs["SHAP_Interaction_Strength"].max()
        
        significant_pairs["SHAP_Interaction_Strength"] = significant_pairs["SHAP_Interaction_Strength"].apply(lambda x: f"{x:.6f}")
        significant_pairs["Strength (%)"] = significant_pairs["Strength (%)"].apply(lambda x: f"{x:.6f}")
        
        other_row = pd.DataFrame([{
            "Feature_Pair": f"Other {len(noise_pairs)} pairs (each)",
            "SHAP_Interaction_Strength": f"< {max_noise_val:.6f}",
            "Strength (%)": "< 2.00"
        }])
        final_shap_pairs = pd.concat([significant_pairs, other_row], ignore_index=True)
    else:
        significant_pairs["SHAP_Interaction_Strength"] = significant_pairs["SHAP_Interaction_Strength"].apply(lambda x: f"{x:.6f}")
        significant_pairs["Strength (%)"] = significant_pairs["Strength (%)"].apply(lambda x: f"{x:.6f}")
        final_shap_pairs = significant_pairs

    print("\nGlobal SHAP Pairwise Interactions:")
    print(final_shap_pairs.to_string(index=False))

    plot_interaction_heatmap(
        W_shap, feature_names, title="California Housing — SHAP Global Interaction Matrix (XGB)"
    )

    # 3. SINC Pipeline
    wrapper = SincXGBWrapper(model)

    print("\n--- Phase 1: Interaction Mapping ---")
    pds = compute_marginal_pds(
        model_wrapper=wrapper, X_background=X_bg, grid_resolution=30
    )

    print("Scanning interactions (this may take 1-2 minutes)...")
    interactions = compute_pairwise_interactions(
        model_wrapper=wrapper, X_background=X_bg, pds=pds, batch_size=512
    )
    W = build_interaction_network(interactions, feature_names, tau=0.05)
    plot_interaction_heatmap(W, feature_names, title="California Housing — Interaction Network (XGB)")

    print("\n--- Phase 2: Community Detection ---")
    ig = InteractionGraph(W)
    communities = ig.detect_communities(
        resolution=0.01, algorithm="leiden", method="cpm", seed=42, n_iterations=-1
    )
    ig.plot_graph()

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

    print("\n--- Visualizing Active Structures ---")
    try:
        W_meta = analyze_inter_group_interactions(W, communities)
        plot_meta_heatmap(W_meta, results_df=results, title="SINC Map: California Housing (XGB)")
    except Exception as e:
        print(f"Meta-visualization failed: {e}")

    # 4. ARCHIPELAGO
    print("\n" + "=" * 55)
    print("  Running Archipelago Baseline Comparison")
    print("=" * 55)
    
    archipelago_wrapper = ArchipelagoXGBWrapper(model)
    baseline_instance = X_train_np.mean(axis=0) 
    
    # Use X_val_df directly since it is now subset to NUM_TEST_SAMPLES (100)
    total_instances = len(X_val_df)
    
    from collections import defaultdict
    global_scores = defaultdict(float)
    interaction_counts = defaultdict(int)
    
    from src.explainer import Archipelago
    
    print(f"Running explanations for {total_instances} instances...")
    for i, (_, row) in enumerate(X_val_df.iterrows()):
        xformer = TabularXformer(row.values, baseline_instance)
        
        apgo = Archipelago(
            archipelago_wrapper, 
            data_xformer=xformer, 
            output_indices=0  
        )
        
        explanations = apgo.explain(top_k=1)
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
    
    print("\n=== GLOBAL FEATURE COMMUNITIES (ARCHIPELAGO) ===")
    print(final_df.to_string(index=False))