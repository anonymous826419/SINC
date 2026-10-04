import ssl
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_openml
import os
import sys

# Bypass macOS SSL certificate verification for OpenML downloads
ssl._create_default_https_context = ssl._create_unverified_context

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
# SINC & Archipelago Wrappers for XGBoost (Regression)
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
# Data Loading
# ---------------------------------------------------------
def load_bike_sharing_data():
    print("Fetching UCI Bike Sharing Demand dataset (Hourly) from OpenML...")
    # OpenML ID 42712 corresponds to the hourly bike sharing dataset
    bike = fetch_openml(data_id=42712, as_frame=True, parser='auto')
    
    df = bike.data.copy()
    df['cnt'] = bike.target
    
    # Drop non-predictive/leakage columns (instant, date, and target sub-components)
    drop_cols = ['instant', 'dteday', 'casual', 'registered']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Separate features and target
    y = df['cnt'].values.astype(np.float32)
    X_df = df.drop(columns=['cnt'])
    
    # Convert categorical columns to numeric for XGBoost
    for col in X_df.columns:
        if X_df[col].dtype.name in ['category', 'object']:
            X_df[col] = X_df[col].cat.codes if X_df[col].dtype.name == 'category' else pd.factorize(X_df[col])[0]
            
    X = X_df.values.astype(np.float32)
    feature_names = list(X_df.columns)
    
    print(f"Dataset Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {feature_names}")
    return X, y, feature_names

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    X, y, feature_names = load_bike_sharing_data()

    X_train_np, X_val_np, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------------------------------------
    # Define the shared 100-sample evaluation set 
    # ---------------------------------------------------------
    np.random.seed(42)
    eval_indices = np.random.choice(len(X_val_np), size=100, replace=False)
    
    X_eval_np = X_val_np[eval_indices]
    y_eval = y_val[eval_indices]
    X_eval_df = pd.DataFrame(X_eval_np, columns=feature_names)

    # SINC Phase 1 background (still drawn from the training set)
    X_bg = get_background_data(X_train_np, subsample_size=100)

    # 1. Train XGBoost
    print("\n" + "=" * 55)
    print("  Training XGBRegressor (Bike Sharing)")
    print("=" * 55)
    
    model = xgb.XGBRegressor(
        n_estimators=1000, 
        max_depth=6, 
        learning_rate=0.1, 
        n_jobs=-1, 
        tree_method="hist", 
        random_state=40
    )
    model.fit(X_train_np, y_train)
    
    preds = model.predict(X_val_np)
    print(f"  Model MSE (Full Val Set): {mean_squared_error(y_val, preds):.2f}")

    # 2b. Global SHAP Analysis (Strictly on 100 eval samples)
    print("\n" + "=" * 55)
    print("  Global SHAP Analysis (100 Shared Samples)")
    print("=" * 55)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_eval_np)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    total_shap = mean_abs_shap.sum()
    
    shap_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap,
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    shap_importance["Importance (%)"] = (shap_importance["Mean_Abs_SHAP"] / total_shap * 100) if total_shap > 0 else 0.0

    print("\n--- Global SHAP Feature Importance (mean |SHAP|) ---")
    print(shap_importance.to_string(index=False))

    plt.figure()
    shap.summary_plot(shap_values, X_eval_df, plot_type="bar", max_display=10, show=False)
    plt.title("Global SHAP Feature Importance — Bike Sharing (100 Samples)")
    plt.tight_layout()
    plt.savefig("shap_global_bar_bike.png", dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_eval_df, max_display=10, show=False)
    plt.title("SHAP Summary (Beeswarm) — Bike Sharing (100 Samples)")
    plt.tight_layout()
    plt.savefig("shap_summary_beeswarm_bike.png", dpi=150)
    plt.close()

    print("\n--- Global SHAP Pairwise Interactions ---")
    n_feat = len(feature_names)
    
    # Interaction values calculated strictly on the shared 100 instances
    shap_inter = explainer.shap_interaction_values(X_eval_np)

    shap_main_effect = np.abs(shap_inter[:, np.arange(n_feat), np.arange(n_feat)]).mean(axis=0)
    W_shap = np.abs(shap_inter).mean(axis=0) * 2.0
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
            "Strength (%)": f"< {threshold_pct:.2f}"
        }])
        final_shap_pairs = pd.concat([significant_pairs, other_row], ignore_index=True)
    else:
        significant_pairs["SHAP_Interaction_Strength"] = significant_pairs["SHAP_Interaction_Strength"].apply(lambda x: f"{x:.6f}")
        significant_pairs["Strength (%)"] = significant_pairs["Strength (%)"].apply(lambda x: f"{x:.6f}")
        final_shap_pairs = significant_pairs

    print("\nGlobal SHAP Pairwise Interactions:")
    print(final_shap_pairs.to_string(index=False))

    plot_interaction_heatmap(
        W_shap, feature_names, title="Bike Sharing — SHAP Global Interaction Matrix (100 Samples)"
    )

    # 3. SINC Pipeline
    print("\n" + "=" * 55)
    print("  Running SINC Framework (100 Shared Samples)")
    print("=" * 55)
    
    wrapper = SincXGBWrapper(model)

    print("\n--- Phase 1: Interaction Mapping ---")
    pds = compute_marginal_pds(
        model_wrapper=wrapper, X_background=X_bg, grid_resolution=30
    )

    print("Scanning pairwise interactions...")
    interactions = compute_pairwise_interactions(
        model_wrapper=wrapper, X_background=X_bg, pds=pds, batch_size=512
    )
    W = build_interaction_network(interactions, feature_names, tau=0.05)
    plot_interaction_heatmap(W, feature_names, title="Bike Sharing — SINC Interaction Network")

    print("\n--- Phase 2: Community Detection ---")
    ig = InteractionGraph(W)
    communities = ig.detect_communities(
        resolution=0.05, algorithm="leiden", method="cpm", seed=42, n_iterations=-1
    )
    ig.plot_graph()

    print(f"Found {len(communities)} communities:")
    for i, c in enumerate(communities):
        print(f"  Group {i}: {c}")

    print("\n--- Phase 3: Synergy Validation ---")
    # Synergy impact weighted strictly on the 100 shared evaluation instances
    results = calculate_synergy_score(
        wrapped_model=wrapper,
        X_val_df=X_eval_df,
        y_val=y_eval,
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

    try:
        W_meta = analyze_inter_group_interactions(W, communities)
        plot_meta_heatmap(W_meta, results_df=results, title="SINC Map: Bike Sharing (100 Samples)")
    except Exception as e:
        print(f"Meta-visualization failed: {e}")

    # 4. ARCHIPELAGO
    print("\n" + "=" * 55)
    print("  Running Archipelago Baseline Comparison (100 Shared Samples)")
    print("=" * 55)
    
    archipelago_wrapper = ArchipelagoXGBWrapper(model)
    baseline_instance = X_train_np.mean(axis=0) 
    
    total_instances = len(X_eval_df)
    
    from collections import defaultdict
    global_scores = defaultdict(float)
    interaction_counts = defaultdict(int)
    
    from src.explainer import Archipelago
    
    print(f"Running explanations for {total_instances} instances...")
    # Iterating directly over the exact same 100 shared instances
    for i, (_, row) in enumerate(X_eval_df.iterrows()):
        xformer = TabularXformer(row.values, baseline_instance)
        
        apgo = Archipelago(
            archipelago_wrapper, 
            data_xformer=xformer, 
            output_indices=0  
        )
        
        explanations = apgo.explain(top_k=3)
        if isinstance(explanations, dict):
            explanations = explanations.items()
            
        for feature_set, score in explanations:
            community = tuple(sorted(feature_set))
            global_scores[community] += abs(score)
            interaction_counts[community] += 1
            
        if (i + 1) % 20 == 0:
            print(f"   Processed {i + 1}/{total_instances} instances")
            
    # 5. Aggregate Results
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
            "Attribution (%)": f"< {threshold_pct:.2f}"
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