import sys
import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text

_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
if _here not in sys.path:
    sys.path.insert(0, _here)

from sincV2_parallel import (
    SklearnWrapper,
    run_sinc_pipeline,
)

# ------------------------------------------------------------------ #
# Archipelago Wrappers
# ------------------------------------------------------------------ #
class SklearnArchipelagoWrapper:
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


def make_logic_dataset(n_samples=2000, n_features=5, random_state=42):
    """
    A simple dataset a single Decision Tree can solve perfectly.

    1. AND Gate (Features 0 & 1): Interaction!
       - If F0 > 0.5 AND F1 > 0.5 -> +10 points

    2. Step Linear (Feature 2): No Interaction!
       - Adds 5 points if F2 > 0.5.

    3. Noise (Features 3+): Useless.
    """
    np.random.seed(random_state)
    X = np.random.rand(n_samples, n_features)

    # AND Gate
    mask_and = (X[:, 0] > 0.5) & (X[:, 1] > 0.5)
    y_interaction = 10 * mask_and.astype(float)

    # Solo Feature
    y_linear = 5 * (X[:, 2] > 0.5).astype(float)

    y = y_interaction + y_linear

    feature_names = [f"Noise_{i}" for i in range(n_features)]
    feature_names[0] = "A"
    feature_names[1] = "B"
    feature_names[2] = "C"

    return X, y, feature_names


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # 1. Data
    # ------------------------------------------------------------------ #
    X, y, feature_names = make_logic_dataset()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_val_df = pd.DataFrame(X_val, columns=feature_names)

    # ------------------------------------------------------------------ #
    # 2. Train Single Tree
    # ------------------------------------------------------------------ #
    print("Training Single Decision Tree (Depth 3)...")
    model = DecisionTreeRegressor(
        max_depth=3, random_state=42, min_impurity_decrease=0.1
    )
    model.fit(X_train, y_train)

    score = model.score(X_val, y_val)
    print(f"Tree R²: {score:.4f}")

    if score < 0.8:
        print("Warning: Model is still too weak. The comparison might fail.")
    else:
        print("Success: Model learned the logic perfectly.")

    # ------------------------------------------------------------------ #
    # 3. Print the Ground Truth (White Box)
    # ------------------------------------------------------------------ #
    print("\n--- VISIBLE TREE STRUCTURE ---")
    print("Check if F0 (A) and F1 (B) are nested inside each other:")
    print("-" * 50)
    print(export_text(model, feature_names=feature_names, max_depth=2))
    print("-" * 50)

    # ------------------------------------------------------------------ #
    # 3b. Global SHAP Analysis
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 55)
    print("  Global SHAP Analysis")
    print("=" * 55)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap,
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    print("\n--- Global SHAP Feature Importance (mean |SHAP|) ---")
    print(shap_importance.to_string(index=False))

    plt.figure()
    shap.summary_plot(shap_values, X_val_df, plot_type="bar", show=False)
    plt.title("Global SHAP Feature Importance — Logic Dataset (Tree)")
    plt.tight_layout()
    plt.savefig("tree_shap_global_bar.png", dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_val_df, show=False)
    plt.title("SHAP Summary (Beeswarm) — Logic Dataset (Tree)")
    plt.tight_layout()
    plt.savefig("tree_shap_summary_beeswarm.png", dpi=150)
    plt.close()

    print("\nSaved SHAP plots: tree_shap_global_bar.png, tree_shap_summary_beeswarm.png")

    # ------------------------------------------------------------------ #
    # 3c. Global SHAP Pairwise Interactions
    # ------------------------------------------------------------------ #
    print("\n--- Global SHAP Pairwise Interactions ---")

    n_feat = len(feature_names)
    shap_inter = explainer.shap_interaction_values(X_val)

    shap_main_effect = np.abs(shap_inter[:, np.arange(n_feat), np.arange(n_feat)]).mean(axis=0)
    W_shap = np.abs(shap_inter).mean(axis=0) * 2.0
    np.fill_diagonal(W_shap, 0.0)

    shap_global_importance = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Global_Importance": shap_main_effect,
    }).sort_values(by="SHAP_Global_Importance", ascending=False).reset_index(drop=True)

    print("\nGlobal SHAP Importance (main effects, diagonal):")
    print(shap_global_importance.to_string(index=False))

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

    total_shap_interactions = shap_pairs_df["SHAP_Interaction_Strength"].sum()
    if total_shap_interactions > 0:
        shap_pairs_df["Interaction_Strength_%"] = (
            (shap_pairs_df["SHAP_Interaction_Strength"] / total_shap_interactions) * 100
        ).map("{:.2f}%".format)
    else:
        shap_pairs_df["Interaction_Strength_%"] = "0.00%"

    print("\nTop 10 Global SHAP Pairwise Interactions:")
    print(shap_pairs_df.head(10).to_string(index=False))

    from sincV2_parallel import plot_interaction_heatmap
    plot_interaction_heatmap(
        W_shap, feature_names, title="Logic Dataset — SHAP Global Interaction Matrix (Tree)"
    )

    # ------------------------------------------------------------------ #
    # 4. Wrap the model for SINC
    # ------------------------------------------------------------------ #
    wrapper = SklearnWrapper(model, model_type="regression")

    # ------------------------------------------------------------------ #
    # 5. Run the full SINC v2 pipeline
    # ------------------------------------------------------------------ #
    output = run_sinc_pipeline(
        wrapped_model=wrapper,
        X_train=X_train,
        X_val_df=X_val_df,
        y_val=y_val,
        feature_names=feature_names,
        subsample_size=100,
        grid_resolution=10,    
        batch_size=2048,
        tau=0.01,              
        algo="leiden",
        method="cpm",
        resolution=0.05,       
        n_repeats=5,
        random_seed=42,
        n_jobs=-1,
        chunk_size=10,
    )

    # ------------------------------------------------------------------ #
    # 6. Print Results
    # ------------------------------------------------------------------ #
    communities = output["communities"]
    # .copy() to ensure we don't trigger SettingWithCopyWarning if output is a view
    results_df  = output["results"].copy()

    print(f"\nCommunities Found: {len(communities)}")
    for i, c in enumerate(communities):
        print(f"  Group {i}: {c}")

    # --- ADDED: Calculate SINC Impact percentage ---
    total_impact = results_df["Impact"].sum()
    if total_impact > 0:
        results_df["Impact_%"] = (
            (results_df["Impact"] / total_impact) * 100
        ).map("{:.2f}%".format)
    else:
        results_df["Impact_%"] = "0.00%"
        
    # Reorder columns slightly so Impact_% is next to Impact (optional but cleaner)
    cols = list(results_df.columns)
    if "Impact" in cols and "Impact_%".endswith("%"):
        cols.insert(cols.index("Impact") + 1, cols.pop(cols.index("Impact_%")))
        results_df = results_df[cols]

    print("\n--- Phase 3: Final results ---")
    print(results_df.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 7. ARCHIPELAGO
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 55)
    print("  Running Archipelago Baseline Comparison")
    print("=" * 55)
    
    archipelago_wrapper = SklearnArchipelagoWrapper(model)
    baseline_instance = X_train.mean(axis=0) 
    
    X_val_df_sample = X_val_df.sample(n=100, random_state=42)
    total_instances = len(X_val_df_sample)
    
    global_scores = defaultdict(float)
    interaction_counts = defaultdict(int)
    
    from src.explainer import Archipelago
    
    print(f"Running explanations for {total_instances} instances...")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        for i, (_, row) in enumerate(X_val_df_sample.iterrows()):
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
                
            if (i + 1) % 20 == 0:
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
        
    df_results = pd.DataFrame(archipelago_results)
    df_results = df_results.sort_values(by="Global_Importance", ascending=False).reset_index(drop=True)
    
    total_archipelago_importance = df_results["Global_Importance"].sum()
    if total_archipelago_importance > 0:
        df_results["Global_Importance_%"] = (
            (df_results["Global_Importance"] / total_archipelago_importance) * 100
        ).map("{:.2f}%".format)
    else:
         df_results["Global_Importance_%"] = "0.00%"
         
    print("\n=== TOP GLOBAL FEATURE COMMUNITIES (ARCHIPELAGO) ===")
    print(df_results.to_string(index=False))