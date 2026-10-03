import numpy as np
from abc import ABC, abstractmethod
import torch
from tqdm import tqdm
import itertools
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, log_loss
from copy import deepcopy
import seaborn as sns
import igraph as ig
import leidenalg as la
import time


# ---------------------------------------------------------
# Phase 1: The Map (Interaction Network Construction)
# --------------------------------------------------------


# ---------------------------------------------------------
# Abstract Base Model Wrapper
# ---------------------------------------------------------

class ModelWrapper(ABC):
    """
    Abstract Base Class to standardize model predictions for SINC.
    """
    
    def __init__(self, target_index=1):
        """
        :param target_index: For classification, which class probability to return.
                             (Default 1 for binary classification positive class).
                             Ignored for single-output regression.
        """
        self.target_index = target_index
        self.device = "cpu"

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Input: 2D Numpy Array (Batch Size, Features)
        Output: 1D Numpy Array (Batch Size, ) - Scalar predictions
        """
        pass

    def to_device(self, device_id):
        """
        Moves the model to a specific device (e.g., 'cuda:0').
        Crucial for Multi-GPU worker distribution.
        """
        self.device = device_id
        return self

# ---------------------------------------------------------
# Scikit-Learn / XGBoost / LightGBM Wrapper
# ---------------------------------------------------------
class SklearnWrapper(ModelWrapper):
    def __init__(self, model, model_type='regression', target_index=1):
        super().__init__(target_index)
        self.model = model
        self.model_type = model_type

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.model_type == 'classification':
            
            preds = self.model.predict_proba(X)
            return preds[:, self.target_index]
        else:
           
            return self.model.predict(X)

# ---------------------------------------------------------
# PyTorch Wrapper
# ---------------------------------------------------------
class PyTorchWrapper(ModelWrapper):
    def __init__(self, model, target_index=0, is_classification=False):
        super().__init__(target_index)
        if torch is None:
            raise ImportError("PyTorch is not installed.")
        
        self.model = model
        self.is_classification = is_classification
        self.model.eval() 

    def to_device(self, device_id):
        """
        Actual implementation of moving the internal model to GPU.
        """
        self.device = device_id
        if torch.cuda.is_available() and 'cuda' in str(device_id):
            self.model = self.model.to(device_id)
        else:
            self.model = self.model.cpu()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # 1. Convert Numpy -> Tensor
        tensor_X = torch.from_numpy(X).float()
        
        # 2. Move to correct device (handled by to_device)
        if 'cuda' in str(self.device):
            tensor_X = tensor_X.to(self.device)

        # 3. Inference (No Grad for speed/memory)
        with torch.no_grad():
            output = self.model(tensor_X)

            # 4. Handle Output Shape
        
            if output.ndim > 1 and output.shape[1] > 1:

                output = output[:, self.target_index]
            
            
            if self.is_classification:
                output = torch.sigmoid(output) 

        # 5. Return to CPU Numpy
        return output.cpu().numpy().flatten()
    
# ---------------------------------------------------------
# Marginal Partial Dependence Computation
# ---------------------------------------------------------

def get_background_data(X, subsample_size=100):
    """Creates a consistent reference set for all PD calculations."""
    N = X.shape[0]
    if N > subsample_size:
        idx = np.random.choice(N, subsample_size, replace=False)
        return X[idx]
    return X

def compute_marginal_pds(model_wrapper, X_background, feature_indices=None, grid_resolution=50):
    """
    Step 1: Pre-computes centered PDs using the consistent background set.
    """
    N, M = X_background.shape
    feature_indices = feature_indices if feature_indices is not None else range(M)
    pds = {}
    
    for f_idx in tqdm(feature_indices, desc="Computing Marginals"):
        # 1. Create Grid based on percentiles of the background (or full X if preferred)
        unique_vals = np.unique(X_background[:, f_idx])
        if len(unique_vals) <= grid_resolution:
            grid = unique_vals
        else:
            percentiles = np.linspace(0, 100, num=grid_resolution)
            grid = np.unique(np.percentile(X_background[:, f_idx], percentiles))

        # 2. Calculate PD values
        pd_values = []
        for val in grid:
            X_temp = X_background.copy()
            X_temp[:, f_idx] = val
            preds = model_wrapper.predict(X_temp)
            pd_values.append(np.mean(preds))
            
        pd_values = np.array(pd_values)
        # 3. CENTER the PD (Subtract the mean of the PD values)
        pd_centered = pd_values - np.mean(pd_values)

        pds[f_idx] = {'grid': grid, 'values': pd_centered}
        
    return pds

# ---------------------------------------------------------
# Lookup for Partial Dependence Values
# ---------------------------------------------------------

def get_pd_for_X(X_column, pd_dict_item):
    """
    Interpolates PD values for the actual data points.
    
    Args:
        X_column (np.array): The actual feature values from the dataset (N,)
        pd_dict_item (dict): The dictionary created above {'grid':..., 'values':...}
    
    Returns:
        np.array: The PD effect for every row in X (N,)
    """
    return np.interp(X_column, pd_dict_item['grid'], pd_dict_item['values'])

# ---------------------------------------------------------
# H-statistic for Pairwise Interactions  (Parallel Version)
# ---------------------------------------------------------

def _compute_single_pair(args):
    """
    Worker function: computes H-statistic for one feature pair.

    Designed to be called by joblib in a subprocess — takes all inputs
    as a single tuple so it is picklable without closures.

    Args:
        args: tuple of
            (idx_j, idx_k,
             grid_j, grid_k,
             pd_j_vals_on_grid, pd_k_vals_on_grid,
             X_background,
             model,            # the raw sklearn model / pipeline
             model_type,       # 'classification' or 'regression'
             target_index,     # which class prob to return
             batch_size)

    Returns:
        ((idx_j, idx_k), h_statistic)
    """
    (idx_j, idx_k,
     grid_j, grid_k,
     pd_j_pre, pd_k_pre,
     X_background,
     model, model_type, target_index,
     batch_size) = args

    # Each loky worker is a fresh process — main-process warning filters
    # do not propagate here. Set the filter locally so the console stays clean.
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    N_bg = X_background.shape[0]

    # 1. Cartesian product of the two grids
    mesh_j, mesh_k = np.meshgrid(grid_j, grid_k, indexing='ij')
    flat_j = mesh_j.flatten()
    flat_k = mesh_k.flatten()
    n_grid_points = len(flat_j)

    # 2. Build evaluation matrix
    X_eval = np.tile(X_background, (n_grid_points, 1))
    X_eval[:, idx_j] = np.repeat(flat_j, N_bg)
    X_eval[:, idx_k] = np.repeat(flat_k, N_bg)

    # 3. Batched prediction — call model directly (no wrapper overhead in subprocess)
    all_preds = []
    for start in range(0, X_eval.shape[0], batch_size):
        chunk = X_eval[start: start + batch_size]
        if model_type == 'classification':
            preds = model.predict_proba(chunk)[:, target_index]
        else:
            preds = model.predict(chunk)
        all_preds.append(preds)

    # 4. Joint PD (centered)
    pd_joint          = np.mean(np.concatenate(all_preds).reshape(n_grid_points, N_bg), axis=1)
    pd_joint_centered = pd_joint - np.mean(pd_joint)

    # 5. Marginals already pre-interpolated onto the grid values
    #    (passed in as pd_j_pre / pd_k_pre to avoid pickling the full pds dict)

    # 6. H-statistic
    numer = np.sum((pd_joint_centered - (pd_j_pre + pd_k_pre)) ** 2)
    denom = np.sum(pd_joint_centered ** 2)

    h_stat = float(np.sqrt(numer / denom)) if denom > 1e-9 else 0.0
    return (idx_j, idx_k), h_stat


def compute_pairwise_interactions(
    model_wrapper,
    X_background,
    pds,
    feature_indices=None,
    batch_size=2048,
    n_jobs=-1,
    chunk_size=50,
):
    """
    Computes Friedman's H-statistic for all feature pairs in parallel.

    Each pair is an independent unit of work — embarrassingly parallel.
    Workers are spawned as separate *processes* (not threads) to bypass
    Python's GIL, which is essential for sklearn pipelines.

    Args:
        model_wrapper   : ModelWrapper instance (SklearnWrapper etc.)
        X_background    : np.ndarray (M, F) — consistent background set
        pds             : dict from compute_marginal_pds
        feature_indices : list of int, default all features in pds
        batch_size      : int, predictions per model call inside each worker
        n_jobs          : int, number of parallel workers.
                          -1 = use all available CPU cores.
                          Set to 1 to disable parallelism (useful for debug).
        chunk_size      : int, pairs sent to each worker per dispatch.
                          Larger = less overhead, more RAM per worker.
                          50 is a safe default for 370-feature datasets.

    Returns:
        dict: {(feat_i, feat_j): h_statistic}
    """
    from joblib import Parallel, delayed

    N_bg, M     = X_background.shape
    feat_ids    = feature_indices if feature_indices is not None else list(pds.keys())
    pairs       = list(itertools.combinations(feat_ids, 2))
    n_pairs     = len(pairs)

    # Extract the raw model and its type so the worker can call it directly.
    # We cannot pickle the wrapper itself easily across processes, but the
    # underlying sklearn pipeline IS picklable.
    raw_model    = model_wrapper.model
    model_type   = getattr(model_wrapper, 'model_type', 'regression')
    target_index = getattr(model_wrapper, 'target_index', 1)

    print(f"  Parallelising {n_pairs} pairs across {n_jobs} workers "
          f"(chunk_size={chunk_size})...")

    # Pre-interpolate marginals for every pair onto their respective grids.
    # This is cheap and avoids sending the full pds dict to every worker.
    def _make_args(idx_j, idx_k):
        grid_j = pds[idx_j]['grid']
        grid_k = pds[idx_k]['grid']

        # Cartesian product of grids (same as inside the worker, but cheap here)
        mesh_j, mesh_k = np.meshgrid(grid_j, grid_k, indexing='ij')
        flat_j = mesh_j.flatten()
        flat_k = mesh_k.flatten()

        pd_j_pre = np.interp(flat_j, pds[idx_j]['grid'], pds[idx_j]['values'])
        pd_k_pre = np.interp(flat_k, pds[idx_k]['grid'], pds[idx_k]['values'])

        return (idx_j, idx_k,
                grid_j, grid_k,
                pd_j_pre, pd_k_pre,
                X_background,
                raw_model, model_type, target_index,
                batch_size)

    # Build all argument tuples (list of tuples, one per pair)
    print("  Preparing argument tuples...")
    all_args = [_make_args(j, k) for j, k in tqdm(pairs, desc="Building args")]

    # Dispatch — loky backend spawns true subprocesses
    print("  Running parallel H-statistic computation...")
    results = Parallel(
        n_jobs     = n_jobs,
        backend    = 'loky',      # process-based, bypasses GIL
        verbose    = 5,           # prints progress every ~5% of jobs done
        batch_size = chunk_size,
    )(delayed(_compute_single_pair)(args) for args in all_args)

    # Collect into dict
    interaction_scores = {pair: score for pair, score in results}
    print(f"  Done. {len(interaction_scores)} H-statistics computed.")

    return interaction_scores

# ---------------------------------------------------------
# Process Interaction Scores into DataFrame and Matrix
# ---------------------------------------------------------

def build_interaction_network(interaction_scores, feature_names, tau=0.0):
    """
    Phase 1: The Map (Interaction Network Construction)
    Populates the symmetric adjacency matrix W based on H-statistics.

    Args:
        interaction_scores (dict): Output from compute_pairwise_interactions {(i, j): h_stat}
                                   Note: These are usually H (sqrt), so we square them for H^2.
        feature_names (list): List of feature names.
        tau (float): Noise threshold. If H^2 < tau, we set weight to 0.

    Returns:
        pd.DataFrame: The Weight Matrix W (D x D).
    """
    D = len(feature_names)
    W = pd.DataFrame(0.0, index=feature_names, columns=feature_names)

    print(f"Constructing Network Map with threshold tau={tau}...")

    for (i, j), h_value in interaction_scores.items():

        h_squared = h_value ** 2 

        name_i = feature_names[i]
        name_j = feature_names[j]

        if h_squared < tau:
            weight = 0.0
        else:
            weight = h_squared

        # Populate Symmetric Matrix
        W.at[name_i, name_j] = weight
        W.at[name_j, name_i] = weight

    return W


def plot_interaction_heatmap(W, feature_names, title='Pairwise Interaction Network (H-Statistics)', cmap='hot'):
    """
    Plots a raw heatmap of the feature-to-feature interaction matrix (W).
    
    Args:
        W (pd.DataFrame or np.ndarray): The symmetric weight matrix from the pipeline.
        feature_names (list): List of feature names for axis labeling.
        title (str): Title of the plot.
        cmap (str): Colormap to use (e.g., 'hot', 'viridis', 'magma').
    """

    plt.figure(figsize=(10, 8))
    
    # We use W.values if it's a DataFrame, otherwise assume it's an array
    matrix_data = W.values if hasattr(W, 'values') else W
    
    # Plotting using imshow for that classic "hot" look
    img = plt.imshow(matrix_data, cmap=cmap, aspect='auto')
    plt.colorbar(img, label='Interaction Strength ($H^2$)')
    
    # Labels and Ticks
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.title(title)
    
    # Map the ticks to the feature names
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.yticks(range(len(feature_names)), feature_names)
    
    plt.tight_layout()
    plt.show()

#---------------------------------------------------------
# Phase 2: The Discovery (Community Detection)
#---------------------------------------------------------

class InteractionGraph:
    """
    Phase 2: The Discovery (Community Detection)
    Converts the Interaction Matrix W into a Graph and finds clusters.
    Supports Louvain, Leiden, Modularity, and CPM with iterative refinement.
    """
    
    def __init__(self, adjacency_matrix_df):
        self.matrix = adjacency_matrix_df
        self.G = self._build_graph()
        self.communities = []
        
    def _build_graph(self):
        G = nx.from_pandas_adjacency(self.matrix)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G

    def _get_igraph(self):
        """Helper to convert NetworkX graph to igraph for Leiden/CPM."""
        import igraph as ig
        edges = [tuple(e) for e in self.G.edges()]
        weights = [self.G[u][v]['weight'] for u, v in self.G.edges()]
        
        g_ig = ig.Graph()
        g_ig.add_vertices(list(self.G.nodes()))
        g_ig.add_edges(edges)
        g_ig.es['weight'] = weights
        return g_ig

    def detect_communities(self, algorithm='leiden', method='cpm', resolution=1.0, seed=42, n_iterations=-1):
        """
        Detects communities using various algorithms and objective functions.
        
        :param algorithm: 'louvain' or 'leiden'
        :param method: 'modularity' or 'cpm'
        :param resolution: Resolution parameter (>0).
        :param seed: Random seed.
        :param n_iterations: (Leiden only) Number of iterations. -1 runs until stable.
        """

        # --- OPTION 1: LOUVAIN (via NetworkX) ---
        if algorithm.lower() == 'louvain':
            # NetworkX Louvain is simpler but lacks the refinement phase of Leiden
            louvain_sets = nx.community.louvain_communities(
                self.G, weight='weight', resolution=resolution, seed=seed
            )
            raw_communities = [list(c) for c in louvain_sets]

        # --- OPTION 2: LEIDEN (via leidenalg) ---
        elif algorithm.lower() == 'leiden':
            g_ig = self._get_igraph()
            
            # Map method to the specific partition class
            # Note: RBConfigurationVertexPartition is used for Modularity to support resolution_parameter
            partition_map = {
                'modularity': la.RBConfigurationVertexPartition,
                'cpm': la.CPMVertexPartition
            }
            
            if method.lower() not in partition_map:
                raise ValueError(f"Method '{method}' not supported. Use 'modularity' or 'cpm'.")

            # Leiden includes a refinement step in each iteration
            # This is significantly more robust than Louvain for finding global optima
            partition = la.find_partition(
                g_ig, 
                partition_map[method.lower()], 
                weights='weight', 
                resolution_parameter=resolution,
                seed=seed,
                n_iterations=n_iterations
            )
            
            node_names = list(self.G.nodes())
            raw_communities = [[node_names[idx] for idx in comm] for comm in partition]

        else:
            raise ValueError("Algorithm must be 'louvain' or 'leiden'")

        # Post-process: sort by size and store
        self.communities = sorted([sorted(c) for c in raw_communities], key=len, reverse=True)
        return self.communities

    def plot_graph(self, title="Feature Interaction Network"):
        """
        Visualizes the graph with:
        - Node colors by community
        - Edge thickness and transparency based on interaction strength (H^2)
        """
        if not self.communities:
            print("Please run detect_communities() first.")
            return
             
        plt.figure(figsize=(12, 10))
        
        # 1. Map nodes to community IDs for coloring
        partition_map = {node: i for i, comm in enumerate(self.communities) for node in comm}
        node_colors = [partition_map[node] for node in self.G.nodes()]
        
        # 2. Position nodes using Spring Layout
        # weight='weight' tells the algorithm to pull strongly connected nodes closer
        pos = nx.spring_layout(self.G, weight='weight', k=0.5, iterations=100, seed=42)
        
        # 3. Extract edge weights for visual mapping
        edges = self.G.edges(data=True)
        weights = np.array([d['weight'] for u, v, d in edges])
        
        # Normalize weights for thickness (e.g., scale between 0.5 and 8.0)
        if len(weights) > 0:
            # We use a linear scaling, but you could use log/exp if weights vary wildly
            max_w = weights.max()
            min_w = weights.min()
            widths = 1 + 7 * (weights - min_w) / (max_w - min_w + 1e-9)
            # Map alpha (opacity) between 0.1 and 0.8
            alphas = 0.1 + 0.7 * (weights - min_w) / (max_w - min_w + 1e-9)
        else:
            widths = 1.0
            alphas = 0.3

        # 4. Draw Nodes
        nx.draw_networkx_nodes(
            self.G, pos, 
            node_color=node_colors, 
            cmap=plt.cm.tab20, 
            node_size=700, 
            edgecolors="white", # Adds a border around nodes
            alpha=0.9
        )
        
        # 5. Draw Labels
        nx.draw_networkx_labels(self.G, pos, font_size=9, font_weight="bold")
        
        # 6. Draw Edges individually to apply per-edge alpha
        for i, (u, v, d) in enumerate(edges):
            nx.draw_networkx_edges(
                self.G, pos, 
                edgelist=[(u, v)], 
                width=widths[i], 
                alpha=alphas[i], 
                edge_color="gray"
            )
        
        plt.title(title, fontsize=15)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

#---------------------------------------------------------
# Phase 3: The Validation (Synergy Scoring)
#---------------------------------------------------------

def calculate_synergy_score(wrapped_model, X_val_df, y_val, communities, metric_func, n_repeats=5):
    """
    Evaluates Synergy and Importance using the ModelWrapper.
    
    Args:
        wrapped_model: Instance of ModelWrapper (SklearnWrapper or PyTorchWrapper)
        X_val_df (pd.DataFrame): Validation features
        y_val (np.array): Validation targets
        communities (list): List of feature groups (from Phase 2)
        metric_func: Loss function (MSE or LogLoss)
        n_repeats: Number of permutations to average
    """
    # Use the wrapper to get consistent baseline predictions
    original_preds = wrapped_model.predict(X_val_df.values)
    baseline_score = metric_func(y_val, original_preds)
    
    results = []
    
    for i, group in enumerate(communities):
        # 1. Calculate Joint Impact (Shuffle all features in the group together)
        group_scores = []
        for _ in range(n_repeats):
            X_temp = X_val_df.copy()
            shuffled_idx = np.random.permutation(len(X_temp))
            # Shuffle the block of features
            X_temp.loc[:, group] = X_temp.loc[:, group].values[shuffled_idx]
            
            # Use wrapped_model here to handle PyTorch/Sklearn logic automatically
            new_preds = wrapped_model.predict(X_temp.values)
            group_scores.append(metric_func(y_val, new_preds))
        
        avg_group_score = np.mean(group_scores)
        joint_impact = avg_group_score - baseline_score
        
        # 2. Calculate Synergy (Interaction effect above the sum of individuals)
        synergy = 0.0
        sum_ind_impact = 0.0
        
        if len(group) > 1:
            for feature in group:
                feat_scores = []
                for _ in range(n_repeats):
                    X_temp_f = X_val_df.copy()
                    shuffled_idx_f = np.random.permutation(len(X_temp_f))
                    X_temp_f.loc[:, feature] = X_temp_f.loc[:, feature].values[shuffled_idx_f]
                    
                    feat_preds = wrapped_model.predict(X_temp_f.values)
                    feat_scores.append(metric_func(y_val, feat_preds))
                
                ind_impact = np.mean(feat_scores) - baseline_score
                sum_ind_impact += ind_impact
            
            synergy = joint_impact - sum_ind_impact
        else:
            sum_ind_impact = joint_impact
            synergy = 0.0

        results.append({
            'Group ID': i,           
            'Features': ", ".join(group),
            'Size': len(group),
            'Impact': joint_impact,
            'Synergy': synergy
        })
        
    df = pd.DataFrame(results)
    
    # --- 3. DYNAMIC CLASSIFICATION ---
    if not df.empty:
        max_impact = df['Impact'].max()
        # Threshold for noise (anything less than 1% of max impact)
        noise_threshold = max(0.001, 0.01 * max_impact)
        
        def classify(row):
            if row['Impact'] < noise_threshold:
                return "Weak/Noise"
            if row['Size'] == 1:
                return "Additive (Solo)"
            # If synergy is more than 20% of the total impact, it's highly synergistic
            if abs(row['Synergy']) > (0.2 * row['Impact']):
                return "Synergistic (Complex)"
            return "Additive (Group)"

        df['Nature'] = df.apply(classify, axis=1)
        df = df.sort_values('Impact', ascending=False)

    return df

# ---------------------------------------------------------
# Full SINC Pipeline Wrapper
# ---------------------------------------------------------

def run_sinc_pipeline(
    wrapped_model,
    X_train,
    X_val_df,
    y_val,
    feature_names,
    subsample_size=100,
    grid_resolution=50,
    batch_size=2048,
    tau=0.04,
    algo='leiden',
    method='cpm',
    resolution=1.0,
    n_repeats=5,
    random_seed=42,
    n_jobs=-1,
    chunk_size=50,
):
    """
    Executes the SINC (Synergy Interaction Network Community) pipeline.

    Args:
        wrapped_model (ModelWrapper): An instance of SklearnWrapper or PyTorchWrapper. 
            Standardizes .predict() calls across different framework types.
        X_train (np.ndarray): Training data used to calculate feature distributions 
            and percentile-based grids for H-statistics.
        X_val_df (pd.DataFrame): Validation features used in Phase 3 to calculate 
            Out-of-Sample (OOS) Synergy and Impact scores.
        y_val (np.array): Ground-truth targets for the validation set. Used to 
            evaluate loss metrics during permutation tests.
        feature_names (list of str): The human-readable names of the features, 
            used for labeling dataframes and graph nodes.
        subsample_size (int, optional): The number of rows used as the background 
            reference set. A consistent set is used for both marginal and joint 
            PDs to ensure mathematical validity. Defaults to 100.
        grid_resolution (int, optional): The number of points in the 1D and 2D 
            Partial Dependence grids. Higher values capture more detail but 
            increase computation time. Defaults to 50.
        batch_size (int, optional): The number of samples sent to the model 
            per prediction call. Adjust based on available RAM/VRAM. Defaults to 2048.
        tau (float, optional): The H-squared interaction threshold. Pairs with 
            an interaction strength below this value are excluded from the 
            Network Map. Defaults to 0.04.
        algo (str, optional): The community detection algorithm ('leiden' or 'louvain'). 
            Leiden is recommended for better stability. Defaults to 'leiden'.
        method (str, optional): The objective function for clustering ('cpm' or 'modularity'). 
            CPM is often preferred for feature interaction networks. Defaults to 'cpm'.
        resolution (float, optional): Controls the granularity of clusters. 
            Values >1.0 result in more/smaller groups; <1.0 result in fewer/larger groups. 
            Defaults to 1.0.
        n_repeats (int, optional): The number of permutations performed for each 
            group in Phase 3 to ensure stable Synergy/Impact scores. Defaults to 5.
        random_seed (int, optional): Seed for random number generators to ensure 
            reproducibility of subsampling and clustering. Defaults to 42.

    Returns:
        dict: A dictionary containing:
            - 'results' (pd.DataFrame): Table of groups, their nature, impact, and synergy.
            - 'communities' (list): Raw lists of features belonging to each group.
            - 'weights' (pd.DataFrame): The D x D feature-level interaction matrix.
            - 'meta_weights' (pd.DataFrame): The G x G group-level interaction matrix.
            - 'graph_obj' (InteractionGraph): The object used for network visualization.
    """
    start_total = time.time()
    np.random.seed(random_seed)
    
    # --- PHASE 0: CONSISTENCY SETUP ---
    print("\n" + "="*50)
    print("PHASE 0: Background Initialization")
    print("="*50)
    X_bg = get_background_data(X_train, subsample_size=subsample_size)
    print(f" > Using a consistent reference set of {len(X_bg)} samples.")

    # --- PHASE 1: MAPPING ---
    print("\n" + "="*50)
    print("PHASE 1: Interaction Mapping")
    print("="*50)
    
    print(f" > Computing Marginal PDs (Resolution: {grid_resolution})...")
    pds = compute_marginal_pds(
        wrapped_model, 
        X_bg, 
        grid_resolution=grid_resolution
    )
    
    print(f" > Computing Pairwise Interactions (Batch Size: {batch_size}, "
          f"n_jobs: {n_jobs}, chunk_size: {chunk_size})...")
    scores = compute_pairwise_interactions(
        wrapped_model,
        X_bg,
        pds,
        batch_size=batch_size,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
    )
    
    print(f" > Building Network Map (Threshold tau: {tau})...")
    W = build_interaction_network(scores, feature_names, tau=tau) 

    # --- PHASE 2: DISCOVERY ---
    print("\n" + "="*50)
    print("PHASE 2: Community Discovery")
    print("="*50)
    
    ig_obj = InteractionGraph(W)
    communities = ig_obj.detect_communities(
        algorithm=algo, 
        method=method, 
        resolution=resolution,
        seed=random_seed
    )
    
    print(f" > Detected {len(communities)} interaction groups.")

    # --- PHASE 3: VALIDATION ---
    print("\n" + "="*50)
    print("PHASE 3: Synergy Validation")
    print("="*50)
    
    is_classification = False
    if hasattr(wrapped_model, 'model_type'):
        is_classification = wrapped_model.model_type == 'classification'
    elif hasattr(wrapped_model, 'is_classification'):
        is_classification = wrapped_model.is_classification
        
    metric = log_loss if is_classification else mean_squared_error
    
    results = calculate_synergy_score(
        wrapped_model, 
        X_val_df, 
        y_val, 
        communities, 
        metric, 
        n_repeats=n_repeats
    )
    
    # --- PHASE 4: META-ANALYSIS ---
    meta_df = analyze_inter_group_interactions(W, communities)
    
    elapsed = time.time() - start_total
    print(f"\nSINC Complete in {elapsed:.2f} seconds.")
    
    return {
        'results': results,
        'communities': communities,
        'weights': W,
        'meta_weights': meta_df,
        'graph_obj': ig_obj
    }

# ---------------------------------------------------------
# Macro - Interaction
# ---------------------------------------------------------

def analyze_inter_group_interactions(adjacency_df, communities):
    """
    Collapses the feature-level adjacency matrix into a group-level matrix.
    """
    n_groups = len(communities)
    meta_matrix = np.zeros((n_groups, n_groups))
    
    print(f" Collapsing network into {n_groups}x{n_groups} Meta-Graph...")
    
    # Iterate through every pair of groups
    for i in range(n_groups):
        for j in range(n_groups):
            if i == j:
                # Diagonal: Internal cohesion (Sum of weights INSIDE the group)
                # We halve it because the matrix is symmetric and we don't want double counting
                group_features = communities[i]
                sub_matrix = adjacency_df.loc[group_features, group_features]
                # Sum upper triangle
                total_weight = sub_matrix.values[np.triu_indices(len(group_features), k=1)].sum()
                meta_matrix[i, j] = total_weight
            else:
                # Off-Diagonal: Interaction BETWEEN Group i and Group j
                group_i_feats = communities[i]
                group_j_feats = communities[j]
                
                # Extract the sub-rectangle connecting the two groups
                sub_matrix = adjacency_df.loc[group_i_feats, group_j_feats]
                meta_matrix[i, j] = sub_matrix.values.sum()

    # Convert to DataFrame for readability
    meta_df = pd.DataFrame(
        meta_matrix, 
        index=[f"Grp {i}" for i in range(n_groups)],
        columns=[f"Grp {i}" for i in range(n_groups)]
    )
    
    return meta_df

def plot_meta_heatmap(meta_df, results_df=None, title=None):
    """
    Plots the Meta-Interaction Map.
    
    Args:
        meta_df (pd.DataFrame): The group-level adjacency matrix.
        results_df (pd.DataFrame, optional): The results table from Phase 3. 
                                             If provided, filters out 'Weak/Noise' groups.
                                             If None, plots the full matrix.
        title (str, optional): Custom title for the plot.
    """
    matrix_to_plot = meta_df
    
    # --- Logic for Filtering ---
    if results_df is not None:
        # Filter: Keep only groups that are NOT 'Weak/Noise'
        active_df = results_df[results_df['Nature'] != 'Weak/Noise']
        
        if active_df.empty:
            print("Warning: No active groups found. Plotting full matrix instead.")
        else:
            # Get Group IDs and convert to labels (e.g., 0 -> "Grp 0")
            active_ids = sorted(active_df['Group ID'].values)
            keep_labels = [f"Grp {i}" for i in active_ids]
            
            # Intersection: ensure we only ask for labels that actually exist in the matrix
            valid_labels = [lbl for lbl in keep_labels if lbl in meta_df.index]
            
            # Apply filter
            matrix_to_plot = meta_df.loc[valid_labels, valid_labels]
            
            # Set default title for filtered view if user didn't provide one
            if title is None:
                title = "Meta-Interaction Map (Active Groups Only)"

    # --- Logic for Default Title ---
    if title is None:
        title = "Meta-Interaction Map (Full Network)"

    # --- Plotting ---
    # Dynamic size: smaller if few groups, standard size otherwise
    n_groups = len(matrix_to_plot)
    figsize = (8, 6) if n_groups < 10 else (10, 8)
    
    plt.figure(figsize=figsize)
    sns.heatmap(matrix_to_plot, annot=True, fmt=".2f", cmap="viridis", linewidths=.5)
    plt.title(title)
    plt.show()

# ---------------------------------------------------------
# IMPORTANT — Windows multiprocessing safety
# ---------------------------------------------------------
# On Windows, joblib spawns new processes by starting a fresh Python
# interpreter that imports this module. If any top-level executable code
# exists outside of an `if __name__ == '__main__':` guard, it will run
# again in every worker process — causing infinite recursion / fork bombs.
#
# Rule: this file is a library. Keep all executable code in the caller
# (e.g. sinc_stress_experiment.py) inside `if __name__ == '__main__':`.
# This file intentionally has no top-level executable code.