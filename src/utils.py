import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate
import os
import math
import json
import numpy as np
from pathlib import Path
from typing import List
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = os.path.join(ROOT, "data")
SRC_DIR = os.path.join(ROOT, "src")
BASE_CONFIG_PATH = os.path.join(ROOT, "configs/base_config.json")
ABLATION_CONFIGS_DIR = os.path.join(ROOT, "configs/ablations")
OUTPUT_DIR = os.path.join(ROOT, "output")
SAVE_MODELS_PATH = os.path.join(ROOT, "models")

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape: [1, max_len, dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        return x + self.pe[:, :x.size(1), :]

    
def fit_scalers(trips, use_future_speed: bool,
                past_features: List[str],future_features: List[str],
                static_features: List[str]):
    """
    Fit scalers for past, future, and static features.
    """   
    past_data = []
    future_data = []
    static_data = []

    for i, df in enumerate(trips):
        df = df.reset_index(drop=True).copy()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Past features
        if not len(past_features) == 0:
            past_block = df[past_features].dropna().values
            past_data.append(past_block)
        
        # Future features
        if not len(future_features) == 0:
            route_feats = future_features
            if use_future_speed:
                if "v" not in route_feats:
                    route_feats = ["v"] + route_feats
            future_block = df[route_feats].dropna().values
            future_data.append(future_block)

        # Static features at each time
        if not len(static_features) == 0:
            static_block = df[static_features].dropna().values
            static_data.append(static_block)
        # print(f'{i}/{len(trips)}: Length of trip: {len(df)}')
        

    scaler_past = StandardScaler().fit(np.vstack(past_data)) if len(past_features) > 0 else None
    scaler_future = StandardScaler().fit(np.vstack(future_data)) if len(future_features) > 0 else None
    scaler_static = StandardScaler().fit(np.vstack(static_data)) if len(static_features) > 0 else None

    return scaler_past, scaler_future, scaler_static

def detect_charging_jump(df, soc_col: str ="SOC", soc_min=15.0, soc_max=85.0, tol=1.0) -> bool:
    """
    Checks whether a simulated charging jump (from near SOC_min to SOC_max) exists in the trip.
    
    Args:
        df: DataFrame for a single trip
        soc_col: column name for SOC
        soc_min: minimum SOC before charge
        soc_max: maximum SOC after charge
        tol: tolerance in percentage points for fuzzy matching
    
    Returns:
        List of indices where charging jump is detected
    """
    soc = df[soc_col].values
    is_jumped = False
    
    for t in range(len(soc) - 1):
        if (
            soc[t] >= soc_min - tol and soc[t] <= soc_min + tol and
            soc[t+1] >= soc_max - tol and soc[t+1] <= soc_max + tol
        ):
            is_jumped = True

    return is_jumped

def collate_fn_skip_none(batch: List[torch.Tensor]):
    """
    Custom collate function to skip None values in the batch.
    Args:
        batch: List of tensors, some of which may be None.
    Returns:
        Collated tensor with None values skipped.
    """
    batch = [b for b in batch if b is not None]
    return default_collate(batch) if batch else None

def load_config(base_path=BASE_CONFIG_PATH, ablation_path=None):
    # Load base config
    with open(base_path, 'r') as f:
        base_config = json.load(f)

    # If ablation config is provided
    if ablation_path:
        with open(ablation_path, 'r') as f:
            ablation_config = json.load(f)

        if "override" in ablation_config:
            # Merge overrides into base config
            base_config.update(ablation_config["override"])

    return base_config

def get_full_config(config: dict) -> dict:
    """
    Get the full configuration by merging base config with ablation config if provided.
    
    Args:
        config (dict): Base configuration dictionary.
        ablation_path (str): Path to the ablation configuration file.
    
    Returns:
        dict: Merged configuration dictionary.
    """
    # Load base config

    if "override" in config:
        base_config = load_config()
        # Merge overrides into base config
        base_config.update(config["override"])
        return base_config
    else:
        # If no override, return the base config
        return config

def load_model(model, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load the model from the specified path.

    Args:
        model (nn.Module): The model to load.
        model_path (str): Path to the saved model.
        device (str): Device to load the model on.

    Returns:
        nn.Module: The loaded model.
    """
    
    model = torch.load(model_path, map_location=device)

    return model

def load_model_from_config(config: dict, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load the saved mode based on the configuration.

    Args:
        config (dict): Configuration dictionary containing model parameters.
        device (str): Device to load the model on.

    Returns:
        nn.Module: The constructed model.
    """
    from src.models.build import build_model

    config = get_full_config(config)
    model_path = os.path.join(SAVE_MODELS_PATH, f'{config["name"]}.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    
    model = build_model(config)
    model = load_model(model, model_path, device)
    model.to(device)
    return model

def soc_to_range(soc_now, soc_min, soc_drop_pred, distance_rem, soc_drop_sigma=None, eps=1):
    """
    Linearly transforms SOC drop prediction into range estimate.
    
    Arguments:
        soc_now: Current SOC at time t (float)
        soc_min: Minimum usable SOC (e.g., 0.15)
        soc_drop_pred: Predicted SOC drop to complete the trip
        soc_drop_sigma: Uncertainty in SOC drop
        distance_rem: Known distance to end of trip
        
    Returns:
        predicted_range, predicted_range_sigma
    """
    usable_soc = soc_now - soc_min
    soc_drop_pred = max(soc_drop_pred, eps)  # avoid division by zero


    predicted_range = distance_rem * (usable_soc / soc_drop_pred)

    # Estimate uncertainty using first-order approximation
    if soc_drop_sigma is not None:
        range_sigma = abs((distance_rem * usable_soc) / (soc_drop_pred**2)) * soc_drop_sigma
    
        # For the last 3 km, predict range based on distance only (see evaluate_trip in test.py)
        if distance_rem < 3:
            return None, range_sigma
    else:
        range_sigma = None

    return predicted_range, range_sigma

def dynamic_remaining_range(df, soc_min=15, soc_max=85):
    """
    Estimate remaining range dynamically using local energy rate estimates.
    
    Args:
        df: pd.DataFrame with 'SOC' and 'd' columns (distance in km)
        window_len: number of steps ahead to compute local energy rate
        soc_min: minimum usable SOC

    Returns:
        np.ndarray: estimated remaining range at each timestep
    """
    
    df['d'] = (df['v'] * df['t'].diff().fillna(0)/3600).cumsum()

    trip_dist = df['d'].iloc[-1]
    
    soc = df["SOC"].to_numpy()
    dist = df["d"].to_numpy()
    n = len(df)
    
    if soc[0]==soc[-1]:
        print("SOC at start and end are the same. Cannot estimate range.")
        print(soc)
    if trip_dist == 0:
        print("Trip distance is zero. Cannot estimate range.")
        print(dist)

    if detect_charging_jump(df):
        range0 = (soc_max-soc_min)/((soc[0] - soc[-1] + soc_max - soc_min)/trip_dist)
    else:
        range0 = (soc_max-soc_min)/((soc[0] - soc[-1])/trip_dist)
    
    range_est = np.ones(n) * range0
    for i in range(1, n):
        range_est[i] = range_est[i-1] - (soc[i-1] - soc[i])/(soc_max-soc_min)*range0
    

    df["remaining_range"] = range_est
    return df



