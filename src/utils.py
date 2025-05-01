import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate

import math
import json
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATA_PATH = "data/" # change this to your data path
BASE_CONFIG_PATH = "configs/base_config.json"
ABLATION_CONFIGS_DIR = "configs/ablations/"
OUTPUT_DIR = "output/" 

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
                static_features: List[str]) -> Tuple[StandardScaler, StandardScaler, StandardScaler]:
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
        past_block = df[past_features].dropna().values
        past_data.append(past_block)
        
        # Future features
        route_feats = future_features
        if use_future_speed:
            if "v" not in route_feats:
                route_feats = ["v"] + route_feats
        future_block = df[route_feats].dropna().values
        future_data.append(future_block)

        # Static features at each time
        static_block = df[static_features].dropna().values
        static_data.append(static_block)
        # print(f'{i}/{len(trips)}: Length of trip: {len(df)}')
        

    scaler_past = StandardScaler().fit(np.vstack(past_data))
    scaler_future = StandardScaler().fit(np.vstack(future_data))
    scaler_static = StandardScaler().fit(np.vstack(static_data))

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

def collate_fn_skip_none(batch: List[torch.Tensor]) -> torch.Tensor:
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
