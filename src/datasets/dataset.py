import os
import torch
import numpy as np
from torch.utils.data import Dataset
from ..utils import detect_charging_jump
from typing import List

import fireducks.pandas as pd
from ..utils import fit_scalers, DATA_PATH
from sklearn.model_selection import train_test_split

class SOCDropDataset(Dataset):
    def __init__(self, trips, scaler_past, scaler_future, scaler_static,
                 past_len: int, use_future_speed: bool,
                 past_features: List[str],
                 future_features: List[str],
                 static_features: List[str],
                 future_seq_len: int,
                 stride: int = 30, 
                 SOC_min=0.15, SOC_max=0.85,
                 verbose=False):
        self.samples = []
        self.scaler_past = scaler_past
        self.scaler_future = scaler_future
        self.scaler_static = scaler_static
        self.use_future_speed = use_future_speed
        self.past_len = past_len
        self.past_features = past_features
        self.future_features = future_features
        self.static_features = static_features
        self.future_seq_len = future_seq_len
        self.stride = stride
        self.trip_data = {}
        self._has_charging = False

        
        if 'road_type_enc' in self.past_features:
            self.past_features.remove('road_type_enc')
        if 'road_type_enc' in self.future_features:
            self.future_features.remove('road_type_enc')

            
        for trip_idx, df in enumerate(trips):
            df = df.reset_index(drop=True).copy()

            if len(df)<(15*60):
                # skipping short trips
                continue

            # Optionally check that all expected road_type_{i} columns are present
            expected_cols = [f'road_type_{i}' for i in range(15)]
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 0.0  # fill missing ones with zeros just in case

            # Build base feature arrays
            arrays = {
                "v": df["v"].to_numpy(np.float32),
                "a": df["a"].to_numpy(np.float32),
                "P": df["P"].to_numpy(np.float32),
                "SOC": df["SOC"].to_numpy(np.float32),
                "E": df["E"].to_numpy(np.float32), #TODO: added this
                "Ta": df["Ta"].to_numpy(np.float32),
                "theta": df["theta"].to_numpy(np.float32),
                "alt": df["alt"].to_numpy(np.float32),
                "m": df["m"].to_numpy(np.float32),
                "distance_to_end": df["distance_to_end"].to_numpy(np.float32),
                "distance_rem": df["distance_rem"].to_numpy(np.float32)
            }

            # Add one-hot road type columns
            for col in expected_cols:  # expected_cols = [f"road_type_{i}" for i in range(...)]
                arrays[col] = df[col].to_numpy(np.float32)
            self.trip_data[trip_idx] = df
    
            if detect_charging_jump(df):
                self._has_charging = True
                soc_end = SOC_min - (SOC_max - df["SOC"].iloc[-1]) 
                soc_end_charged = df["SOC"].iloc[-1]
            else:
                soc_end = df["SOC"].iloc[-1]
            trip_len = len(df)

            soc_diff_charged_prev = 0 

            is_past_charged = False
            for t in range(self.past_len, trip_len - 30, self.stride):
                if t - past_len < 0: 
                    continue

                soc_now = df["SOC"].iloc[t]

                if self._has_charging:
                    if is_past_charged:
                        soc_drop = soc_now - soc_end_charged
                    else:
                        soc_diff = soc_now - soc_end
                        soc_diff_charged = soc_now - soc_end_charged
                        if (soc_diff_charged>0) and (soc_diff_charged_prev<0):
                            # at charging point, the above product changes sign
                            soc_drop = soc_diff_charged
                            is_past_charged = True
                        else:
                            soc_drop = soc_diff
                    soc_diff_charged_prev = soc_diff_charged
                else:
                    soc_drop = soc_now - soc_end
                

                if soc_drop < 0 or soc_drop > 100:
                    continue

                self.samples.append({"trip_idx": trip_idx, "t": t, "soc_drop": soc_drop})
            if verbose:
                print(f"[{trip_idx+1}/{len(trips)}] Trip length: {trip_len}, Samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        sample = self.samples[idx]
        trip = self.trip_data[sample["trip_idx"]]
        t = sample["t"]
        soc_drop = sample["soc_drop"]

        # --- Past sequence
        past_window = [trip[col][t - self.past_len:t] for col in self.past_features]
        if any(len(x) != self.past_len for x in past_window):
            return None
        past_seq = np.stack(past_window, axis=1)
        past_seq = self.scaler_past.transform(past_seq)

        # --- Future sequence (fixed number of future waypoints)
        road_type_cols = [col for col in trip.columns if col.startswith("road_type_") and col != 'road_type_enc']
        future_feature_cols = self.future_features + road_type_cols
        if self.use_future_speed:
            future_feature_cols = ["v"] + future_feature_cols

        trip_len = len(trip)
        end_idx = trip_len
        window = trip.iloc[t+1:end_idx]

        if len(window) < self.future_seq_len:
            return None

        idxs = np.linspace(0, len(window)-1, self.future_seq_len, dtype=int)
        selected = window.iloc[idxs]

        # Apply scaling to continuous features only
        future_scaled = self.scaler_future.transform(selected[self.future_features].values)
        road_onehot = selected[road_type_cols].values

        future_seq = np.concatenate([future_scaled, road_onehot], axis=1)

        # --- Static features
        static_feat = np.array([trip[col][t] for col in self.static_features], dtype=np.float32)
        static_feat = self.scaler_static.transform([static_feat])[0]

        return {
            "past_seq": torch.tensor(past_seq, dtype=torch.float32),
            "future_seq": torch.tensor(future_seq, dtype=torch.float32),
            "static_feat": torch.tensor(static_feat, dtype=torch.float32),
            "target": torch.tensor(soc_drop, dtype=torch.float32)
        }


def build_datasets(config: dict, verbose: bool = False):
    """
    Build datasets for training and validation.

    Args:
        config (dict): Configuration dictionary containing dataset parameters.

    Returns:
        tuple: Training and validation datasets.
    """

    # Loading all data
    trips = []
    num_files = len(os.listdir(DATA_PATH))
    if verbose:
        print(f"Number of files: {num_files}")
    for i, file in enumerate(os.listdir(DATA_PATH)):
        # if i>60:
        #     break
        if not file.endswith('.csv'):
            continue
        
        df = pd.read_csv(os.path.join(DATA_PATH, file))
        dist = (df['v']*5/18).cumsum()/1000

        # Remove all 'gps_' columns
        df = df.loc[:, ~df.columns.str.startswith('gps_')]

        # Add distance remaining column
        df['distance_rem'] = dist.iloc[-1] - dist
        trips.append(df)
        if verbose:
            print(f'Loading trips from data: {i}/{num_files}', end='\r')
        
    # Load trips
    train_val_trips = trips[:int(len(trips)*0.8)]
    test_trips = trips[int(len(trips)*0.8):]
    train_trips, val_trips = train_test_split(train_val_trips, test_size=0.2, random_state=42)

    # Fit scalers
    scaler_past, scaler_future, scaler_static = fit_scalers(
        train_val_trips,
        use_future_speed=config["use_future_speed"],
        past_features=config["past_features"],
        future_features=config["future_features"],
        static_features=config["static_features"]
    )
    if verbose:
        print("Scalers fitted")

    # Create datasets
    train_dataset = SOCDropDataset(
        trips=train_trips,
        scaler_past=scaler_past,
        scaler_future=scaler_future,
        scaler_static=scaler_static,
        past_len=config["past_seq_len"],
        use_future_speed=config["use_future_speed"],
        past_features=config["past_features"],
        future_features=config["future_features"],
        static_features=config["static_features"],
        future_seq_len=config["future_seq_len"],
        SOC_min=config["SOC_min"],
        SOC_max=config["SOC_max"]
    )
    if verbose:
        print(f"Train dataset size: {len(train_dataset)}")

    val_dataset = SOCDropDataset(
        trips=val_trips,
        scaler_past=scaler_past,
        scaler_future=scaler_future,
        scaler_static=scaler_static,
        past_len=config["past_seq_len"],
        use_future_speed=config["use_future_speed"],
        past_features=config["past_features"],
        future_features=config["future_features"],
        static_features=config["static_features"],
        future_seq_len=config["future_seq_len"],
        SOC_min=config["SOC_min"],
        SOC_max=config["SOC_max"]
    )
    if verbose:
        print(f"Validation dataset size: {len(val_dataset)}")

    test_dataset = SOCDropDataset(
        trips=test_trips,
        scaler_past=scaler_past,
        scaler_future=scaler_future,
        scaler_static=scaler_static,
        past_len=config["past_seq_len"],
        use_future_speed=config["use_future_speed"],
        past_features=config["past_features"],
        future_features=config["future_features"],
        static_features=config["static_features"],
        future_seq_len=config["future_seq_len"],
        SOC_min=config["SOC_min"],
        SOC_max=config["SOC_max"]
    )
    if verbose:
        print(f"Test dataset size: {len(test_dataset)}")
        print("Datasets created")

    return train_dataset, val_dataset, test_dataset