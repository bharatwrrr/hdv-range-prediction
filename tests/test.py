import os
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
import fireducks.pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils import soc_to_range, dynamic_remaining_range  # import at the top
from src.utils import load_config
from src.models.losses import gaussian_nll, gaussian_crps

def set_plot_style():
    # Plotting Options
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.5
    plt.rcParams['axes.titlepad'] = 6.0
    plt.rcParams['axes.labelpad'] = 2.0
    plt.rcParams['figure.figsize'] = [16,8]
    plt.rcParams['figure.dpi'] = 160
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['scatter.edgecolors'] = 'face'


def evaluate_trip(dataset, model, trip_id, config=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Evaluates the model on all samples from a specific trip_id (as defined in the dataset).
    Returns a dictionary with prediction results, including predicted range.
    """
    model.eval()

    result = {
        "t": [],
        "predicted_soc_drop": [],
        "true_soc_drop": [],
        "sigma": [],
        "avg_ci_width": []
    }
    
    if config is None:
        config = load_config()
        print("No config provided, using default (baseline) config.")

    trip_df = dataset.trip_data[trip_id].reset_index(drop=True)
    trip_df = dynamic_remaining_range(trip_df)
    
    num_samples = len(dataset.samples)
    last_range = trip_df['remaining_range'].iloc[0]
    with torch.no_grad():
        for idx, sample in enumerate(dataset.samples):
            if sample["trip_idx"] != trip_id:
                continue
            
            item = dataset[idx]
            if item is None:
                continue

            past_seq = item["past_seq"].unsqueeze(0).to(device) if "past_seq" in item else None
            future_seq = item["future_seq"].unsqueeze(0).to(device) if "future_seq" in item else None
            static_feat = item["static_feat"].unsqueeze(0).to(device) if "static_feat" in item else None
            target = item["target"].to(device)
            model.cuda()
            mu, sigma = model(past_seq, future_seq, static_feat)
            
            # Calculate metrics
            result.setdefault("loss_list", []).append((gaussian_nll(mu, sigma, target) + config["loss_alpha"] * mse_loss(mu, sigma)).mean().item())
            result.setdefault("nll_list", []).append(gaussian_nll(mu, sigma, target).mean().item())   
            result.setdefault("crps_list", []).append(gaussian_crps(mu, sigma, target).mean().item())
            result.setdefault("mae_list", []).append(torch.abs(mu - target).sum().item())

            # Extract from raw data
            t_index = sample["t"]
            soc_now = trip_df["SOC"].iloc[t_index]
            distance_rem = trip_df["distance_rem"].iloc[t_index]

            # Compute range estimate
            range_mu, range_sigma = soc_to_range(
                soc_now=soc_now,
                soc_min=config["SOC_min"],
                soc_drop_pred=mu.item(),
                distance_rem=distance_rem,
                soc_drop_sigma=sigma.item()
                
            )
            if range_mu is None:
                range_mu = last_range
            
            last_range = range_mu
            result.setdefault("mu_range", []).append(range_mu)
            result.setdefault("sigma_range", []).append(range_sigma)
            result.setdefault("true_range", []).append(trip_df["remaining_range"].iloc[t_index])
            result.setdefault("true_soc", []).append(trip_df["SOC"].iloc[t_index])
            result["t"].append(t_index)
            result["predicted_soc_drop"].append(mu.item())
            result["true_soc_drop"].append(target.item())
            result["sigma"].append(sigma.item())
            result["avg_ci_width"].append(sigma.item() * 1.64)
            print(f"Evaluating trip {trip_id} - {idx + 1}/{num_samples}", end="\r")
    
    return result

def evaluate_trip_baseline():
    pass

def compute_rolling_efficiency(trip_df: pd.DataFrame,
                               battery_capacity_kwh=564,
                               soc_col="SOC",
                               dist_col="distance_rem",
                               window_size=10000,
                               min_periods=0,
                               init_eff=0.7,
                               eps=1e-5):
    """
    Computes rolling average energy efficiency (kWh/km), starting from init_eff
    and adapting only on moving segments.  Stationary points just carry forward
    the last known efficiency.
    """
    soc = pd.Series(trip_df[soc_col])
    dist_rem = pd.Series(trip_df[dist_col])

    delta_soc = soc.diff(-1).clip(lower=-0.00025)       
    delta_dist = dist_rem - dist_rem.shift(-1)          

    inst_eff = (delta_soc / (delta_dist + eps)) * (battery_capacity_kwh / 100)

    # rolling‐mean over only the moving rows
    rolling_eff = inst_eff.rolling(window=window_size,
                                   min_periods=min_periods).mean()

    adaptive_eff = rolling_eff.fillna(init_eff)
    adaptive_eff = adaptive_eff.ffill().fillna(init_eff)

    # clamp to +-20% of init_eff to avoid wild spikes
    adaptive_eff = adaptive_eff.clip(lower=init_eff * 0.8,upper=init_eff * 1.2)

    return adaptive_eff, inst_eff


def bdm_estimator(soc_now: pd.Series,
                                          distance_remaining: pd.Series,
                                          soc_min=15,
                                          battery_capacity_kwh=564,
                                          efficiency_kwh_per_km=0.2,
                                          eps=1e-5):
    """
    Vectorized version to estimate SOC drop and range using static or variable efficiency.

    Args:
        soc_now (pd.Series): Current SOC values (%).
        distance_remaining (pd.Series): Distance to destination (km).
        soc_min (float): Minimum usable SOC in percent (e.g., 15 for 15%).
        battery_capacity_kwh (float): EV battery capacity in kWh.
        efficiency_kwh_per_km (float or pd.Series): Constant or time-varying energy consumption (kWh/km).
        eps (float): Small value to prevent divide-by-zero.

    Returns:
        pd.Series: Predicted SOC drop for each timestep.
        pd.Series: Estimated remaining range (km) at each timestep.
    """

    usable_soc = (soc_now.subtract(soc_min)).clip(lower=eps)

    # Support scalar or Series for efficiency
    if not isinstance(efficiency_kwh_per_km, pd.Series):
        efficiency_kwh_per_km = pd.Series(efficiency_kwh_per_km, index=soc_now.index)

    energy_needed = distance_remaining.mul(efficiency_kwh_per_km)
    soc_drop_pred = energy_needed / (battery_capacity_kwh + eps) * 100
    range_estimate = (usable_soc / 100 * battery_capacity_kwh) / efficiency_kwh_per_km.add(eps)

    return soc_drop_pred, range_estimate



def get_test_metrics(model, test_loader : DataLoader, criterion, 
                     device = "cuda" if torch.cuda.is_available() else "cpu",
                     return_metrics_list: bool = False):
    """Validate the model on the validation set."""

    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0
    total_nll = 0
    total_crps = 0

    loss_list = []
    mae_list = []
    nll_list = []
    crps_list = []
    avg_ci_width_list = []


    with torch.no_grad():
        for i, batch in enumerate(test_loader):

            past_seq = batch["past_seq"].to(device) if "past_seq" in batch else None
            future_seq = batch["future_seq"].to(device) if "future_seq" in batch else None
            static = batch["static_feat"].to(device) if "static_feat" in batch else None
            target = batch["target"].to(device)

            mu, sigma = model(past_seq, future_seq, static)
            loss = criterion(mu, sigma, target)
            loss = loss.mean()
            loss_list.append(loss)
            avg_ci_width_list.append(sigma.mean().item() * 1.64)
            
            total_loss += loss.item()

            # Accumulate MAE
            batch_mae = torch.abs(mu - target).mean().item()
            batch_nll = gaussian_nll(mu, sigma, target).mean().item()
            batch_crps = gaussian_crps(mu, sigma, target).mean().item()

            mae_list.append(batch_mae)
            nll_list.append(batch_nll)
            crps_list.append(batch_crps)

            total_mae += batch_mae
            total_nll += batch_nll
            total_crps += batch_crps
            total_samples += target.size(0)
            print(f"Evaluating test set - {i + 1}/{len(test_loader)}", end="\r")

    if total_samples == 0:
        return float('nan'), float('nan')
    
    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    avg_nll = total_nll / total_samples
    avg_crps = total_crps / total_samples

    avg_ci_width = torch.mean(torch.tensor(avg_ci_width_list)).item()
    if return_metrics_list:
        return loss_list, mae_list, nll_list, crps_list, avg_ci_width_list
    else:
        return avg_loss, avg_mae, avg_nll, avg_crps, avg_ci_width


def plot_soc_drop_and_range_with_shaded(result_dict, title="SOC Drop Prediction With Shaded"):
    set_plot_style()
    save_path = './plots/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    t = np.array(result_dict["t"])
    true_soc_drop = np.array(result_dict["true_soc_drop"])
    pred_soc = np.array(result_dict["predicted_soc_drop"])
    sigma_soc = np.array(result_dict["sigma"]) if "sigma" in result_dict else np.zeros(pred_soc.shape)
    true_soc = np.array(result_dict["true_soc"])

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # find time when remaining range hits zero (true)
    shade_start = None
    min_soc_index = np.where(true_soc <= min(true_soc))[0][0]

    if "true_range" in result_dict:
        true_range = np.array(result_dict["true_range"])
        zero_idx = np.where(true_range <= 1)[0] # initially this was 0.5, but there were some cases where the minimum range was around 0.8

        if len(zero_idx)>1:
            zero_idx = zero_idx[-1]
        # print(zero_idx)
        if true_range[zero_idx] <= 1:
            shade_start = t[zero_idx]
            if len(shade_start) > 1 or isinstance(shade_start, np.ndarray):
                shade_start = shade_start[0]
    else:
        true_range = None
        raise ValueError("True range is not available in the result dictionary")
    
    #  Plot SOC Drop (subplot 0) 
    axs[0].plot(t, true_soc_drop, label="True SOC Drop", color="black")
    axs[0].plot(t, pred_soc, label="Predicted SOC Drop", color="blue")
    axs[0].plot(t[:min_soc_index], true_soc[:min_soc_index], label="True SOC", color="#B47AEA")
    axs[0].fill_between(t,
                        pred_soc - 1.64 * sigma_soc,
                        pred_soc + 1.64 * sigma_soc,
                        color="blue", alpha=0.2, label="90% CI")

    # Shade region below SOC_min
    axs[0].axhspan(axs[0].get_ylim()[0], 15, color="#210124", alpha=0.3)
    axs[0].text(0.3, 1200/(axs[0].get_ylim()[1]),
                    "Below minimum SOC", color="#210124", rotation=0, va="top")
    
    if "baseline_soc_drop" in result_dict:
        axs[0].plot(range(len(result_dict["baseline_soc_drop"])), result_dict["baseline_soc_drop"], label="Baseline SOC Drop", color="orange")

    if shade_start is not None:
        # Calculate center of shaded region
        center_x = (shade_start + t[-1]) / 2
        center_y = axs[0].get_ylim()[1] * 0.9  # Adjust vertically as needed
        axs[0].axvspan(shade_start, t[-1], color="red", alpha=0.1)
        axs[0].text(center_x, center_y,
                    "Need charging", color="red", rotation=0, va="top", ha="center")
    axs[0].set_ylabel("SOC Drop (%)")
#     axs[0].set_title(title)
    handles, labels = axs[0].get_legend_handles_labels()
    order = [1, 4, 0, 2, 3]  # Change the order as needed
    axs[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper center', ncols=2)
    axs[0].grid(True)

    #  Plot Range Estimate (subplot 1) 
    if "mu_range" in result_dict and "sigma_range" in result_dict:
#         zero_range_idx = [range_ > 0.5 for range_ in result_dict['mu_range']]
        pred_range = np.array(result_dict["mu_range"])
        sigma_range = np.array(result_dict["sigma_range"])
        # only plot up to shade_start
        if shade_start is not None:
            mask = t <= shade_start
            center_x = (shade_start + t[-1]) / 2  # Define center_x when shade_start is not None
        else:
            mask = slice(None)
            center_x = (t[0] + t[-1]) / 2  # Provide a fallback value for center_x
        axs[1].plot(t[mask], pred_range[mask], label="Predicted Range", color="green")
        axs[1].plot(t[mask], true_range[mask], label="True Range", color="black")
        axs[1].fill_between(t[mask],
                            pred_range[mask] - 1.64 * sigma_range[mask],
                            pred_range[mask] + 1.64 * sigma_range[mask],
                            color="green", alpha=0.2, label="90% CI")
        # shade out the rest
        if shade_start is not None:
            center_y = axs[1].get_ylim()[1] * 0.9
            axs[1].axvspan(shade_start, t[-1], color="red", alpha=0.1)
            axs[1].text(center_x, center_y,
                        "Cannot complete\nwithout charge", color="red", rotation=0, va="top", ha="center")
        axs[1].set_ylabel("Remaining Range (km)")
        axs[1].set_xlabel("Time (s)")
        axs[1].grid(True)
        # if max of predicted range (with 90 % confidence high) is too high compared to true range, set y limit
        if max((pred_range) + 1.64 * np.array(sigma_range)) > 2.5 * max(true_range):
            axs[1].set_ylim(-50, 2.5 * max(true_range))

        if "baseline_range" in result_dict:
            if shade_start is not None:
                print(type(shade_start))
                axs[1].plot(range(shade_start), result_dict["baseline_range"][:shade_start], label="Baseline Range", color="orange")
            else:
                axs[1].plot(range(len(result_dict["baseline_range"])), result_dict["baseline_range"], label="Baseline Range", color="orange")
        axs[1].legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}{title}.png")
    plt.show()

