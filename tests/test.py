import os
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
import numpy as np
import matplotlib.pyplot as plt
from src.utils import soc_to_range, dynamic_remaining_range  # import at the top
from src.utils import load_config
from src.models.losses import gaussian_nll, gaussian_crps

def set_plot_style():
    # Plotting Options
    # Plot Options
    # plt.rcParams['font.family'] = 'Times New Roman'
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
    # matplotlib.rcParams.update(matplotlib.rcParamsDefault)





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
            result.setdefault("loss_list", []).append((gaussian_nll(mu, sigma, target) + config["loss_alpha"] * F.mse_loss(mu, sigma)).mean().item())
            result.setdefault("nll_list", []).append(gaussian_nll(mu, sigma, target).mean().item())   
            result.setdefault("crps_list", []).append(gaussian_crps(mu, sigma, target).mean().item())
            result.setdefault("mae_list", []).append(torch.abs(mu - target).sum().item())

            # --- Extract from raw data
            t_index = sample["t"]
            soc_now = trip_df["SOC"].iloc[t_index]
            distance_rem = trip_df["distance_rem"].iloc[t_index]

            # --- Compute range estimate
            range_mu, range_sigma = soc_to_range(
                soc_now=soc_now,
                soc_min=config["SOC_min"],
                soc_drop_pred=mu.item(),
                soc_drop_sigma=sigma.item(),
                distance_rem=distance_rem
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

            print(f"Evaluating trip {trip_id} - {idx + 1}/{num_samples}", end="\r")

    return result





def get_test_metrics(model, test_loader : DataLoader, criterion, device = "cuda" if torch.cuda.is_available() else "cpu"):
    """Validate the model on the validation set."""

    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_mape = 0.0
    total_samples = 0
    total_nll = 0
    total_crps = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):

            past_seq = batch["past_seq"].to(device) if "past_seq" in batch else None
            future_seq = batch["future_seq"].to(device) if "future_seq" in batch else None
            static = batch["static_feat"].to(device) if "static_feat" in batch else None
            target = batch["target"].to(device)

            mu, sigma = model(past_seq, future_seq, static)
            loss = criterion(mu, sigma, target)
            loss = loss.mean()
            
            total_loss += loss.item()

            # Accumulate MAE
            batch_mae = torch.abs(mu - target).sum().item()
            batch_mape = torch.abs((mu - target) / target).sum().item() * 100 
            batch_nll = gaussian_nll(mu, sigma, target).sum().item()
            batch_crps = gaussian_crps(mu, sigma, target).sum().item()
            total_mae += batch_mae
            total_mape += batch_mape
            total_nll += batch_nll
            total_crps += batch_crps
            total_samples += target.size(0)
            print(f"Evaluating test set - {i + 1}/{len(test_loader)}", end="\r")

    if total_samples == 0:
        return float('nan'), float('nan')
    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    avg_mape = total_mape / total_samples
    avg_nll = total_nll / total_samples
    avg_crps = total_crps / total_samples
    return avg_loss, avg_mae, avg_mape, avg_nll, avg_crps


def plot_results(result_dict, title="SOC Drop Prediction"):
    t = result_dict["t"]
    true = result_dict["true_soc_drop"]
    pred = result_dict["predicted_soc_drop"]
    sigma = result_dict["sigma"]

    plt.figure(figsize=(12, 5))
    plt.plot(t, true, label="True SOC Drop", color="black")
    plt.plot(t, pred, label="Predicted SOC Drop", color="blue")
    plt.fill_between(t,
                     np.array(pred) - 1.64 * np.array(sigma),
                     np.array(pred) + 1.64 * np.array(sigma),
                     color="blue", alpha=0.2, label="90% Confidence Interval")

    plt.xlabel("Timestep")
    plt.ylabel("SOC Drop to End")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_soc_drop_and_range(result_dict, title="SOC Drop Prediction"):
    set_plot_style()
    save_path = './plots/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    t = result_dict["t"]
    true_soc = result_dict["true_soc_drop"]
    pred_soc = result_dict["predicted_soc_drop"]
    sigma_soc = result_dict["sigma"]

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Plot SOC Drop ---
    axs[0].plot(t, true_soc, label="True SOC Drop", color="black")
    axs[0].plot(t, pred_soc, label="Predicted SOC Drop", color="blue")
    axs[0].fill_between(t,
                        np.array(pred_soc) - 1.64 * np.array(sigma_soc),
                        np.array(pred_soc) + 1.64 * np.array(sigma_soc),
                        color="blue", alpha=0.2, label="90% Confidence Interval")

    axs[0].set_ylabel("SOC Drop(%)")
    axs[0].set_title(title)
    axs[0].legend()
    axs[0].grid(True)

    # --- Plot Range Estimate if available ---
    if "mu_range" in result_dict and "sigma_range" in result_dict:
        pred_range = result_dict["mu_range"]
        sigma_range = result_dict["sigma_range"]
        true_range = result_dict["true_range"]

        axs[1].plot(t, pred_range, label="Predicted Range Remaining", color="green")
        axs[1].plot(t, true_range, label="True Range Remaining", color="black")
        axs[1].fill_between(t,
                            np.array(pred_range) - 1.64 * np.array(sigma_range),
                            np.array(pred_range) + 1.64 * np.array(sigma_range),
                            color="green", alpha=0.2, label="90% Confidence Interval")
        axs[1].set_ylabel("Remaining Range (km)")
        axs[1].set_xlabel("Time (s)")
        axs[1].legend()
#         axs[1].set_xlim((4000,5000))
        axs[1].grid(True)

        # if max of predicted range (with 90 % confidence high) is too high compared to true range, set y limit
        if max((pred_range) + 1.64 * np.array(sigma_range)) > 1.5 * max(true_range):
            axs[1].set_ylim(-50, 1.5 * max(true_range))


    plt.tight_layout()
    plt.savefig(f"{save_path}{title}.png")
    plt.show()


def plot_soc_drop_and_range_with_shaded(result_dict, title="SOC Drop Prediction With Shaded"):
    set_plot_style()
    save_path = './plots/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    t = np.array(result_dict["t"])
    true_soc_drop = np.array(result_dict["true_soc_drop"])
    pred_soc = np.array(result_dict["predicted_soc_drop"])
    sigma_soc = np.array(result_dict["sigma"])
    true_soc = np.array(result_dict["true_soc"])

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # find time when remaining range hits zero (true)
    shade_start = None
    min_soc_index = np.where(true_soc <= min(true_soc))[0][0]

    if "true_range" in result_dict:
        true_range = np.array(result_dict["true_range"])
        zero_idx = np.where(true_range <= 1) # initially this was 0.5, but there were some cases where the minimum range was around 0.8
        
        if len(zero_idx)>1:
            zero_idx = zero_idx[-1]
        if true_range[zero_idx] <= 1:
            shade_start = t[zero_idx]
            
    # --- Plot SOC Drop (subplot 0) ---
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
    
    if shade_start is not None:
        axs[0].axvspan(shade_start, t[-1], color="red", alpha=0.1)
        axs[0].text(shade_start, axs[0].get_ylim()[1]*0.9,
                    "Need charging", color="red", rotation=0, va="top")
    axs[0].set_ylabel("SOC Drop (%)")
#     axs[0].set_title(title)
    axs[0].legend(ncols=2)
    axs[0].grid(True)

    # --- Plot Range Estimate (subplot 1) ---
    if "mu_range" in result_dict and "sigma_range" in result_dict:
#         zero_range_idx = [range_ > 0.5 for range_ in result_dict['mu_range']]
        pred_range = np.array(result_dict["mu_range"])
        sigma_range = np.array(result_dict["sigma_range"])
        # only plot up to shade_start
        if shade_start is not None:
            mask = t <= shade_start
        else:
            mask = slice(None)
        axs[1].plot(t[mask], pred_range[mask], label="Predicted Range", color="green")
        axs[1].plot(t[mask], true_range[mask], label="True Range", color="black")
        axs[1].fill_between(t[mask],
                            pred_range[mask] - 1.64 * sigma_range[mask],
                            pred_range[mask] + 1.64 * sigma_range[mask],
                            color="green", alpha=0.2, label="90% CI")
        # shade out the rest
        if shade_start is not None:
            axs[1].axvspan(shade_start, t[-1], color="red", alpha=0.1)
            axs[1].text(shade_start, axs[1].get_ylim()[1]*0.9,
                        "Cannot complete\nwithout charge", color="red", rotation=0, va="top")
        axs[1].set_ylabel("Remaining Range (km)")
        axs[1].set_xlabel("Time (s)")
        axs[1].legend()
        axs[1].grid(True)
        # if max of predicted range (with 90 % confidence high) is too high compared to true range, set y limit
        if max((pred_range) + 1.64 * np.array(sigma_range)) > 2.5 * max(true_range):
            axs[1].set_ylim(-50, 2.5 * max(true_range))

    plt.tight_layout()
    plt.savefig(f"{save_path}{title}.png")
    plt.show()

