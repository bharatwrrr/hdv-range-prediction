import os
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from torch import nn, optim

from src.models import build_model, gaussian_nll
from src.datasets import build_datasets
from src.utils import load_config, collate_fn_skip_none, BASE_CONFIG_PATH


def validate(model, val_loader, criterion, device, max_batches=20):
    """Validate the model on the validation set."""

    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break

            past_seq = batch["past_seq"].to(device)
            future_seq = batch["future_seq"].to(device)
            static = batch["static_feat"].to(device)
            target = batch["target"].to(device)

            mu, sigma = model(past_seq, future_seq, static)
            loss = criterion(mu, sigma, target)
            loss = loss.mean()

            total_loss += loss.item()

            # Accumulate MAE
            batch_mae = torch.abs(mu - target).sum().item()
            total_mae += batch_mae
            total_samples += target.size(0)

    model.train()
    if total_samples == 0:
        return float('nan'), float('nan')
    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    return avg_loss, avg_mae



def train(config_path="configs/base_config.json", verbose: bool = False):
    # Load config
    config = load_config(BASE_CONFIG_PATH, config_path)
    model_name = config["name"]
    # Save the model with the unique name
    model_path = f"models/{model_name}.pth"
    if os.path.exists(model_path):
        print(f"Model {model_name} already exists. Skipping training.")
        return
    # set_seed(42)

    # Init Weights & Biases
    wandb.init(project="soc_drop_prediction", config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, test_dataset = build_datasets(config, verbose=verbose)
    if verbose:
        print(f"Train dataset size: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=12,
                              collate_fn=collate_fn_skip_none)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=12,
                            collate_fn=collate_fn_skip_none) #TODO: shuffle=True for validation? 
    
    # Build model
    model = build_model(config).to(device) #TODO: Implement Distributed Data Parallelism
    if verbose:
        print(f"Model: {model}")
    optimizer = optim.Adam(model.parameters(), lr=config["init_lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=config["patience"])
    criterion = lambda mu, sigma, target: gaussian_nll(mu, sigma, target) + \
                                          config["loss_alpha"] * F.mse_loss(mu, target)

    # Make sure the models directory exists
    os.makedirs("models", exist_ok=True)
    
    best_val_loss = float('inf')
    losses = []
    mae = []
    torch.cuda.empty_cache()

    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        total_mae = 0
        total_samples = 0
        for i, batch in enumerate(train_loader):
            past_seq = batch["past_seq"].to(device)
            future_seq = batch["future_seq"].to(device)
            static_feat = batch["static_feat"].to(device)
            target = batch["target"].to(device)

            mu, sigma = model(past_seq, future_seq, static_feat)
            loss = criterion(mu, sigma, target)
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()
    
            # Accumulate MAE
            batch_mae = torch.abs(mu - target).sum().item()
            total_mae += batch_mae
            total_samples += target.size(0)

            # Get loss and mae for validation
            if i % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_loss, val_mae = validate(model, val_loader, criterion, device)
                model.train()
                scheduler.step(val_loss)

            
            losses.append(loss)
            mae.append(batch_mae/total_samples)
            wandb.log({
                "epoch": epoch,
                "Step": i,
                "train_loss": loss.item(),
                "train_batch_mae": batch_mae/target.size(0),
                "val_loss": val_loss if i>10 else float('nan'),
                "val_mae": val_mae if i>10 else float('nan'),
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            if verbose:
                print(f"[Epoch {epoch+1}, Step {i}] Train Loss: {loss.item():.4f}, Train MAE: {batch_mae/target.size(0):.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        print(f"[Epoch {epoch+1}] Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
        wandb.log({"train_loss": loss.item(), "val_loss": val_loss, "val_mae": val_mae, "epoch": epoch + 1})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
            torch.save(model, model_path)
            wandb.run.summary["best_val_loss"] = best_val_loss

if __name__ == "__main__":
    import argparse    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/base_config.json")
    args = parser.parse_args()
    train(args.config)
