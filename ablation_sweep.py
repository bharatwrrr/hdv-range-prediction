from train import train
import os
from src.utils import load_config, ABLATION_CONFIGS_DIR, OUTPUT_DIR

print(f"Using ablation configs dir: {ABLATION_CONFIGS_DIR}")


def ablation_sweep(ablation_configs_dir = ABLATION_CONFIGS_DIR, output_dir = OUTPUT_DIR, 
                   which_ablation : str = "all", continue_training: bool = False, verbose: bool = False):
    """
    Perform an ablation study by running experiments with different configurations.
    
    Args:
        base_config_path (str): Path to the base configuration file.
        ablation_configs_dir (str): Directory containing ablation configuration files.
        output_dir (str): Directory to save the results of the experiments.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ablation_configs_dir = os.path.join(script_dir, ABLATION_CONFIGS_DIR)
    print(f"Running ablation study with {which_ablation} ablation")
    config_file_paths = []
    if which_ablation == "arch" or which_ablation == "all":
        config_file_paths.extend([os.path.join(ablation_configs_dir, "arch", file) 
                                  for file in os.listdir(os.path.join(ablation_configs_dir, "arch")) if file.endswith('.json')])
    if which_ablation == "loss" or which_ablation == "all":
        config_file_paths.extend([os.path.join(ablation_configs_dir, "loss", file)
                                   for file in os.listdir(os.path.join(ablation_configs_dir, "loss")) if file.endswith('.json')])
    if which_ablation == "feature" or which_ablation == "all":
        config_file_paths.extend([os.path.join(ablation_configs_dir, "feature", file)
                                   for file in os.listdir(os.path.join(ablation_configs_dir, "feature")) if file.endswith('.json')])
    if which_ablation == "sequence" or which_ablation == "all":
        config_file_paths.extend([os.path.join(ablation_configs_dir, "sequence", file)
                                   for file in os.listdir(os.path.join(ablation_configs_dir, "sequence")) if file.endswith('.json')])

    if verbose:
        print(f"Config file paths: {config_file_paths}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    if len(config_file_paths) == 0:
        print("No config files found.")
        return

    # Iterate over all JSON files
    for config_file_path in config_file_paths:
        config = load_config(ablation_path=config_file_path)

        # Run the experiment
        print(f"Running experiment with config: {config}")
        train(config_path=config_file_path, continue_training=continue_training, verbose=verbose)

if __name__ == "__main__":
    ablation_sweep()
