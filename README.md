# HDV Range Prediction

This repository is the official implementation of the work from the research article: [Reducing Range Prediction Uncertainty in Heavy-Duty Electric Vehicles Using a Route-Informed Transformer-Based Model](https://www.sae.org/publications/technical-papers/content/2025-24-0122/). The article is published in the Proceedings of the 17th International Conference on Engines and Vehicles for Sustainable Transport 2025.


## Features

- ETL pipelines for HDV dataset processing
- Evaluation metrics and visualization tools
- Scripts for training, testing, and deploying models
- Multiple model architectures (LSTM and Transformer)
- Comprehensive ablation studies for feature analysis

## Prerequisites

- Python 3.7+
- Required Python packages (see Installation section)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bharatwrrr/hdv-range-prediction.git
   cd hdv-range-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Directory Structure

- `data/` - Raw and processed datasets
- `configs/` - Configuration files for model architecture, feature selection, loss function, and input sequence configurations
  - `ablations/` - Configuration files for ablation studies
    - `arch/` - Architecture variations
    - `feature/` - Feature ablation configurations
    - `loss/` - Loss function configurations
    - `lstm/` - LSTM-specific configurations
    - `sequence/` - Sequence length configurations
- `src/` - Source code for preprocessing, modeling, and evaluation
  - `datasets/` - Dataset handling and preprocessing
  - `models/` - Model implementations and utilities
- `tests/` - Get results on test dataset
- `README.md` - Project documentation

## Usage

### Basic Training

Modify configuration files in `configs/` to set up your experiment parameters.

To train a model with a specific configuration:

1. Set up the config file in `configs/ablations/` if not already present
2. Run the training script:
   ```bash
   python train.py --config configs/ablations/your_config.json --continue-training --verbose
   ```

### Ablation Studies

To reproduce the ablation studies from the paper:

1. **Run all ablations** (architecture, loss, feature, sequence):
   ```bash
   python ablation_sweep.py
   ```

2. **Run specific ablation category** (e.g., architecture only):
   ```bash
   python ablation_sweep.py --which_ablation arch
   ```

3. **Continue training with verbose output**:
   ```bash
   python ablation_sweep.py --continue_training --verbose
   ```

### Available Ablation Categories

- `arch` - Architecture variations (big_fusion, small_fusion)
- `feature` - Feature ablation studies (SOC, distance, road type, etc.)
- `loss` - Loss function variations (different alpha values)
- `sequence` - Sequence length variations (past and future sequences)
- `lstm` - LSTM-specific configurations

## Model Architectures

This project implements and compares two main architectures:

- **LSTM Model**: Recurrent neural network for sequential data processing
- **Transformer Model**: Attention-based architecture for improved feature learning

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

## License

This project is licensed under the MIT License.