
def build_transformer_model(config: dict):
    """
    Build the transformer model based on the configuration.

    Args:
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        nn.Module: The constructed transformer model.
    """
    from .transformer_model import TransformerSOCDropPredictor
    return TransformerSOCDropPredictor(
        past_input_dim=len(config["past_features"]) if config['name'] != 'no_past' else 0,
        future_input_dim=len(config["future_features"]) + (config["road_num_classes"] if config["use_road_type"] else 0) if config['name'] != 'no_future' else 0,
        static_dim=len(config["static_features"]) if config['name'] != 'no_static' else 0,
        past_hidden_dim=config["past_encoder_hidden_dim"],
        future_hidden_dim=config["future_encoder_hidden_dim"],
        fused_hidden_dim=config["fused_hidden_dim"]
    )

def build_lstm_model(config: dict):
    """
    Build the LSTM model based on the configuration.
    Args:
        config (dict): Configuration dictionary containing model parameters.
    Returns:
        nn.Module: The constructed LSTM model.
    """
    from .lstm_model import LSTMModel
    return LSTMModel(
        past_dim=len(config["past_features"]),
        static_dim=len(config["static_features"]),
        n_layers = 3
    )


def build_model(config: dict):
    """
    Build the model based on the configuration.

    Args:
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        nn.Module: The constructed model.
    """
    if config["past_encoder_type"] == "transformer":
        return build_transformer_model(config)
    elif config["past_encoder_type"] == "lstm":
        return build_lstm_model(config)
    else:
        raise ValueError(f"Unknown model type: {config['model']}")
    