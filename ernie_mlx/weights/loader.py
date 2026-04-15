"""Load safetensors weights into MLX models with key remapping."""
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def _remap_transformer_key(key: str) -> str:
    """Remap a safetensors weight key to match MLX model parameter names.

    Transformations:
    - 'to_out.0.weight' -> 'to_out.weight' (remove ModuleList index)
    - 'adaLN_modulation.1.*' -> 'adaLN_modulation.layers.1.*' (nn.Sequential)
    """
    # to_out.0.weight -> to_out.weight
    key = key.replace("to_out.0.", "to_out.")
    # adaLN_modulation.1.* -> adaLN_modulation.layers.1.*
    key = key.replace("adaLN_modulation.1.", "adaLN_modulation.layers.1.")
    return key


def _remap_text_encoder_key(key: str) -> str:
    """Remap text encoder weight keys.

    Typical pattern: language_model.model.* -> *
    """
    # Strip 'language_model.model.' prefix
    if key.startswith("language_model.model."):
        key = key[len("language_model.model."):]
    # nn.Sequential index remapping if needed
    key = key.replace("mlp.gate_proj", "mlp.gate_proj")
    key = key.replace("mlp.up_proj", "mlp.up_proj")
    key = key.replace("mlp.down_proj", "mlp.down_proj")
    return key


def _remap_vae_key(key: str) -> str:
    """Remap VAE decoder weight keys.

    Transformations:
    - 'bn.running_mean' -> 'bn_running_mean' (flat attr on VAEDecoder)
    - 'bn.running_var' -> 'bn_running_var'
    - Skip encoder.*, quant_conv.*, bn.num_batches_tracked
    """
    # BN stats
    if key == "bn.running_mean":
        return "bn_running_mean"
    if key == "bn.running_var":
        return "bn_running_var"
    return key


def _transpose_conv_weights(weights: dict, model_params: dict) -> dict:
    """Transpose Conv2d weights from PyTorch NCHW [Cout,Cin,kH,kW] to MLX NHWC [Cout,kH,kW,Cin]."""
    for key, arr in weights.items():
        if key in model_params and arr.ndim == 4:
            # PyTorch: [Cout, Cin, kH, kW] -> MLX: [Cout, kH, kW, Cin]
            weights[key] = arr.transpose(0, 2, 3, 1)
    return weights


def load_transformer_weights(model: nn.Module, model_dir: str) -> None:
    """Load transformer weights from safetensors shards.

    Args:
        model: MLX ErnieImageTransformer model
        model_dir: directory containing safetensors files and index.json
    """
    model_dir = Path(model_dir)
    index_file = model_dir / "diffusion_pytorch_model.safetensors.index.json"

    if index_file.exists():
        # Sharded model
        with open(index_file) as f:
            index = json.load(f)
        shard_files = set(index["weight_map"].values())
    else:
        # Single file
        shard_files = list(model_dir.glob("*.safetensors"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {model_dir}")

    # Get model parameter names for shape reference
    model_params = dict(nn.utils.tree_flatten(model.parameters()))

    # Load and remap all shards
    all_weights = {}
    for shard in sorted(shard_files):
        shard_path = model_dir / shard if isinstance(shard, str) else shard
        weights = mx.load(str(shard_path))
        for old_key, arr in weights.items():
            new_key = _remap_transformer_key(old_key)
            all_weights[new_key] = arr

    # Transpose Conv2d weights
    all_weights = _transpose_conv_weights(all_weights, model_params)

    # Verify all model params have matching weights
    missing = set(model_params.keys()) - set(all_weights.keys())
    extra = set(all_weights.keys()) - set(model_params.keys())
    if missing:
        print(f"WARNING: Missing weights: {missing}")
    if extra:
        print(f"INFO: Extra weights (unused): {extra}")

    # Load into model
    weights_list = list(all_weights.items())
    model.load_weights(weights_list)


def load_text_encoder_weights(model: nn.Module, model_dir: str) -> None:
    """Load Mistral3 text encoder weights."""
    model_dir = Path(model_dir)

    # Get model parameter names
    model_params = dict(nn.utils.tree_flatten(model.parameters()))

    all_weights = {}
    for shard_path in sorted(model_dir.glob("*.safetensors")):
        weights = mx.load(str(shard_path))
        for old_key, arr in weights.items():
            new_key = _remap_text_encoder_key(old_key)
            # Skip vision tower and lm_head
            if old_key.startswith("language_model.lm_head"):
                continue
            if old_key.startswith("multi_modal_projector"):
                continue
            if old_key.startswith("vision_tower"):
                continue
            if new_key in model_params:
                all_weights[new_key] = arr

    missing = set(model_params.keys()) - set(all_weights.keys())
    if missing:
        print(f"WARNING: Missing text encoder weights: {missing}")

    weights_list = list(all_weights.items())
    model.load_weights(weights_list)


def load_vae_weights(model: nn.Module, model_dir: str) -> None:
    """Load VAE decoder weights."""
    model_dir = Path(model_dir)

    model_params = dict(nn.utils.tree_flatten(model.parameters()))

    all_weights = {}
    for shard_path in sorted(model_dir.glob("*.safetensors")):
        weights = mx.load(str(shard_path))
        for key, arr in weights.items():
            new_key = _remap_vae_key(key)
            if new_key in model_params:
                all_weights[new_key] = arr

    # Transpose Conv2d weights
    all_weights = _transpose_conv_weights(all_weights, model_params)

    missing = set(model_params.keys()) - set(all_weights.keys())
    if missing:
        print(f"WARNING: Missing VAE weights: {missing}")

    weights_list = list(all_weights.items())
    model.load_weights(weights_list)
