"""Count storage and work directly from the network; KiB/MiB memory budgets."""
from torch import nn

SRAM_BYTES = 300 * 1024
PSRAM_BYTES = 8 * 1024 * 1024


def describe_model(net):
    layers = []
    for layer in net.net:
        if isinstance(layer, nn.Linear):
            params = sum(p.numel() for p in layer.parameters())
            layers.append({
                "name": f"linear{len(layers) + 1}",
                "in": layer.in_features, "out": layer.out_features,
                "params": params, "bytes_f32": params * 4,
                "macs": layer.in_features * layer.out_features,
            })
    params = sum(layer["params"] for layer in layers)
    # Adjacent input and output buffers coexist during a dense layer.
    activations = max(layer["in"] + layer["out"] for layer in layers)
    return {
        "layers": layers, "total_params": params,
        "total_bytes_f32": params * 4, "total_bytes_int8": params,
        "total_macs": sum(layer["macs"] for layer in layers),
        "fits_300kb_sram_f32": params * 4 <= SRAM_BYTES,
        "fits_300kb_sram_int8": params <= SRAM_BYTES,
        "activation_bytes_f32": activations * 4,
        "activation_bytes_int8": activations,
        "fits_8mb_psram_f32": params * 4 <= PSRAM_BYTES,
        "fits_8mb_psram_int8": params <= PSRAM_BYTES,
    }
