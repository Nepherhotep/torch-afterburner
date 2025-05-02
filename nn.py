import torch
import torch.nn as nn
import copy
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class DryingLinearLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dry_ratio=0.1,
        dry_strength=0.001,
        child_layer_names=None,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dry_ratio = dry_ratio
        self.dry_strength = dry_strength
        self.child_layer_names = child_layer_names or []

        dry_count = int(out_features * dry_ratio)
        mask = torch.ones(out_features)
        mask[-dry_count:] = 0.0
        self.register_buffer("dry_mask", mask.view(-1, 1))

        if dry_strength is not None:
            self.linear.weight.register_hook(self._dry_hook)

    def forward(self, x):
        return self.linear(x)

    def _dry_hook(self, grad):
        with torch.no_grad():
            self.linear.weight -= (
                self.dry_strength * (1 - self.dry_mask) * self.linear.weight
            )
            if self.linear.bias is not None:
                self.linear.bias -= (
                    self.dry_strength * (1 - self.dry_mask.squeeze()) * self.linear.bias
                )
        return grad

    def get_child_layers(self):
        return self.child_layer_names

    def get_dry_metrics(self, prune_threshold=1e-2):
        """
        Calculates L2 norm and sparsity of dry and wet neuron weights and biases.

        Args:
            prune_threshold (float): Values below this are considered 'zero' for sparsity calculation.

        Returns:
            dict:
                - dry_L2: L2 norm of dry weights + biases
                - dry_sparsity: % of dry values below prune_threshold
                - wet_sparsity: % of wet values below prune_threshold
                - dry_bias_sparsity: sparsity of dry biases (optional)
                - wet_bias_sparsity: sparsity of wet biases (optional)
        """
        with torch.no_grad():
            W = self.linear.weight  # shape: (out_features, in_features)
            B = self.linear.bias  # shape: (out_features,)

            # Boolean index masks
            dry_indices = (1 - self.dry_mask).squeeze().bool()  # dry neuron rows
            wet_indices = self.dry_mask.squeeze().bool()  # wet neuron rows

            # Slice weights and biases
            W_dry = W[dry_indices]
            W_wet = W[wet_indices]

            B_dry = (
                B[dry_indices] if B is not None else torch.tensor([], device=W.device)
            )
            B_wet = (
                B[wet_indices] if B is not None else torch.tensor([], device=W.device)
            )

            # Flatten
            dry_all = torch.cat([W_dry.flatten(), B_dry])
            wet_all = torch.cat([W_wet.flatten(), B_wet])

            # Metrics
            return {
                "dry_L2": torch.norm(dry_all).item(),
                "dry_sparsity": (dry_all.abs() < prune_threshold).float().mean().item(),
                "wet_sparsity": (wet_all.abs() < prune_threshold).float().mean().item(),
                "dry_bias_sparsity": (
                    (B_dry.abs() < prune_threshold).float().mean().item()
                    if B.numel() > 0
                    else None
                ),
                "wet_bias_sparsity": (
                    (B_wet.abs() < prune_threshold).float().mean().item()
                    if B.numel() > 0
                    else None
                ),
            }

    def export_compressed(self) -> nn.Linear:
        """
        Exports a compressed nn.Linear layer by slicing off the dry (last) neurons
        based on dry ratio
        """
        out_features = self.linear.out_features
        dry_count = int(out_features * self.dry_ratio)
        keep_count = out_features - dry_count

        W = self.linear.weight[:keep_count]
        B = self.linear.bias[:keep_count] if self.linear.bias is not None else None

        new_layer = nn.Linear(self.linear.in_features, keep_count)
        with torch.no_grad():
            new_layer.weight.copy_(W)
            if B is not None:
                new_layer.bias.copy_(B)

        return new_layer


def getattr_recursive(obj, attr_path):
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def setattr_recursive(obj, attr_path, value):
    *parents, last = attr_path.split(".")
    for attr in parents:
        obj = getattr(obj, attr)
    setattr(obj, last, value)


class DryModelExporter:
    def __init__(self, model: nn.Module):
        self.original_model = model
        self.model = copy.deepcopy(model)

    def export(self) -> nn.Module:
        self._process_module(self.model)
        return self.model

    def _process_module(self, module: nn.Module):
        for name, child in module.named_children():
            if self._is_drying_layer(child):
                compressed = child.export_compressed()
                setattr(module, name, compressed)
                self._patch_child_layers(compressed, child)
            else:
                self._process_module(child)

    def _is_drying_layer(self, module: nn.Module) -> bool:
        return hasattr(module, "export_compressed") and callable(
            module.export_compressed
        )

    def _patch_child_layers(self, compressed_layer: nn.Linear, original_layer):
        if not hasattr(original_layer, "get_child_layers"):
            return

        for child_name in original_layer.get_child_layers():
            try:
                child_layer = getattr_recursive(self.model, child_name)
                if isinstance(child_layer, nn.Linear):
                    self._replace_linear_input(
                        child_name, child_layer, compressed_layer.out_features
                    )
                else:
                    raise TypeError(
                        f"Unsupported child layer type for '{child_name}': {type(child_layer)}"
                    )
            except AttributeError:
                raise AttributeError(f"Child layer '{child_name}' not found in model.")

    def _replace_linear_input(
        self, layer_path: str, layer: nn.Linear, new_in_features: int
    ):
        new_layer = nn.Linear(new_in_features, layer.out_features)
        with torch.no_grad():
            new_layer.weight.copy_(layer.weight[:, :new_in_features])
            new_layer.bias.copy_(layer.bias)
        setattr_recursive(self.model, layer_path, new_layer)
