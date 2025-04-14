import torch
import torch.nn as nn


class DryingLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, dry_ratio=0.1, dry_strength=0.001):
        """
        A Linear layer that progressively "dries" a portion of its neurons
        by applying decay after each backward pass.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            dry_ratio (float): Fraction (0–1) of output neurons to dry.
            dry_strength (float): Drying strength (decay rate applied after backward).
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dry_ratio = dry_ratio
        self.dry_strength = dry_strength

        # Define the binary mask for drying
        dry_count = int(out_features * dry_ratio)
        mask = torch.ones(out_features)
        mask[-dry_count:] = 0.0  # Dry the last portion
        self.register_buffer("dry_mask", mask.view(-1, 1))

        # Register backward hook once
        self.linear.weight.register_hook(self._dry_hook)

    def forward(self, x):
        return self.linear(x)

    def _dry_hook(self, grad):
        """Called automatically after .backward(). Applies drying."""
        with torch.no_grad():
            self.linear.weight -= (
                self.dry_strength * (1 - self.dry_mask) * self.linear.weight
            )
            if self.linear.bias is not None:
                self.linear.bias -= (
                    self.dry_strength * (1 - self.dry_mask.squeeze()) * self.linear.bias
                )
        return grad  # Pass gradients through unchanged
