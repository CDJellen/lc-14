from typing import Optional, List

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt



def visualize_feature_maps(
  model: "torch.nn.Module",
  image_tensor: "torch.Tensor",
  layers_to_visualize: Optional[List[str]] = None
) -> None:
    """Visualizes feature maps of a PyTorch model.

    Args:
      model: The PyTorch model.
      image_path: Path to the input image.
      transformations: transformations to be applied.
      layers_to_visualize: A list of layer names to visualize. If None,
        visualizes all conv and pooling layers.
    """
    # Get feature maps
    activations = {}

    def get_activation(name):
      def hook(model, input, output):
        # Move output to CPU and detach before storing
        activations[name] = output.detach().cpu()
      return hook
    
    # Register hooks for specified layers or all relevant layers
    if layers_to_visualize is None:
      for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
          layer.register_forward_hook(get_activation(name))
    else:
      for name in layers_to_visualize:
        layer = dict(model.named_modules()).get(name)
        if layer is not None:
          layer.register_forward_hook(get_activation(name))
        else:
          print(f"Warning: Layer '{name}' not found in model.")

    # Run the model (we don't need the output, just the activations)
    with torch.no_grad():
      model.eval()
      _ = model(image_tensor)

    # Plot the feature maps
    for name, act in activations.items():
      num_features = act.shape[1]
      rows = int(np.ceil(np.sqrt(num_features)))
      cols = int(np.ceil(num_features / rows))

      fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
      axes = axes.ravel()  # Flatten axes array for easy indexing

      for i in range(num_features):
        if i < len(axes):  # prevents error when num_features != rows*cols
          axes[i].imshow(act[0, i, :, :], cmap='viridis')
          axes[i].axis('off')
          axes[i].set_title(f'{name} - Feature {i+1}')

      for j in range(num_features, rows*cols):
        if j<len(axes):
          axes[j].remove()

      fig.suptitle(f"Feature Maps for Layer: {name}", fontsize=16)
      plt.tight_layout(rect=[0, 0, 1, 0.96])
      plt.show()
