import os
from typing import Tuple

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


def visualize_transformations(
  image_filepath: "os.PathLike",
  transform: "A.BaseCompose",
  figsize: Tuple[int, int] = (12, 8),
) -> None:
  """Applies Albumentations transforms one by one and visualizes the result.

  Args:
    image: The input image (NumPy array).
    transform: The Albumentations Compose object.
  """
  # Load the image
  image = cv2.imread(image_filepath)

  plt.figure(figsize=figsize)
  plt.subplot(1, len(transform.transforms) + 1, 1)
  plt.imshow(image)
  plt.title("Original Image")
  plt.axis("off")

  current_image = image
  for i, t in enumerate(transform.transforms):
    if isinstance(t, ToTensorV2):
      continue
    # Apply the individual transform
    transformed = t(image=current_image)
    transformed_image = transformed["image"]

    plt.subplot(1, len(transform.transforms) + 1, i + 2)
    plt.imshow(transformed_image)
    plt.title(type(t).__name__)  # Display the transform's name
    plt.axis("off")

  plt.tight_layout()
  plt.show()