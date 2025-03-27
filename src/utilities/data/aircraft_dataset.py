import os
from glob import glob
from typing import List, Callable, Optional

import torch
import pandas as pd
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from PIL import Image


class AircraftDataset(Dataset):
  """
  Custom Dataset class for the Aircraft Images dataset.
  """

  def __init__(
    self,
    image_dir: str,
    labels_fp: Optional[str] = None,
    transformations: List[Callable] = None,
    mode: str = "train",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 2020,
    extension: str = ".jpg"
  ) -> None:
    """
    Initializes the dataset.

    Args:
      image_dir: The path to the directory containing the images
      labels_fp: The filepath to the CSV file containing labels and geometry info.
      transformations: A list of transformations to apply to the images and geometry.
    """
    if mode not in ["train", "val", "test"]:
      raise ValueError("Invalid mode. Must be one of 'train', 'val', or 'test'.")
    if train_frac + val_frac > 1:
      raise ValueError("train_frac + val_frac must be less than 1.")
    
    if not transformations:
      transformations = A.Compose([
        ToTensorV2(),
      ])

    self.image_filepaths = list(sorted(glob(os.path.join(image_dir, f"*{extension}"))))  # get all files with the .jpg extension
    np.random.seed(seed)
    np.random.shuffle(self.image_filepaths)

    # Split the dataset into train, validation, and test sets
    num_images = len(self.image_filepaths)
    train_end = int(train_frac * num_images)
    val_end = train_end + int(val_frac * num_images)
    
    self.train_image_filepaths = self.image_filepaths[:train_end]
    self.val_image_filepaths = self.image_filepaths[train_end:val_end]
    self.test_image_filepaths = self.image_filepaths[val_end:]

    if labels_fp:
      self.labels_df = pd.read_csv(labels_fp, converters={'geometry':lambda x:list(eval(x))})  # parse the list of tuples from string literal
    else:
      self.labels_df = None

    self.transformations = transformations

    # Create a mapping from image filename to index for efficient lookup
    self.filename_to_index = {os.path.basename(fp): i for i, fp in enumerate(self.train_image_filepaths + self.val_image_filepaths + self.test_image_filepaths)}

    self.mode = mode

  def __len__(self):
    """
    Returns the total number of samples in the dataset.
    """
    if self.mode == "train":
      return len(self.train_image_filepaths)
    elif self.mode == "val":
      return len(self.val_image_filepaths)
    else:
      return len(self.test_image_filepaths)

  def __getitem__(self, idx):
    """
    Loads and returns a sample from the dataset at the given index.
    """
    if self.mode == "train":
      img_path = self.train_image_filepaths[idx]
    elif self.mode == "val":
      img_path = self.val_image_filepaths[idx]
    else:
      img_path = self.test_image_filepaths[idx]
    img_name = os.path.basename(img_path)

    # Load image
    image = Image.open(img_path).convert("RGB")

    # Get count of aircraft for the current image if labels are provided
    if self.labels_df is not None:
      annotations = self.labels_df[(self.labels_df['image_id'] == img_name) & (self.labels_df['class'] == "Airplane")]
      count = len(annotations)
    else:
      count = -1

    # Convert count to a PyTorch tensor
    count = torch.tensor(count, dtype=torch.float32)  # Use float for regression

    # Convert PIL Image to NumPy array
    image = np.array(image)

    # Apply transformations if any
    if self.transformations:
      image = self.transformations(image=image)['image']

    return image, count
