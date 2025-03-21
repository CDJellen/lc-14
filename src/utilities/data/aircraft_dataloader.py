from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.utilities.data.aircraft_dataset import AircraftDataset


def collate_fn(
  batch: list
  ) -> tuple:
  """The collation function for batching in the DataLoader"""
  images, counts = zip(*batch)
  images = torch.stack(images, dim=0)
  counts = torch.stack(counts, dim=0)
  return images, counts

def get_dataloader(
  image_dir: str,
  labels_fp: Optional[str] = None,
  extension: str = ".jpg",
  transformations: transforms.Compose = None,
  mode: str = "train",
  train_frac: float = 0.8,
  val_frac: float = 0.1,
  seed: int = 2020,  
  batch_size: int = 8,
  shuffle: bool = True,
  num_workers: int = 2
) -> torch.utils.data.DataLoader:
  """
  Creates and returns a PyTorch DataLoader for the Aircraft Count dataset.

  Args:
    image_dir: The path to the directory containing the images.
    labels_fp: The filepath to the CSV file containing labels (counts).
    transformations: A list of callable transformations (e.g., from Albumentations).
    batch_size: The number of samples per batch.
    shuffle: Whether to shuffle the dataset.
    num_workers: The number of subprocesses to use for data loading.

  Returns:
    A PyTorch DataLoader.
  """

  dataset = AircraftDataset(
    image_dir=image_dir,
    labels_fp=labels_fp,
    transformations=transformations,
    mode=mode,
    train_frac=train_frac,
    val_frac=val_frac,
    seed=seed,
    extension=extension,
  )

  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    collate_fn=collate_fn
  )

  return dataloader
