import albumentations as A
from albumentations.pytorch import ToTensorV2


alpha_transformations = A.Compose([
  A.Resize(256, 256),  # This transform resizes each image before subsequent processing
  A.HorizontalFlip(p=0.5),  # This transform flips each image horizontally with a probability of 0.5
  A.RandomBrightnessContrast(p=0.2),  # This transform randomly adjusts the brightness and contrast of each image with a probability of 0.2
  A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # This transformation normalizes from standard RGB to grayscale
  ToTensorV2(),  # This transform maps our image to a torch.Tensor object
])
