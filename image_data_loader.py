import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional, List
import numpy as np
from pixel_bit_encoding import PixelEncoder

class ImageDataset(Dataset):
    """Dataset for loading and encoding images for sequence learning."""
    
    def __init__(
        self,
        image_dir: str,
        image_size: Tuple[int, int] = (32, 32),
        grayscale: bool = False,
        transform: Optional[transforms.Compose] = None,
        linearize: bool = False
    ):
        """
        Initialize the image dataset.
        
        Args:
            image_dir: Directory containing the images
            image_size: Target size for all images (width, height)
            grayscale: If True, convert images to grayscale
            transform: Optional additional transforms to apply
            linearize: If True, flatten spatial dimensions into one dimension
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.grayscale = grayscale
        self.transform = transform
        self.linearize = linearize
        
        # List all image files
        self.image_files = []
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        for file in os.listdir(image_dir):
            if os.path.splitext(file)[1].lower() in valid_extensions:
                self.image_files.append(os.path.join(image_dir, file))
        
        if not self.image_files:
            raise ValueError(f"No valid images found in {image_dir}")
        
        # Create base transform
        self.base_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        
        if grayscale:
            self.base_transform = transforms.Compose([
                transforms.Grayscale(1),
                self.base_transform
            ])
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB' if not self.grayscale else 'L')
        
        # Apply base transform
        image = self.base_transform(image)
        
        # Apply additional transforms if provided
        if self.transform is not None:
            image = self.transform(image)
        
        # Encode image
        if self.grayscale:
            # Ensure shape is [..., 1] for grayscale
            if image.dim() == 2:
                image = image.unsqueeze(-1)
            encoded = PixelEncoder.encode_grayscale(image)
        else:
            # Ensure shape is [..., 3] for RGB
            if image.dim() == 3 and image.shape[0] == 3:
                image = image.permute(1, 2, 0)  # Change from CxHxW to HxWxC
            encoded = PixelEncoder.encode_rgb(image)
        
        # Linearize spatial dimensions if requested
        if self.linearize:
            encoded = encoded.reshape(-1, encoded.shape[-1])
        
        return encoded

class ImageDataLoader:
    """Wrapper class for creating data loaders for image datasets."""
    
    def __init__(
        self,
        train_dir: str,
        val_dir: Optional[str] = None,
        image_size: Tuple[int, int] = (32, 32),
        batch_size: int = 32,
        grayscale: bool = False,
        num_workers: int = 4,
        val_split: float = 0.2,
        transform: Optional[transforms.Compose] = None,
        linearize: bool = False
    ):
        """
        Initialize the image data loader.
        
        Args:
            train_dir: Directory containing training images
            val_dir: Optional directory containing validation images
            image_size: Target size for all images (width, height)
            batch_size: Batch size for data loaders
            grayscale: If True, convert images to grayscale
            num_workers: Number of workers for data loading
            val_split: Fraction of training data to use for validation if val_dir is None
            transform: Optional additional transforms to apply
            linearize: If True, flatten spatial dimensions into one dimension
        """
        self.batch_size = batch_size
        self.grayscale = grayscale
        self.linearize = linearize
        self.image_size = image_size
        
        # Create training dataset
        self.train_dataset = ImageDataset(
            train_dir,
            image_size=image_size,
            grayscale=grayscale,
            transform=transform,
            linearize=linearize
        )
        
        # Create validation dataset
        if val_dir:
            self.val_dataset = ImageDataset(
                val_dir,
                image_size=image_size,
                grayscale=grayscale,
                transform=transform,
                linearize=linearize
            )
        else:
            # Split training data for validation
            train_size = int((1 - val_split) * len(self.train_dataset))
            val_size = len(self.train_dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset,
                [train_size, val_size]
            )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    @property
    def train_size(self) -> int:
        return len(self.train_dataset)
    
    @property
    def val_size(self) -> int:
        return len(self.val_dataset)
    
    @property
    def sequence_length(self) -> int:
        """Return the length of encoded sequences (24 for RGB, 8 for grayscale)."""
        return 8 if self.grayscale else 24
    
    @property
    def spatial_dim(self) -> int:
        """Return the size of the spatial dimension when linearized."""
        return self.image_size[0] * self.image_size[1]

# Example usage
if __name__ == "__main__":
    # Create data loader with spatial dimensions (H,W)
    spatial_loader = ImageDataLoader(
        train_dir="images256x256",
        image_size=(32, 32),
        batch_size=32,
        grayscale=False,  # Use RGB
        linearize=False  # Keep spatial dimensions
    )
    
    # Create data loader with linearized dimensions
    linear_loader = ImageDataLoader(
        train_dir="images256x256",
        image_size=(32, 32),
        batch_size=32,
        grayscale=False,  # Use RGB
        linearize=True  # Flatten spatial dimensions
    )
    
    # Print dataset information
    print(f"Training set size: {spatial_loader.train_size}")
    print(f"Sequence length: {spatial_loader.sequence_length}")
    
    # Example training loop with spatial dimensions
    for batch in spatial_loader.train_loader:
        # batch shape: [batch_size, height, width, sequence_length]
        print(f"Spatial batch shape: {batch.shape}")
        break
    
    # Example training loop with linearized dimensions
    for batch in linear_loader.train_loader:
        # batch shape: [batch_size, height*width, sequence_length]
        print(f"Linear batch shape: {batch.shape}")
        break 