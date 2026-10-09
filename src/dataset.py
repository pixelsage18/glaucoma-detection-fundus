import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# --------------------------------------------------
# Settings
# --------------------------------------------------

RANDOM_SEED = 42
BATCH_SIZE = 16


# --------------------------------------------------
# Image transforms
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # Simple data augmentation
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ToTensor(),

    # ImageNet normalization
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Dataset class
# --------------------------------------------------

class FundusDataset(Dataset):

    def __init__(self, metadata_path, split, transform=None):

        # Read metadata
        df = pd.read_csv(metadata_path)

        # Keep only rows belonging to this split
        self.df = df[
            df["split"] == split
        ].reset_index(drop=True)

        self.transform = transform


    def __len__(self):

        return len(self.df)


    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = row["image_path"]
        label = int(row["label"])

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, label


# --------------------------------------------------
# Create datasets
# --------------------------------------------------

def create_datasets(metadata_path):

    train_dataset = FundusDataset(
        metadata_path=metadata_path,
        split="train",
        transform=train_transform
    )

    val_dataset = FundusDataset(
        metadata_path=metadata_path,
        split="val",
        transform=eval_transform
    )

    test_dataset = FundusDataset(
        metadata_path=metadata_path,
        split="test",
        transform=eval_transform
    )

    return train_dataset, val_dataset, test_dataset


# --------------------------------------------------
# Create DataLoaders
# --------------------------------------------------

def create_dataloaders(metadata_path, batch_size=BATCH_SIZE):

    train_dataset, val_dataset, test_dataset = create_datasets(
        metadata_path
    )

    # Generator makes training shuffle reproducible
    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader


# --------------------------------------------------
# Quick test
# --------------------------------------------------

if __name__ == "__main__":

    metadata_path = "data/processed/metadata_with_splits.csv"

    train_loader, val_loader, test_loader = create_dataloaders(
        metadata_path
    )

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    # Take one batch from the training loader
    images, labels = next(iter(train_loader))

    print("\nBatch image shape:")
    print(images.shape)

    print("\nBatch label shape:")
    print(labels.shape)

    print("\nLabels in first batch:")
    print(labels)
