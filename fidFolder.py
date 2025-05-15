import torch
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader

# Paths to your image folders
folder1 = 'pic1'
folder2 = 'pic2'

def calculate_fid_score(folder1, folder2):
    # Basic transform: resize and normalize
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    # Load datasets (you still need dummy subfolders inside folders!)
    dataset1 = datasets.ImageFolder(root=folder1, transform=transform)
    dataset2 = datasets.ImageFolder(root=folder2, transform=transform)

    loader1 = DataLoader(dataset1, batch_size=32, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=32, shuffle=False)

    # Initialize FID metric on CPU
    fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=True)

    # Update with real images
    for images, _ in loader1:
        fid_metric.update(images, real=True)  # No .cuda()

    # Update with generated images
    for images, _ in loader2:
        fid_metric.update(images, real=False)  # No .cuda()

    # Compute FID score
    fid_value = fid_metric.compute().item()
    return fid_value
