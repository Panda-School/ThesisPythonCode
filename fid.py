import torch
import numpy as np
from torchvision.models import inception_v3
from torchvision import transforms
from scipy.linalg import sqrtm
from PIL import Image

def get_inception_features(image, model, device):
    """Extract features from the InceptionV3 model."""
    image = image.convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model(image)
    
    return features.cpu().numpy().reshape(-1)

def calculate_fid(act1, act2):
    """Compute the Fréchet Inception Distance (FID) between two activations."""
    mu1, mu2 = act1.mean(axis=0), act2.mean(axis=0)
    sigma1, sigma2 = np.cov(act1, rowvar=False), np.cov(act2, rowvar=False)

    # Compute sqrt of product of covariances
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def compute_fid_between_images(image1: Image, image2: Image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.fc = torch.nn.Identity()
    model.eval()

    act1 = get_inception_features(image1, model, device)
    act2 = get_inception_features(image2, model, device)

    # FID approximation: squared L2 distance between features
    fid_value = np.sum((act1 - act2) ** 2)
    return fid_value

