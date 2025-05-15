import torch
import numpy as np
from torchvision.models import inception_v3
from torchvision import transforms
from scipy.linalg import sqrtm
from PIL import Image
from tqdm import tqdm
import os

def get_inception_features(image, model, device):
    """Extract features from InceptionV3 model."""
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

def calculate_fid(mu1, sigma1, mu2, sigma2):
    """Compute Fréchet Inception Distance between two distributions."""
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    
    # Numerical stability fix
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def compute_fid_between_folders(folder1, folder2, inputImages=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.fc = torch.nn.Identity()  # Remove final classification layer
    model.eval()

    def load_folder_images(folder_path):
        features = []
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        for filename in tqdm(image_files, desc=f'Processing {folder_path}'):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert('RGB')
            feat = get_inception_features(image, model, device)
            features.append(feat)
        return np.array(features)

    def load_input_images(folder_path):
        features = []
        #image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        for filename in tqdm(inputImages, desc=f'Processing input images'):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert('RGB')
            feat = get_inception_features(image, model, device)
            features.append(feat)
        return np.array(features)

    feats1 = load_folder_images(folder1) if inputImages is None else load_input_images(folder1)
    feats2 = load_folder_images(folder2)

    # Compute statistics
    mu1, sigma1 = feats1.mean(axis=0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = feats2.mean(axis=0), np.cov(feats2, rowvar=False)

    fid_score = calculate_fid(mu1, sigma1, mu2, sigma2)
    return fid_score
