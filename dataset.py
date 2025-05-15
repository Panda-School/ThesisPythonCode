import json
import os
from tqdm import tqdm
from typing import List, Dict, Any, Set
import torch
import torch.nn as nn
import torchvision.models as models
from scipy import linalg
import numpy as np

def get_url(caption_id: str, local: bool = False) -> str:
    """
    Get the URL for a given caption ID.
    """
    if local:
        return f"http://192.168.2.39/snli-ve/images/{caption_id}"
    else:
        return f"https://hazeveld.org/snli-ve/images/{caption_id}"

def calculate_activation_statistics(images, model, device):
    """Calculate activation statistics (mean & covariance) for the given images using InceptionV3"""
    model.eval()
    act = []

    with torch.no_grad():
        pred = model(images)

    # Get activations
    activations = pred.cpu().numpy()

    # Calculate mean and covariance
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    return mu, sigma


def load_snli_ve_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    Load the SNLI-VE dataset from a JSONL file and filter for neutral relations.
    Removes duplicate captionIDs to ensure unique samples.
    
    Args:
        data_path (str): Path to the SNLI-VE dataset JSONL file
        
    Returns:
        List[Dict[str, Any]]: List of samples with neutral relations, with duplicate captionIDs removed
        
    Raises:
        FileNotFoundError: If the dataset file doesn't exist
        json.JSONDecodeError: If any line in the JSONL file is invalid JSON
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")
    
    if not data_path.endswith('.jsonl'):
        print(f"Warning: Expected a .jsonl file, got {data_path}")
    
    neutral_samples = []
    seen_caption_ids: Set[str] = set()
    line_number = 0
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_number, line in tqdm(enumerate(f, 1)):
                try:
                    sample = json.loads(line.strip())
                    caption_id = sample.get('captionID')
                    
                    # Skip if no captionID or if we've seen this captionID before
                    if not caption_id or caption_id in seen_caption_ids:
                        continue
                        
                    if sample.get('gold_label') == 'neutral':
                        neutral_samples.append(sample)
                        seen_caption_ids.add(caption_id)
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON at line {line_number}: {str(e)}")
                    continue
    except Exception as e:
        raise Exception(f"Error reading file at line {line_number}: {str(e)}")
    
    print(f"Removed {line_number - len(neutral_samples)} duplicate samples")
    return neutral_samples

def main():
    # Example usage
    data_path = "./dataset/snli_1.0_train.jsonl"  # Replace with actual path
    neutral_samples = load_snli_ve_dataset(data_path)
    print(f"Found {len(neutral_samples)} unique neutral samples")

    # Save the neutral samples to a new JSONL file
    output_path = "./dataset/snli_1.0_train_neutral.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in neutral_samples:
            f.write(json.dumps(sample) + '\n')

if __name__ == "__main__":
    main()
