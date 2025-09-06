"""
Enhanced Configuration for PyTorch ATC System
"""
import torch
import subprocess

def check_gpu_setup():
    """GPU setup check"""
    print("🔍 GPU Setup Check")
    print("=" * 40)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return True
    return False

# Check GPU
gpu_available = check_gpu_setup()
DEVICE = torch.device('cuda' if gpu_available else 'cpu')
print(f"✅ Using device: {DEVICE}")

# Dataset configuration
DATASET_CONFIG = {
    'possible_paths': ['Indian_bovine_breeds', 'indian-bovine-breeds', 'dataset', 'data'],
    'image_size': (224, 224),
    'max_images_per_breed': 100,
    'supported_formats': ('.jpg', '.jpeg', '.png', '.bmp')
}

# Model configuration optimized for RTX 3050
MODEL_CONFIG = {
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'test_size': 0.2,
    'num_workers': 4
}

# ATC scoring criteria
ATC_SCORING_CRITERIA = {
    'body_length': {'min': 120, 'max': 180, 'weight': 0.15, 'unit': 'cm'},
    'height_at_withers': {'min': 110, 'max': 150, 'weight': 0.20, 'unit': 'cm'},
    'chest_width': {'min': 35, 'max': 55, 'weight': 0.15, 'unit': 'cm'},
    'rump_angle': {'min': 15, 'max': 35, 'weight': 0.10, 'unit': 'degrees'},
    'udder_attachment': {'min': 1, 'max': 9, 'weight': 0.20, 'unit': 'score'},
    'leg_structure': {'min': 1, 'max': 9, 'weight': 0.10, 'unit': 'score'},
    'overall_conformation': {'min': 1, 'max': 9, 'weight': 0.10, 'unit': 'score'}
}
