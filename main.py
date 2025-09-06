# atc_main_fixed.py
from __future__ import annotations

import os
import cv2
import json
import time
import math
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from typing import Tuple, Dict, Any, List

import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# -----------------------
# Configuration
# -----------------------
CONFIG: Dict[str, Any] = {
    "device": "auto",  # "auto" | "cuda" | "mps" | "cpu"
    "seed": 42,
    "dataset": {
        "possible_paths": [
            "/Users/aditya/Documents/atc/Indian Bovine Breeds",
            "./Indian Bovine Breeds",
            "./data/Indian_Bovine_Breeds"
        ],
        "supported_formats": (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        "image_size": (224, 224),
        "max_images_per_breed": 100,
        "limit_num_breeds": 0
    },
    "train": {
        "epochs": 25,
        "batch_size": 32,
        "val_size": 0.2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "warmup_epochs": 2,
        "head_only_epochs": 3,
        "early_stop_patience": 6,
        "num_workers": 0,
        "pin_memory": True
    },
    "eval": {"num_test_samples": 10, "topk": 3},
    "paths": {
        "artifacts_dir": "outputs",
        "best_model": "outputs/best_atc_model.pth",
        "final_checkpoint": "outputs/complete_atc_model.pth",
        "cm_fig": "outputs/confusion_matrix.png",
        "report_json": "outputs/classification_report.json",
        "test_results": "outputs/test_results.json",
        "gradcam_dir": "outputs/gradcam",
        "quick_weights": "cattle_resnet50.pth"
    },
    "atc_scoring": {
        "body_length": {"min": 120, "max": 180, "unit": "cm", "weight": 0.2},
        "height_at_withers": {"min": 110, "max": 150, "unit": "cm", "weight": 0.2},
        "chest_width": {"min": 35, "max": 55, "unit": "cm", "weight": 0.15},
        "rump_angle": {"min": 15, "max": 35, "unit": "deg", "weight": 0.1},
        "udder_attachment": {"min": 1, "max": 9, "unit": "score", "weight": 0.175},
        "leg_structure": {"min": 1, "max": 9, "unit": "score", "weight": 0.125},
        "overall_conformation": {"min": 1, "max": 9, "unit": "score", "weight": 0.05},
    },
}

# -----------------------
# Utilities
# -----------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def pick_device(pref: str = "auto") -> torch.device:
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def ensure_dirs() -> None:
    os.makedirs(CONFIG["paths"]["artifacts_dir"], exist_ok=True)
    os.makedirs(CONFIG["paths"]["gradcam_dir"], exist_ok=True)

# -----------------------
# Dataset
# -----------------------
class CattleDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images        # numpy array (N, H, W, C), RGB
        self.labels = labels        # numpy array (N,)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        label = int(self.labels[idx])
        # convert numpy (H,W,C) -> PIL for torchvision v1 transforms
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label

# -----------------------
# Load dataset
# -----------------------
def load_dataset() -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    dataset_path = "/Users/aditya/Documents/atc/Indian Bovine Breeds"
    for p in CONFIG["dataset"]["possible_paths"]:
        if os.path.exists(p):
            dataset_path = p
            break
    if dataset_path is None:
        raise FileNotFoundError("Dataset not found. Update CONFIG['dataset']['possible_paths'].")

    print(f"✅ Found dataset at: {dataset_path}")
    images: List[np.ndarray] = []
    labels: List[str] = []
    breed_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    if CONFIG["dataset"]["limit_num_breeds"] > 0:
        breed_folders = breed_folders[: CONFIG["dataset"]["limit_num_breeds"]]

    print("📸 Loading images...")
    for breed in sorted(breed_folders):
        breed_path = os.path.join(dataset_path, breed)
        image_files = [f for f in os.listdir(breed_path) if f.lower().endswith(CONFIG["dataset"]["supported_formats"])]
        image_files = image_files[: CONFIG["dataset"]["max_images_per_breed"]]
        count = 0
        for fname in image_files:
            fpath = os.path.join(breed_path, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, CONFIG["dataset"]["image_size"], interpolation=cv2.INTER_AREA)
            images.append(img)
            labels.append(breed)
            count += 1
        print(f"Processing {breed}... Loaded {count} images")

    le = LabelEncoder()
    y = le.fit_transform(labels)
    X = np.array(images)
    print(f"✅ Dataset loaded: {len(X)} images, {len(le.classes_)} breeds")
    return X, y, le

# -----------------------
# Transforms (use weights.transforms())
# -----------------------
def get_transforms():
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()  # includes resize/crop/to_tensor/normalize

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2,0.2,0.2,0.05),
        preprocess,  # final ToTensor + Normalize
    ])

    val_tf = preprocess
    return train_tf, val_tf

# -----------------------
# Model
# -----------------------
class CattleClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_feats, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def freeze_backbone(model: CattleClassifier, freeze: bool = True) -> None:
    for name, p in model.backbone.named_parameters():
        if name.startswith('fc.'):
            p.requires_grad = True
        else:
            p.requires_grad = not (name.startswith('layer4') or name.startswith('fc')) if freeze else True

# -----------------------
# ATC scorer (deterministic)
# -----------------------
class ATCScorer:
    def __init__(self, criteria: Dict[str, Dict[str, Any]]):
        self.criteria = criteria
        print("🐄 ATC Scorer initialized")

    def _norm(self, val: float, k: str) -> float:
        c = self.criteria[k]
        v = (val - c["min"]) / (c["max"] - c["min"]) * 100.0
        return float(max(0.0, min(100.0, v)))

    def extract_body_parameters(self, image: np.ndarray) -> Dict[str, Any]:
        img = image.copy()
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((5,5), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return self._defaults()
        c = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        H,W = gray.shape[:2]
        body_length = 120 + (w / max(1,W)) * 60
        height_at_withers = 110 + (h / max(1,H)) * 40
        chest_width = 35 + (min(w,h) / max(1, max(W,H))) * 20
        ROI = gray[y:y+h, x:x+w] if h>0 and w>0 else gray
        ys, xs = np.nonzero(ROI > np.mean(ROI)) if ROI.size>0 else ([],[])
        if len(xs) < 10:
            rump_angle = 25.0
        else:
            xs_c = xs - xs.mean(); ys_c = ys - ys.mean()
            cov = np.cov(np.vstack([xs_c, ys_c]))
            eigvals, eigvecs = np.linalg.eig(cov)
            major = eigvecs[:, np.argmax(eigvals)]
            angle = math.degrees(math.atan2(major[1], major[0]))
            rump_angle = float(max(15.0, min(35.0, abs(angle))))
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)
        udder_attachment = int(1 + round(min(8, 8 * edge_density)))
        leg_structure = int(1 + round(min(8, 8 * (1 - edge_density))))
        overall_conformation = float(6.0)
        return {
            "body_length": round(body_length,1),
            "height_at_withers": round(height_at_withers,1),
            "chest_width": round(chest_width,1),
            "rump_angle": round(rump_angle,1),
            "udder_attachment": udder_attachment,
            "leg_structure": leg_structure,
            "overall_conformation": overall_conformation
        }

    def _defaults(self):
        return {
            "body_length": 150.0, "height_at_withers": 125.0, "chest_width": 45.0,
            "rump_angle": 25.0, "udder_attachment": 5, "leg_structure": 5, "overall_conformation": 6.0
        }

    def score(self, params: Dict[str, Any]) -> Dict[str, Any]:
        total = 0.0; details = {}
        for k,v in params.items():
            if k not in self.criteria: continue
            norm = self._norm(float(v), k)
            w = self.criteria[k]["weight"]
            total += norm * w
            details[k] = {"value": float(v), "unit": self.criteria[k]["unit"], "normalized": round(norm,1), "weighted": round(norm*w,2)}
        grade = ('A+ (Excellent)' if total >=90 else 'A (Very Good)' if total>=80 else 'B (Good)' if total>=70 else 'C (Average)' if total>=60 else 'D (Below Average)')
        return {"total_score": round(total,2), "grade": grade, "detailed_scores": details}

# -----------------------
# Training helpers
# -----------------------
class EarlyStopping:
    def __init__(self, patience:int=5, min_delta:float=1e-4):
        self.patience = patience; self.min_delta = min_delta; self.counter = 0; self.best = None; self.should_stop=False
    def step(self, metric: float) -> None:
        if self.best is None or metric < self.best - self.min_delta:
            self.best = metric; self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

def make_schedulers(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch: int):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        return 1.0
    warmup = LambdaLR(optimizer, lr_lambda=lr_lambda)
    plateau = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    return warmup, plateau

@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval(); criterion = nn.CrossEntropyLoss()
    running_loss = 0.0; correct=0; total=0; all_preds=[]; all_targets=[]
    for images, targets in loader:
        images = images.to(device); targets = targets.to(device)
        outputs = model(images); loss = criterion(outputs, targets)
        running_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
        all_preds.append(preds.cpu().numpy()); all_targets.append(targets.cpu().numpy())
    avg_loss = running_loss / max(1, len(loader))
    acc = 100.0 * correct / max(1, total)
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    y_true = np.concatenate(all_targets) if all_targets else np.array([])
    return avg_loss, acc, y_true, y_pred

# -----------------------
# Train loop
# -----------------------
def train(model: CattleClassifier, train_loader, val_loader, device: torch.device, epochs: int, head_only_epochs: int):
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["train"]["label_smoothing"])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["train"]["learning_rate"], weight_decay=CONFIG["train"]["weight_decay"])
    warmup, plateau = make_schedulers(optimizer, CONFIG["train"]["warmup_epochs"], epochs)
    stopper = EarlyStopping(patience=CONFIG["train"]["early_stop_patience"])
    best_val_acc = -1.0
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(epochs):
        if epoch < head_only_epochs:
            freeze_backbone(model, freeze=True)
        else:
            freeze_backbone(model, freeze=False)
        model.train()
        running_loss=0.0; correct=0; total=0
        start = time.time()
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device); targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images); loss = criterion(outputs, targets)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                outputs = model(images); loss = criterion(outputs, targets); loss.backward(); optimizer.step()
            running_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item(); total += targets.size(0)
            if (batch_idx % 10)==0:
                print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f} - Acc: {100.0*correct/max(1,total):.1f}%")
        warmup.step()
        train_loss = running_loss / max(1, len(train_loader)); train_acc = 100.0 * correct / max(1, total)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, num_classes=len(set(train_loader.dataset.labels)))
        plateau.step(val_loss)
        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{epochs} | {elapsed:.1f}s | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["paths"]["best_model"])
            print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
        stopper.step(val_loss)
        if stopper.should_stop:
            print("⏹️ Early stopping triggered."); break
    # also save quick-checkpoint
    torch.save(model.state_dict(), CONFIG["paths"]["quick_weights"])
    return best_val_acc

# -----------------------
# Prediction & testing
# -----------------------
@torch.no_grad()
def predict_topk(model, img, device, label_encoder, k=3):
    model.eval()
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    inp = preprocess(img).unsqueeze(0).to(device)
    logits = model(inp)
    probs = F.softmax(logits, dim=1)
    confs, idxs = torch.topk(probs, k=min(k, probs.shape[1]), dim=1)
    confs = confs[0].cpu().numpy().tolist()
    idxs = idxs[0].cpu().numpy().tolist()
    breeds = label_encoder.inverse_transform(idxs)
    return {"topk_breeds": list(map(str, breeds)), "topk_probs": [float(c) for c in confs]}

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], save_path: str) -> None:
    from matplotlib import ticker
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(max(6, len(labels)*0.5), max(5, len(labels)*0.5)))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=labels, yticklabels=labels, ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True)); ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.close(fig)

def test_system(model: CattleClassifier, le: LabelEncoder, atc: ATCScorer, X_val: np.ndarray, y_val: np.ndarray, device: torch.device) -> List[Dict[str, Any]]:
    print(f"\n🧪 Testing ATC System with {CONFIG['eval']['num_test_samples']} samples...")
    indices = np.random.choice(len(X_val), min(CONFIG['eval']['num_test_samples'], len(X_val)), replace=False)
    results = []
    for i, idx in enumerate(indices, 1):
        print(f"\n🔍 Test {i}:\n" + "-"*30)
        img = X_val[idx]; actual_label = y_val[idx]; actual_breed = le.inverse_transform([actual_label])[0]
        topk = predict_topk(model, img, device, le, k=CONFIG['eval']['topk'])
        pred_breed = topk["topk_breeds"][0]; confidence = topk["topk_probs"][0]
        params = atc.extract_body_parameters(img); atc_score = atc.score(params)
        print(f"Actual: {actual_breed}")
        print(f"Predicted: {pred_breed} ({confidence*100:.1f}%)")
        print(f"ATC Score: {atc_score['total_score']}/100")
        print(f"Grade: {atc_score['grade']}")
        res = {"animal_id": f"TEST_{i:03d}", "timestamp": datetime.now().isoformat(), "actual_breed": actual_breed, "predicted_breed": pred_breed, "confidence": float(confidence), "topk": topk, "body_parameters": params, "atc_score": atc_score}
        results.append(res)
    with open(CONFIG["paths"]["test_results"], 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Testing completed! Results saved to {CONFIG['paths']['test_results']}")
    return results

def split_data(X, y, val_size=CONFIG["train"]["val_size"], test_size=0.1, seed=CONFIG["seed"]):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=seed, stratify=y_train_val)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def create_data_loaders(X, y, train_data, val_data, test_data, train_tf, val_tf):
    train_dataset = CattleDataset(*train_data, transform=train_tf)
    val_dataset = CattleDataset(*val_data, transform=val_tf)
    test_dataset = CattleDataset(*test_data, transform=val_tf)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["train"]["batch_size"], shuffle=True, num_workers=CONFIG["train"]["num_workers"], pin_memory=CONFIG["train"]["pin_memory"])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["train"]["batch_size"], shuffle=False, num_workers=CONFIG["train"]["num_workers"], pin_memory=CONFIG["train"]["pin_memory"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["train"]["batch_size"], shuffle=False, num_workers=CONFIG["train"]["num_workers"], pin_memory=CONFIG["train"]["pin_memory"])
    return train_loader, val_loader, test_loader


# -----------------------
# Main
# -----------------------
def main():
    set_seed(CONFIG["seed"])
    device = pick_device(CONFIG["device"])
    ensure_dirs()
    print(f"Using device: {device}\n")

    # Load dataset
    X, y, le = load_dataset()
    num_classes = len(le.classes_)

    # Split dataset
    train_data, val_data, test_data = split_data(X, y)

    # Transforms & DataLoaders
    train_tf, val_tf = get_transforms()
    train_loader, val_loader, test_loader = create_data_loaders(
        X, y, train_data, val_data, test_data, train_tf, val_tf
    )

    # Model
    model = CattleClassifier(num_classes=num_classes).to(device)

    # Train or Load
    choice = input("Do you want to retrain the model? (y/n): ").strip().lower()
    if choice == "y":
        best_val_acc = train(model, train_loader, val_loader, device, epochs=CONFIG["train"]["epochs"], head_only_epochs=CONFIG["train"]["head_only_epochs"])
        torch.save(model.state_dict(), CONFIG["paths"]["quick_weights"])
    else:
        print("Loading pretrained weights...")
        model.load_state_dict(torch.load(CONFIG["paths"]["quick_weights"], map_location=device))
        model.eval()

    # ATC Scorer + Test
    atc = ATCScorer(CONFIG["atc_scoring"])
    test_system(model, le, atc, test_data[0], test_data[1], device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=CONFIG["device"], help="auto|cuda|mps|cpu")
    args = parser.parse_args()
    CONFIG["device"] = args.device
    main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=CONFIG["device"], help="auto|cuda|mps|cpu")
    args = parser.parse_args()
    CONFIG["device"] = args.device
    main()