"""
PyTorch Training Utilities
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from config import DEVICE, MODEL_CONFIG

class Trainer:
    """Training manager for cattle classification"""
    
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=MODEL_CONFIG['learning_rate'],
            weight_decay=MODEL_CONFIG['weight_decay']
        )
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Progress update
            if batch_idx % 20 == 0:
                print(f'Batch [{batch_idx}/{len(self.train_loader)}] - '
                      f'Loss: {loss.item():.4f} - '
                      f'Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        epoch_loss = total_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs=None):
        """Main training loop"""
        if epochs is None:
            epochs = MODEL_CONFIG['epochs']
        
        print(f"🚀 Starting training for {epochs} epochs...")
        best_val_acc = 0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            epoch_time = time.time() - start_time
            
            print(f'Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%')
            print(f'Time: {epoch_time:.2f}s - LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f'✅ New best model saved! Val Acc: {val_acc:.2f}%')
        
        print(f'\n🎉 Training completed! Best Val Acc: {best_val_acc:.2f}%')
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies
