import torch
import time

print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

if torch.cuda.is_available():
    # Large tensor operations on GPU
    print("Starting GPU computation...")
    
    for i in range(10):
        a = torch.randn(2000, 2000, device='cuda')
        b = torch.randn(2000, 2000, device='cuda') 
        c = torch.mm(a, b)  # Matrix multiplication on GPU
        print(f"Iteration {i+1}: GPU memory used = {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        time.sleep(2)  # Pause so you can see GPU usage in nvidia-smi
    
    print("✅ All computations done on GPU!")
else:
    print("❌ CUDA not available")
