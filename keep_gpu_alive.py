import torch
import time

print("GPU Keep-Alive Started")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Small tensor on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(100, 100).to(device)

counter = 0
while True:
    # Tiny computation every 60 seconds
    y = torch.matmul(x, x)
    counter += 1
    if counter % 10 == 0:
        print(f"Keep-alive tick: {counter} (GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB)")
    time.sleep(60)
