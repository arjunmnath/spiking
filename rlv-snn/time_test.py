import time
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from main import RLVSLSNNNumPy

print("Initializing...")
t0 = time.time()
net = RLVSLSNNNumPy()
print(f"Init: {time.time()-t0:.2f}s")

img = torch.rand(1, 28, 28)
print("Extracting...")
t0 = time.time()
feat = net.extract_features(img)
print(f"Extract: {time.time()-t0:.2f}s")
