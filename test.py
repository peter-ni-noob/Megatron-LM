import torch 
import numpy as np

x=np.load("/data/dataset_test/0/token_float32.npy")
y=np.load("/data/dataset_test/0/loss_mask.npy")

print(x.shape)
print(y.shape)