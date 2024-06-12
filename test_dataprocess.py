import numpy as np

data=np.load("/data/wikitext-103-megatron/0/token_float32.npy")
label =np.load("/data/wikitext-103-megatron/0/label_float32.npy")
print(data.shape)
