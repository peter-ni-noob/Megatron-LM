import torch
import random
# Set the seed
seed = 42
torch.manual_seed(seed)

# Generate some random numbers
a = torch.randn(3, 3)
b = torch.randn(3, 3)

# Verify that the same sequence of random numbers is generated
c = torch.randn(3, 3)
d = torch.randn(3, 3)

# torch.manual_seed(seed)
e = torch.randn(3, 3)
f = torch.randn(3, 3)

assert (a == e).all()
assert (b == f).all()