# from megatron.core.tensor_parallel.random import _CUDA_RNG_STATE_TRACKER    
import torch
import torch.nn as nn

# def setSeed():
#     _CUDA_RNG_STATE_TRACKER.reset()
#     seed0=1234
#     seed1=1111
#     _CUDA_RNG_STATE_TRACKER.add("seed0",seed0)
#     _CUDA_RNG_STATE_TRACKER.add("seed1",seed1)


class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.dropout0=nn.Dropout(0.5)
        self.dropout1=nn.Dropout(0.5)


    def forward(self,x):
        print(x)
        x1=self.dropout0(x)
        print(x1)
        x2=self.dropout0(x)
        print(x2)
        x3=self.dropout0(x)
        print(x3)
        x4=self.dropout0(x)
        print(x4)
        return x2
# print(torch.rand([10],device=0))
torch.cuda.manual_seed(1234)
model=Net().cuda(0)
model.train()
data=torch.ones([1,10],dtype=torch.float32,device=0,requires_grad=True)
y=model(data)

class Net0(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.dropout0=nn.Dropout(0.5)
        self.dropout1=nn.Dropout(0.5)


    def forward(self,x):
        print(x)
        x1=self.dropout0(x)
        print(x1)
        x2=self.dropout1(x)
        print(x2)
        x3=self.dropout0(x)
        print(x3)
        xx=torch.rand([100],device=0)
        x4=self.dropout1(x)
        print(x4)
        return x2

torch.cuda.manual_seed(1234)
model=Net0().cuda(0)
model.train()
data=torch.ones([1,10],dtype=torch.float32,device=0,requires_grad=True)
y=model(data)