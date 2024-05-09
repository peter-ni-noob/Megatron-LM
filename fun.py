# import torch
# w=torch.tensor([[1.0,1.],[1.,1.]],requires_grad=True)
# x=torch.tensor([[1.0,2.],[4.,3.]],requires_grad=True)
# b=torch.tensor([1.0,4.0],requires_grad=True)
# print(b.size())
# y=torch.matmul(x,w)+b
# c=torch.randn_like(y)
# y.backward(c)
# print(b.grad)
# print(c.sum(dim=0))


