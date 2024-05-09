import torch 



data=[]
data.append(torch.load("/root/Megatron-LM/experiments/mygpt2/activation/rank_0_100_tensor_cnt_0_beforemlp_0").cpu())
data.append(torch.load("/root/Megatron-LM/experiments/mygpt2/activation/rank_1_900_tensor_cnt_0_beforemlp_1").cpu())
res=torch.load("/root/Megatron-LM/experiments/mygpt2/activation/rank_0_900_tensor_cnt_0_MLPout").cpu()

print(data[0])
print(res.size())
# print(data[0]+data[1]==res)