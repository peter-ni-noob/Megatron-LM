# import util
# import torch

# for i in range(1000):
#     data=torch.randn([10,1024,768])
#     util.save_count_tensor(data,"test")
import sys
sys.path.append("/root/Megatron-LM/")

import myutil
import torch
torch.distributed.init_process_group(
               backend="nccl",
        )
for i in range(1000):
    data=torch.randn([10,1024])
    myutil.save_count_tensor(data,"test",per_iteration_save=100)


# data0=torch.load("/root/Megatron-LM/experiments/mygpt2/activation/0_tensor_cnt_0_test")
# data1=torch.load("/root/Megatron-LM/experiments/mygpt2/activation/100_tensor_cnt_0_test")

# print(data0==data1)
# print(data0.size())
# print(data1.size())
# print(data0)

def test_save_count_tensor():
    assert True