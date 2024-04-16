import os
import argparse
import torch
import torch.distributed
from datetime import timedelta
_TENSOR_MODEL_PARALLEL_GROUP = None
class A:
    def __init__(self) -> None:
        self.__a=5



def print_os_environ(val=True):
    if val== True:
        env_vars = os.environ
        print("All Environment Variables:")
        for key, value in env_vars.items():
            if key =="rank".upper():
                print(f"{key}: {value}")
                global _TENSOR_MODEL_PARALLEL_GROUP
                _TENSOR_MODEL_PARALLEL_GROUP=value

print_os_environ()
torch.distributed.init_process_group(
            backend="nccl",
            world_size=2,
            rank=int(_TENSOR_MODEL_PARALLEL_GROUP),
            timeout=timedelta(minutes=10),
        )
group_dp=None
group_tt=None
group_pp=None
def test_get_rank_theory():
    global group_dp
    global group_tt
    global group_pp
    for i in range(2):
     if(int(_TENSOR_MODEL_PARALLEL_GROUP) == i):
        group_dp = torch.distributed.new_group(
                    [i]
                )
        group_pp = torch.distributed.new_group(
                [i]
            )
     group_tt = torch.distributed.new_group(
                [0,1]
            )
    print(torch.distributed.get_rank())
    print(torch.distributed.get_rank(group_tt))
    print(torch.distributed.get_rank(group_dp))
test_get_rank_theory()

# print(_TENSOR_MODEL_PARALLEL_GROUP)
# rank = torch.distributed.get_rank()
# # # world_size = torch.distributed.get_world_size()
# # # print(world_size)
# print(rank)

# print_os_environ.ii=5
# print_os_environ()
# print(print_os_environ.ii)
# print(print_os_environ)





# l=int(os.environ["p"])
# print(l)

# parser = argparse.ArgumentParser()
# parser.add_argument("--p", type=int)
# args = parser.parse_args()
# print(args.p)