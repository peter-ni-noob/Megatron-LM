#get model runtime info
import torch


class Counter:
    #count iteration
    cnt=0
    main_rank=0
    pair={}
 


def save_count_tensor(target_tensor:list[torch.Tensor]|torch.Tensor,info:str,rank=0,per_iteration_save=100,default_save_path="/root/Megatron-LM/experiments/mygpt2/activation"):
    """according to rank,save tensor
    文件命名: rank_count__tensor_cnt_(list count)_info
    """
    import os
    cnt=0
    rank_info=str(rank)+info
    if rank_info not in Counter.pair:
        Counter.pair[rank_info]=0
    if(rank==torch.distributed.get_rank() and Counter.pair[rank_info] % per_iteration_save==0):
        if isinstance(target_tensor,list):
            for i in target_tensor:
                filename="rank_"+str(rank)+"_"+str(Counter.pair[rank_info])+"_tensor_cnt"+str(cnt)+"_"+info
                torch.save(i,os.path.join(default_save_path,filename))
                cnt+=1
        else:
            filename="rank_"+str(rank)+"_"+str(Counter.pair[rank_info])+"_tensor_cnt_0_"+info
            torch.save(target_tensor,os.path.join(default_save_path,filename))
    Counter.pair[rank_info]+=1
   
