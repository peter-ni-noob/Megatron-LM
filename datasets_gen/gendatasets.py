import json
import math

import numpy as np
import torch
import logging
from dataclasses import dataclass

import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any
from utils import get_args

from datasets import _LambadaDataset

from utils import get_tokenizer
from utils import set_args
from utils import _build_tokenizer
from utils import get_ltor_masks_and_position_ids



import sys
import os





@dataclass
class Config:
    valid_data:str="/data/lambada_origin/lambada_test.json"
    seq_length:int=1024
    strict_lambada:bool=False
    vocab_file:str="/root/Megatron-LM/experiments/gpt2_tokenizer_element/gpt2-vocab.json"
    merge_file:str="/root/Megatron-LM/experiments/gpt2_tokenizer_element/gpt2-merges.txt"
    save_path:str="/root/Megatron-LM/experiments/"
    micro_batch_size:int =1
    num_workers:int =2
    reset_position_ids:bool =False
    reset_attention_mask:bool=False
    eod_mask_loss:bool =False
    log_interval:int =10







def _build_lambada_dataset(args):
    """Build lambada dataset."""
    
    
    tokenizer = get_tokenizer()

    # assert len(args.valid_data) == 1
    val_dataset = _LambadaDataset(args.valid_data, tokenizer.eod, tokenizer,
                                  args.seq_length, args.strict_lambada)
    logging.info(' > found {} samples.'.format(len(val_dataset)))

    return val_dataset



def build_data_loader(dataset, micro_batch_size, num_workers, drop_last,
        task_collate_fn=None):
    """Data loader. Note that batch-size is the local (per GPU) batch-size."""

    # Sampler.
    world_size = 1
    rank = 1
    # sampler = torch.utils.data.distributed.DistributedSampler(
    #     dataset, num_replicas=world_size, rank=rank)

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=micro_batch_size,
                                            #   sampler=sampler,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              drop_last=drop_last,
                                              pin_memory=True,
                                              collate_fn=task_collate_fn)

    return data_loader



def process_batch(batch):
    """Process batch and produce inputs for the model."""
    args = get_args()
    tokenizer = get_tokenizer()

    loss_mask = batch['pad_mask'].long().cuda().contiguous().byte()
    tokens_ = batch['text'].long().cuda().contiguous()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)#值为多少？

    return tokens, labels, attention_mask, position_ids, loss_mask




def gen_format_data(data_loader):
    args=get_args()
    with torch.no_grad():
        datachunk=[]
        labelchunk=[]
        maskchunk=[]
        cnt=0
        size=len(data_loader)
        # For all the batches in the dataset.
        for iteration, batch in enumerate(data_loader):
            if iteration % args.log_interval == 0:
                logging.info('> working on batch: {}'.format(iteration))
            # Forward evaluation.
            tokens, labels, attention_mask, position_ids, loss_mask = process_batch(
        batch)
            datachunk.append(torch.reshape(tokens,[-1]))
            labelchunk.append(torch.reshape(labels,[-1]))
            maskchunk.append(torch.reshape(loss_mask,[-1]))
            cnt+=1
            if(cnt%10000==0 or cnt==size):
                
                main_dir=args.save_path #modify this path to save result npy
                bk=cnt//10000
                path=os.path.join(main_dir,str(bk))
                os.mkdir(path=path)

                outputdata=torch.stack(datachunk,dim=0)
                outputlabel=torch.stack(labelchunk,dim=0)
                mask=torch.stack(maskchunk,dim=0)
                outputdata[outputdata>50256]=50256
                outputlabel[outputlabel>50256]=50256
                outputdata=outputdata.cpu().numpy()
                outputdata=np.ascontiguousarray(outputdata,dtype=np.float32)
                outputlabel=outputlabel.cpu().numpy()
                outputlabel=np.ascontiguousarray(outputlabel,dtype=np.float32)
                mask=mask.cpu().numpy()
                mask=np.ascontiguousarray(mask,dtype=np.float32)

                np.save(os.path.join(path,"token_float32.npy"),outputdata)
                np.save(os.path.join(path,"label_float32.npy"),outputlabel)
                np.save(os.path.join(path,"loss_mask.npy"),mask)

                datachunk.clear()
                labelchunk.clear()
                maskchunk.clear()
                print(bk)
        sys.exit()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)


    args = Config()
    logging.info(f"config:{args}")
    #init args,tokenizer
    set_args(args)
    _build_tokenizer()
    
    dataset = _build_lambada_dataset(args)
    dataloader = build_data_loader(dataset, args.micro_batch_size,
                                   args.num_workers, drop_last=False)
    gen_format_data(dataloader)
    
    