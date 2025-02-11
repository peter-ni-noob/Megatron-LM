# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch
import numpy as np 

import os
if os.getenv("LOCAL_RANK") is None:
    os.environ["LOCAL_RANK"]="0"
from functools import partial
from typing import Union
from megatron.training import get_args
from megatron.training.global_vars import set_global_variables
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import load_args_from_checkpoint
# from megatron.training import print_rank_0
# from megatron.training import get_timers
# from megatron.training import get_tokenizer
# from megatron.core import mpu
# from megatron.core.enums import ModelType
# from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
# from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
# from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset

from megatron.core.tensor_parallel.layers import VocabParallelEmbedding, RowParallelLinear, ColumnParallelLinear
from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
import megatron.legacy.model as m
from megatron.legacy.model.transformer import ParallelAttention

class Test:        

    transformer_config = TransformerConfig(num_layers=1, hidden_size=12,
                                           num_attention_heads=2, use_cpu_initialization=True,attention_dropout=0.0,hidden_dropout=0.0)
    tmp_dir="./temp/"
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.timeout(100)
    def test_embedding_init(self):

        Utils.initialize_model_parallel(1, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        

        tp1 = VocabParallelEmbedding(num_embeddings=16, embedding_dim=4,
                                     init_method=self.transformer_config.init_method,
                                     config=self.transformer_config).weight
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(4, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(41)  # intentionally different.
        tp4 = VocabParallelEmbedding(num_embeddings=16, embedding_dim=4,
                                     init_method=self.transformer_config.init_method,
                                     config=self.transformer_config).weight

        if torch.distributed.get_rank() == 0:
            assert tp4.shape[0] * 4 == tp1.shape[0]
            assert torch.allclose(tp1[:4], tp4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.timeout(100)
    def test_row_init(self):

        Utils.initialize_model_parallel(1, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        tp1 = RowParallelLinear(input_size=16, output_size=16,
                                init_method=self.transformer_config.init_method,
                                bias=True, input_is_parallel=False,
                                config=self.transformer_config,
                                skip_bias_add=False).weight
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(4, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(41)  # intentionally different.
        tp4 = RowParallelLinear(input_size=16, output_size=16,
                                init_method=self.transformer_config.init_method,
                                bias=True,
                                input_is_parallel=False,
                                config=self.transformer_config,
                                skip_bias_add=False).weight
        
        if torch.distributed.get_rank() == 0:
            assert tp4.shape[1] * 4 == tp1.shape[1]
            assert torch.allclose(tp1[:, :4], tp4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.timeout(100)
    def test_col_init(self):

        Utils.initialize_model_parallel(1, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        tp1 = ColumnParallelLinear(input_size=16, output_size=16,
                                   init_method=self.transformer_config.init_method,
                                   bias=True, config=self.transformer_config,
                                   skip_bias_add=False).weight
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(4, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(41)  # intentionally different.
        tp4 = ColumnParallelLinear(input_size=16, output_size=16,
                                   init_method=self.transformer_config.init_method,
                                   bias=True, config=self.transformer_config,
                                   skip_bias_add=False).weight
        
        if torch.distributed.get_rank() == 0:
            assert tp4.shape[0] * 4 == tp1.shape[0]
            assert torch.allclose(tp1[:4], tp4)
    @classmethod
    def save_npy(cls,basename: str, x: Union[torch.Tensor, np.ndarray]):
        if isinstance(x, torch.Tensor):
            x = x.cpu().detach().numpy().astype(np.float32)
        assert isinstance(x, np.ndarray)
        np.save(os.path.join(cls.tmp_dir, basename), x)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.timeout(100)
    def test_ParallelAttention(self):

        Utils.initialize_model_parallel(1, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
        args=parse_args(None,False)

        if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
            assert args.load is not None, "--use-checkpoints-args requires --load argument"
            load_args_from_checkpoint(args)
        validate_args(args, args_defaults)

        set_global_variables(args)

        f = ParallelAttention(self.transformer_config,0).cuda()
        f.train()
        # mask = torch.triu(torch.ones(2, 2), diagonal=1).bool().cuda().reshape([1,2,2]).repeat(2,1,1)
        torch.save(f.state_dict(),"./PMHAweight.pt")
        x=torch.randn([2,2,12],dtype=torch.float32).requires_grad_().cuda()
        # print(x)
        
        y,bias=f(x,None)
        self.save_npy("y_PMHA",y)
        grad=torch.ones_like(y).cuda()
        self.save_npy("grad",grad)
        y.backward(grad)
        self.save_npy("query_key_value_weight_grad",f.query_key_value.weight.grad)
        self.save_npy("query_key_value_bias_grad",f.query_key_value.bias.grad)
        self.save_npy("dense_weight_grad",f.dense.weight.grad)
        self.save_npy("x_PMHA",x)
        # self.save_npy("xgrad",x.grad)
        self.save_npy("dense_bias_grad",f.dense.bias.grad)
        Utils.destroy_model_parallel()

        # Utils.initialize_model_parallel(2, 1)
        # torch.manual_seed(42)
        # model_parallel_cuda_manual_seed(41)  # intentionally different.
        # f = m.transformer.ParallelAttention(self.transformer_config,0).cuda()

        # x=torch.randn([2,2,12]).requires_grad_().cuda()
        # print(x)
        # y,bias=f(x,None)
        # print(y)
        # Utils.destroy_model_parallel()

        
        # if torch.distributed.get_rank() == 0:
        #     assert tp4.shape[0] * 4 == tp1.shape[0]
        #     assert torch.allclose(tp1[:4], tp4)


a=Test()
a.test_ParallelAttention()
print("done")