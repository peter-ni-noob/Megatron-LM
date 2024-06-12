#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_HOME=/usr/local/cuda


CHECKPOINT_PATH=/root/Megatron-LM/experiments/mygpt2
VOCAB_FILE=/root/Megatron-LM/experiments/gpt2_tokenizer_element/gpt2-vocab.json
MERGE_FILE=/root/Megatron-LM/experiments/gpt2_tokenizer_element/gpt2-merges.txt
DATA_PATH=/data/my-gpt2_text_document

GPT_ARGS="
    --num-layers 1 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 8 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

DISTRIBUTED_ARGS="
    --nproc_per_node 1 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6001
"

torchrun pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
