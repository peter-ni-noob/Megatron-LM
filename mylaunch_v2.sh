GPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
CHECKPOINT_PATH=/root/Megatron-LM/experiments/mygpt2
VOCAB_FILE=/root/Megatron-LM/experiments/gpt2_tokenizer_element/gpt2-vocab.json
MERGE_FILE=/root/Megatron-LM/experiments/gpt2_tokenizer_element/gpt2-merges.txt
DATA_PATH=/data/megatron-origin-dataset/my-gpt2_text_document
GPT_ARGS="--num-layers 1
--hidden-size 768
--num-attention-heads 12
--seq-length 1024
--max-position-embeddings 1024
--micro-batch-size 12
--global-batch-size 192
--lr 0.0005
--train-iters 150000
--lr-decay-iters 150000
--lr-decay-style cosine
--lr-warmup-iters 2000
--weight-decay .1
--adam-beta2 .999
--fp16
--log-interval 10
--save-interval 2000
--eval-interval 200
--eval-iters 10
"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_HOME=/usr/local/cuda
TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"
torchrun $DISTRIBUTED_ARGS \
        pretrain_gpt_nexus_dataset.py \
        --tensor-model-parallel-size 2 \
        --pipeline-model-parallel-size 1 \
        $GPT_ARGS \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        --distributed-backend nccl \
        $TENSORBOARD_ARGS