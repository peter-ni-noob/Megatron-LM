export CUDA_DEVICE_MAX_CONNECTIONS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=3,2

/usr/local/bin/torchrun \
    --nproc_per_node 1 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6001 \
    /root/Megatron-LM/tasks/main.py \
    --task LAMBADA \
    --num-layers 1 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --fp16 \
    --vocab-file /root/Megatron-LM/experiments/gpt2_tokenizer_element/gpt2-vocab.json \
    --valid-data /data/lambada_origin/lambada_test.json \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file /root/Megatron-LM/experiments/gpt2_tokenizer_element/gpt2-merges.txt \
    --load /root/Megatron-LM/experiments/mygpt2 \
    --micro-batch-size 1 \
    --log-interval 10 \
    --no-load-optim \
    --no-load-rng