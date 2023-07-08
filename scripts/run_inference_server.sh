model=TheBloke/vicuna-7B-1.1-HF
num_shard=2
volume=./models
docker run --gpus all --shm-size 1g -p 8080:80 \
--env CUDA_VISIBLE_DEVICES=2,3 \
-v $volume:/data ghcr.io/huggingface/text-generation-inference:sha-b4aa87d --model-id $model --num-shard $num_shard \
--max-concurrent-requests 20 --max-best-of 1 \
--max-total-tokens 2048 --max-input-length 1500 \
--max-batch-total-tokens 20000 \
