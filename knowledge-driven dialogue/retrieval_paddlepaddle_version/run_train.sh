export CUDA_VISIBLE_DEVICES=4

PYTHON_PATH="/home/ld/zhangxiyuan01/paddle_install/paddle/paddle-env/python27-gcc482/bin/python"

$PYTHON_PATH -u train.py --task_name 'match' \
                   --use_cuda \
                   --batch_size 128 \
                   --data_dir "./data" \
                   --vocab_path "./dict/char_dict/char.dict" \
                   --checkpoints "./model" \
                   --save_steps 1000 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.1 \
                   --validation_steps 1000000 \
                   --skip_steps 100 \
                   --learning_rate 0.1 \
                   --epoch 30 \
                   --max_seq_len 170 \

