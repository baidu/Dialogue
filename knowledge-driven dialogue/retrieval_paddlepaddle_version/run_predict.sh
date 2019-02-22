export CUDA_VISIBLE_DEVICES=5

PYTHON_PATH="/home/ld/zhangxiyuan01/paddle_install/paddle/paddle-env/python27-gcc482/bin/python"

$PYTHON_PATH -u predict.py --task_name 'match' \
                   --use_cuda \
                   --batch_size 128 \
                   --init_checkpoint "./model" \
                   --data_dir "./data" \
                   --vocab_path "./dict/char_dict/char.dict" \
                   --max_seq_len 170

