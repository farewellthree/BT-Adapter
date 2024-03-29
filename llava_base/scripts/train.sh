export PYTHONPATH="./:$PYTHONPATH"
torchrun --nproc_per_node=8 --master_port=29001 bt_adapter/train/train_mem.py \
          --model_name_or_path /Path/to/your/llava-7b-lightening-v1-1 \
          --mm_vision_tower True \
          --version v1 \
          --data_path /Path/to/your/video_chatgpt_training.json \
          --video_folder /Path/to/your/activity_CLIPL14_8layer \
          --tune_mm_mlp_adapter True \
          --mm_use_vid_start_end \
          --bf16 False \
          --fp16 True \
          --output_dir ./BTAdapter_LLava_7B-1.1_Checkpoints \
          --num_train_epochs 3 \
          --per_device_train_batch_size 1 \
          --per_device_eval_batch_size 1 \
          --gradient_accumulation_steps 1 \
          --evaluation_strategy "no" \
          --save_strategy "steps" \
          --save_steps 3000 \
          --save_total_limit 3 \
          --learning_rate 2e-5 \
          --weight_decay 0. \
          --warmup_ratio 0.03 \
          --lr_scheduler_type "cosine" \
          --logging_steps 100 \
          --tf32 False \
          --model_max_length 2048 \
          --gradient_checkpointing True \
          --lazy_preprocess True