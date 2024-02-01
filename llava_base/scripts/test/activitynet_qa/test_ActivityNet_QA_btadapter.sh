export PYTHONPATH="./:$PYTHONPATH"
python bt_adapter/eval/run_inference_activitynet_qa.py \
    --video_dir <path-to-video-dir> \
    --gt_file_question <test_q.json> \
    --gt_file_answers <test_a.json> \
    --output_dir <path-to-out-dir> \
    --output_name <output-filename> \
    --model-name <path-to-llava-dir> \
    --frames <path-to-video-num_frames> \
    --num-frames 64 \
    --use-btadapter True \
    --btadapter_weight <path-to-btadapter-weight> \
    --projection_path <path-to-projection-weight> \ #Comment out this line for zero-shot evaluation