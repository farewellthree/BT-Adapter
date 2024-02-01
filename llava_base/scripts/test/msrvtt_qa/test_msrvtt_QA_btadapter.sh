export PYTHONPATH="./:$PYTHONPATH"
python bt_adapter/eval/run_inference_msvd_qa.py \
    --video_dir <path-to-video-dir> \
    --gt_file <msrvtt_qa_test.json> \
    --output_dir <path-to-out-dir> \
    --output_name <output-filename> \
    --model-name <path-to-llava-dir> \
    --num-frames 64 \
    --use-btadapter True \
    --btadapter_weight <path-to-btadapter-weight> \
    --projection_path <path-to-projection-weight> \ #Comment out this line for zero-shot evaluation