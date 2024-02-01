export PYTHONPATH="./:$PYTHONPATH"
python quantitative_evaluation/evaluate_benchmark_1_correctness.py \
    --pred_path <path-to-prediction-file-generated-using-inference-script> \
    --output_dir <output-directory-path> \
    --output_json <path-to-save-annotation-final-combined-json-file> \
    --api_key <openai-api-key-to-access-GPT3.5-Turbo-model> \
    --num_tasks 3