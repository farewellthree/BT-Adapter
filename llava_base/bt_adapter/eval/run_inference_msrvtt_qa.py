import os
import argparse
import json
from tqdm import tqdm
from bt_adapter.eval.model_utils import initialize_model, load_video, load_video_rawframes
from bt_adapter.inference import video_chatgpt_infer

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--projection_path", type=str, required=True)
    parser.add_argument("--num-frames", type=int, required=False, default=100)
    parser.add_argument("--use-btadapter", type=bool, required=False, default=False)
    parser.add_argument("--btadapter_weight", type=str, required=False)

    return parser.parse_args()


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args)
    # Load both ground truth file containing questions and answers
    with open(args.gt_file) as file:
        gt_file = json.load(file)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results
    conv_mode = args.conv_mode

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    index = 0
    
    for sample in tqdm(gt_file):
        video_name = sample['video']
        question = sample['question']
        answer = sample['answer']
        index += 1

        sample_set = {'id': id, 'question': question, 'answer': answer}
        video_path = os.path.join(args.video_dir, video_name)
        # Check if the video exists
        if video_path is not None:
            video_frames = load_video(video_path, num_frm=args.num_frames)

        try:
            # Run inference on the video and add the output to the list
            output = video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower,
                                             tokenizer, image_processor, video_token_len)
            sample_set['pred'] = output
            output_list.append(sample_set)
        except Exception as e:
            print(f"Error processing video file '{video_path}': {e}")

    # Save the output list to a JSON file
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
        json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
