# BT-Adapter
This repository is used for video instruction tuning and video-conversation evaluation.

## Getting Started

### Installation
Git clone our repository, creating a python environment and activate it via the following command

```bash
git clone https://github.com/farewellthree/BT-Adapter.git
cd BT-Adapter/llava_base
conda create --name btadapter_llava python=3.10
conda activate btadapter_llava
bash scripts/install.sh
```

Additionally, install FlashAttention for training,
```bash
pip install ninja
pip install flash-attn==1.0.7 --no-build-isolation
```

### Prepare Datasets
Please follow [VideoChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT) to prepare VideoChatGPT-100K and VideoChatGPT benchmark.

Specifically, BT-Adapter needs the output of the 20th CLIP layers rather than the 23th layers. We prepare a [script](https://github.com/farewellthree/BT-Adapter/blob/main/mmaction2/save_clip_features.py)
to extract CLIP features from ActivityNet raw frames. To extract features from raw videos, you can modify this script or the [script](https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/scripts/save_spatio_temporal_clip_features.py) from VideoChatGPT.


## Training
To finetuning BT-Adapter-LLaVa on VideoChatGPT-100K, run 
```bash
bash scripts/train.sh
```

## Evaluation
All evaluation script can be found [here](https://github.com/farewellthree/BT-Adapter/tree/main/llava_base/scripts/test).

For instance, to evaluate the temporal score of BT-Adapter-LLaVa on VideoChatGPT benchmark, we first run the inference to get prediction results: 
```bash
bash scripts/test/temporal/test_temp_btadapter.sh
```
and then execute the corresponding evaluation script to perform benchmarking:
```bash
bash scripts/test/temporal/temp_score.sh
```
