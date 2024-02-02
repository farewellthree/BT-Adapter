# BT-Adapter
This repository is used for video-text pretraining and downstream video task evaluation.

## Getting Started

### Installation
Git clone our repository, creating a python environment and activate it via the following command

```bash
git clone https://github.com/farewellthree/BT-Adapter.git
cd BT-Adapter/mmaction2
conda create --name btadapter_mmaction python=3.10
conda activate btadapter_mmaction
bash install.sh
```

### Prepare Datasets
You can follow [CLIP4clip](https://github.com/ArrowLuo/CLIP4Clip) and [UMT](https://github.com/OpenGVLab/unmasked_teacher/blob/main/multi_modality/DATASET.md) for the acquisition of videos and annotation.

Once the dataset is already, set the path in each config [here](https://github.com/farewellthree/BT-Adapter/tree/main/mmaction2/configs/btadapter).

Considering there might be multiple versions of annotations for the dataset, our code may not be compatible with your annotations. In such cases, you just need to modify the corresponding dataset class in [video_text_dataset.py](https://github.com/farewellthree/BT-Adapter/blob/main/mmaction2/mmaction/datasets/video_text_dataset.py), to output the paths of all videos along with their corresponding captions.

## Training
To train BT-Adapter-OpenaiCLIP-L/14 on Webvid2M, run 
```bash
torchrun --nproc_per_node=8 --master_port=20001 tools/train.py configs/btadapter/pretrain/webvid2m_btadaper.py --launcher pytorch
```

To train BT-Adapter-EvaCLIP-G on Webvid2M, run 
```bash
torchrun --nproc_per_node=8 --master_port=20001 tools/train.py configs/btadapter/pretrain/webvid2m_evabtadapter.py --launcher pytorch
```

## Evaluation
All evaluation configs can be found [here](https://github.com/farewellthree/BT-Adapter/tree/main/mmaction2/configs/btadapter/zero-shot).

For instance, we evaluate BT-Adapter on MSR-VTT, run 
```bash
torchrun --nproc_per_node=8 --master_port=20001 tools/test.py mmaction2/configs/btadapter/zero-shot/msrvtt_btadapterl14.py --checkpoint /Path/to/your/pretrained/weights --launcher pytorch
```
