# BT-Adapter

### BT-Adapter: Video Conversation is Feasible Without Video Instruction Tuning

---

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-for-all-video-conversation-is-feasible/video-based-generative-performance)](https://paperswithcode.com/sota/video-based-generative-performance?p=one-for-all-video-conversation-is-feasible)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-for-all-video-conversation-is-feasible/video-based-generative-performance-1)](https://paperswithcode.com/sota/video-based-generative-performance-1?p=one-for-all-video-conversation-is-feasible)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-for-all-video-conversation-is-feasible/video-based-generative-performance-5)](https://paperswithcode.com/sota/video-based-generative-performance-5?p=one-for-all-video-conversation-is-feasible)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-for-all-video-conversation-is-feasible/video-based-generative-performance-4)](https://paperswithcode.com/sota/video-based-generative-performance-4?p=one-for-all-video-conversation-is-feasible)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-for-all-video-conversation-is-feasible/video-based-generative-performance-3)](https://paperswithcode.com/sota/video-based-generative-performance-3?p=one-for-all-video-conversation-is-feasible)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-for-all-video-conversation-is-feasible/video-based-generative-performance-2)](https://paperswithcode.com/sota/video-based-generative-performance-2?p=one-for-all-video-conversation-is-feasible)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-for-all-video-conversation-is-feasible/zeroshot-video-question-answer-on-msrvtt-qa)](https://paperswithcode.com/sota/zeroshot-video-question-answer-on-msrvtt-qa?p=one-for-all-video-conversation-is-feasible)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-for-all-video-conversation-is-feasible/zeroshot-video-question-answer-on-msvd-qa)](https://paperswithcode.com/sota/zeroshot-video-question-answer-on-msvd-qa?p=one-for-all-video-conversation-is-feasible)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-for-all-video-conversation-is-feasible/zeroshot-video-question-answer-on-activitynet)](https://paperswithcode.com/sota/zeroshot-video-question-answer-on-activitynet?p=one-for-all-video-conversation-is-feasible)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-for-all-video-conversation-is-feasible/zero-shot-video-retrieval-on-msr-vtt)](https://paperswithcode.com/sota/zero-shot-video-retrieval-on-msr-vtt?p=one-for-all-video-conversation-is-feasible)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-for-all-video-conversation-is-feasible/zero-shot-video-retrieval-on-activitynet)](https://paperswithcode.com/sota/zero-shot-video-retrieval-on-activitynet?p=one-for-all-video-conversation-is-feasible)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-for-all-video-conversation-is-feasible/zero-shot-video-retrieval-on-lsmdc)](https://paperswithcode.com/sota/zero-shot-video-retrieval-on-lsmdc?p=one-for-all-video-conversation-is-feasible)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-for-all-video-conversation-is-feasible/zero-shot-video-retrieval-on-didemo)](https://paperswithcode.com/sota/zero-shot-video-retrieval-on-didemo?p=one-for-all-video-conversation-is-feasible)

---
| Paper | Weights | Video-text Pretraining | Downstream Evaluation | Instruction Tuning | VideoChatGPT Evaluation |
| :---: | :---: | :---: | :---: | :---: | :---: | 
| <a href='https://arxiv.org/abs/2309.15785'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> | <a href='https://huggingface.co/farewellthree/BTAdapter-Weight'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'> | [Video-text Pretraining](mmaction2/README.md#Training) | [Downstream Evaluation](mmaction2/README.md#Evaluation) | [Instruction Tuning](llava_base/README.md#Training) | [VideoChatGPT Evaluation](llava_base/README.md#Evaluation) |

## Overview and Highlights
### 💡 Plug-and-use, parameter-efficient, multimodal-friendly, and temporal-sensitive structure

<div align=center>
<img src="images/method.jpg" width="700px">
</div>

### 💡 State-of-the-art zero-shot results on various video tasks using thousands of fewer GPU hours

<div align=center>
<img src="images/downstream.jpg" width="400px">
</div>

### 💡 State-of-the-art video conversation results with and without video instruction tuning

<div align=center>
<img src="images/radar.jpg" width="400px">
</div>

## Qualitative Results
The Evaluation of BT-Adapter's Performance across Different Situations.
### 👀 The Sequence of Actions
<div align=center>
<img src="images/visualization1.jpg" width="600px">
</div>

### 👀 Unusual Actions
<div align=center>
<img src="images/visualization2.jpg" width="600px">
</div>

### 👀 Complex Actions and Scenes In A Long Video
<div align=center>
<img src="images/visualization3.jpg" width="600px">
</div>

## Citation
If you find the code useful for your research, please consider citing our paper:
```
@article{liu2023one,
  title={One for all: Video conversation is feasible without video instruction tuning},
  author={Liu, Ruyang and Li, Chen and Ge, Yixiao and Shan, Ying and Li, Thomas H and Li, Ge},
  journal={arXiv preprint arXiv:2309.15785},
  year={2023}
}
```
