## FutureCap 

"a future in the past" -- Assassin's Creed 

This repository contains the reference code for the paper _Efficient Modeling of Future Context for Image Captioning_. 
In this paper, we aims to utilize mask-based non-autoregressive image caption (NAIC) model to improve the performance of conventional image captioning model with dynamic distribution calibration.

## 1. Requirements

torch==1.10.1 

transformers==4.11.3 

clip

## 2. Dataset 

To run the code, annotations and detection features for the COCO dataset are needed. Please download the annotations file [annotations.zip](https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view?usp=sharing) and extract it.
Detection features are computed with the pre-trained model provided by [CLIP](https://github.com/openai/CLIP). 

## 3. Training 

First, run  `python train_NAIC.py` to obtain the non-autoregressive image captioning model, which serves as a teacher model. Then, run `python train_combine.py` to conduct a distribution calibration of conventional transformer image captioning model.

Training arguments are as followings:

| Argument | Possible values |
|------|------|
| `--batch_size` | Batch size (default: 10) |
| `--workers` | Number of workers (default: 0) |
| `--warmup` | Warmup value for learning rate scheduling (default: 10000) |
| `--resume_last` | If used, the training will be resumed from the last checkpoint. |
| `--data_path` | Path to COCO dataset file |
| `--annotation_folder` | Path to folder with COCO annotations |


## 4. Evaluation 

To reproduce the results reported in our paper, download the pretrained model file from google drive and place it in the ckpt folder.

Run `python inference.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--batch_size` | Batch size (default: 10) |
| `--workers` | Number of workers (default: 0) |
| `--data_path` | Path to COCO dataset file |
| `--annotation_folder` | Path to folder with COCO annotations |



## 5. Acknowledgements 

This repository is based on [M2T](https://github.com/aimagelab/meshed-memory-transformer) and [Huggingface](https://github.com/huggingface/transformers), and you may refer to it for more details about the code. 


