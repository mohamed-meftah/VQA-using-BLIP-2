# Vision-Language Question Answering (VQA) using BLIP and LoRA Fine-Tuning

## Overview
This repository contains code for training a Vision-Language Question Answering (VQA) model using the **BLIP** (Bootstrapped Language-Image Pretraining) model and fine-tuning it with **LoRA** (Low-Rank Adaptation) for parameter-efficient training. The project processes the [IconDomainVQA](https://example.com) dataset (image-question-answer pairs) to train a model capable of generating answers based on visual and textual inputs.

## Features
- **Dataset Handling**: Automated download, extraction, and preprocessing of the IconDomainVQA dataset.
- **Model Selection**: Utilizes the BLIP-VQA model for image-based question answering.
- **LoRA Fine-Tuning**: Implements parameter-efficient training with Low-Rank Adaptation.
- **Training Pipeline**: Supports mixed-precision training (AMP) and the AdamW optimizer.
- **Evaluation & Inference**: Includes validation metrics and real-world inference capabilities.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -q peft transformers bitsandbytes datasets torch tqdm
gdown 1--CmXocqkrcPR-bbSDztnxBM9dBC8uUJ
unzip /content/IconDomainVQAData.zip
# BLIP-VQA is loaded from Hugging Face
from transformers import BlipForQuestionAnswering
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
outputs = model.generate(**encoding)
generated_text = processor.decode(outputs[0], skip_special_tokens=True)
print(f"Answer: {generated_text}")  # Output: "box"
model.save_pretrained("./save_model")
processor.save_pretrained("./save_model")
