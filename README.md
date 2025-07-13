## 🧠 Introduction

**Vision-Language Models (VLMs)** have seen remarkable progress, especially with the rise of **mobile-friendly architectures** like [MobileVLM](https://github.com/Meituan-AutoML/MobileVLM). These lightweight models unlock a wide range of applications across edge devices and real-time scenarios. However, fine-tuning such models remains challenging due to the **high computational cost** and **modality complexity** (text + image).

To tackle this, **Low-Rank Adaptation (LoRA)** has been widely adopted as a parameter-efficient fine-tuning strategy. Yet, traditional LoRA with **fixed-rank** settings often lacks the flexibility and expressiveness needed for effectively training mobile VLMs.

### 🌊 Enter HyDRA

We present **HyDRA**, a novel **Hybrid Decomposed Rank Adaptation** framework designed for **efficient and adaptive fine-tuning of mobile VLMs**. HyDRA introduces:

- ✅ **Hierarchical Optimization**  
  - **Coarse-grained scheduling** assigns different ranks to different transformer layers.  
  - **Fine-grained tuning** further adjusts rank **within** each layer for better control and expressiveness.

- 🔁 **Dynamic Rank Adjustment**  
  - A lightweight **performance prediction model** automatically learns and updates rank configurations **end-to-end during fine-tuning**.

### 📈 Key Benefits

- Up to **4.7% performance improvement** across multiple benchmarks, without increasing trainable parameter counts.
- Outperforms **fixed-rank LoRA** and even **full fine-tuning** in some tasks.
- Designed specifically for **mobile VLMs** with both image and text modalities.

## ⚙️ HyDRA Training Process

The training process of **HyDRA** also follows a **two-stage** design built on top of [MobileVLM](https://github.com/Meituan-AutoML/MobileVLM):

### 🧩 Stage I: Initialization Pretrain

❄️ frozen vision encoder + 🔥 learnable LDP projector + ❄️ frozen LLM

- Training time: ~1–1.5 hours for it on 8× A100 (80G)  


### 🧠 Stage II: Instruction Fine-tuning with Dynamic Rank Adaptation

❄️ frozen vision encoder + 🔥 learnable LDP projector + 🔥 learnable LLM + ✅ hybrid-rank LoRA adapter + 🔁 dynamic-rank scheduler

- This stage initializes coarse and fine-grained rank structures for different transformer layers.
- A lightweight performance model automatically adjusts LoRA ranks during fine-tuning.
- Training time: this training process takes around 2~3.5 hours for model on 8x A100 (80G) with a batch size of 128 and an average of approximately 46G/52G of GPU memory required.


> 📝 **Note**: To train with fewer GPUs or lower memory, reduce `per_device_train_batch_size` and increase `gradient_accumulation_steps` accordingly to keep the **global batch size** constant:  
> `per_device_train_batch_size × gradient_accumulation_steps × num_gpus`

## 1️⃣ Prepare Base Checkpoints

Download **MobileLLaMA** base models (1.7B / 2.7B) from Hugging Face:  
- [MobileLLaMA-1.7B](https://huggingface.co/mtgv/MobileLLaMA-1.4B-Chat)  
- [MobileLLaMA-2.7B](https://huggingface.co/mtgv/MobileLLaMA-2.7B-Chat)

Alternatively, you can skip this step — the model will be automatically downloaded by 🤗 `transformers` during training.

---

## 2️⃣ Prepare Data

Assume your project root is `/path/to/project/hydra`, organize your folders as:

cd /path/to/project/hydra
mkdir -p data/pretrain_data data/finetune_data data/benchmark_data

###📦 Pretrain Data (for Stage I)

cd data/pretrain_data
# Download LLaVA-558K dataset (provided by LLaVA team)
cd ${work_dir}/data/pretrain_data
Download the LLaVA-558K from here, which is provided by LLaVA team.
here(https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)

###📚 Instruction Tuning Data (for Stage II)

# Download the instruction tuning annotations

# Collect images from COCO, GQA, OCR-VQA, TextVQA, VisualGnome (Part1 & Part2)
