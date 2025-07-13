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
