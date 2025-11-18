# Qwen2.5-VL-7B Fine-Tuning on Nautilus (NRP)
## Visual-Language Model for Jam-Causing Material Detection in MRFs

---

## Overview
This repository documents the fine-tuning of **Qwen2.5-VL-7B** using **QLoRA** on the **Pacific Research Platform (PRP) Nautilus** Kubernetes cluster.  
The goal is to develop a **Vision-Language Model (VLM)** capable of identifying **jam-causing material** on recycling conveyor belts in Material Recovery Facilities (MRFs).

The model is trained on a **proprietary dataset** containing real conveyor-belt images from active MRF environments, including challenging conditions such as variable lighting, cluttered scenes, motion blur, overlapping waste, and diverse contamination patterns.

---

## Background
Urban and multi-family residential areas suffer from low recycling participation and high contamination due to:

- Missing or unclear signage  
- Behavioral barriers  
- Inconsistent reminders  
- Overflowing bins  
- Difficulty distinguishing recyclables, organics, and landfill waste  

These issues lead to improper sorting, which negatively affects downstream MRF operations.

One of the most expensive operational problems in modern recycling facilities is **jam-causing material**. These include:

- Plastic film  
- Ropes and cords  
- Hoses  
- Straps  
- Wire  
- Tanglers of any kind  
- Flexible packaging and long linear items  

These objects wrap around rotating machinery and **cause immediate shutdowns**, reducing throughput and increasing labor costs.

---

## Why Jam-Causing Material Matters
Jam-causing material leads to:

1. **Machine Stoppages**  
   Equipment must be shut down to manually cut tangled items.

2. **Higher Maintenance Costs**  
   Frequent removal increases technician labor, wear, and tool usage.

3. **Lower Throughput**  
   Jams slow the line, reducing total recoverable material.

4. **Budget Impact**  
   Unexpected downtime drains operational budgets.

5. **Reduced Sustainability Performance**  
   Slowdowns reduce material recovery and increase landfill volumes.

An automated detector focused specifically on jam-causing items can prevent many of these problems.

---

## Project Motivation
This work supports the larger **Recykool Project**, which brings together:

1. **Citizen Science**  
   Public participation in capturing and labeling waste images.

2. **AI Digital Twins**  
   Virtual simulations of MRF operations that use real data to identify inefficiencies.

3. **Vision-Language Models**  
   Models like Qwen2.5-VL provide classification, bounding boxes, and natural-language reasoning about waste on conveyor belts.

Fine-tuning Qwen on jam-causing materials enables earlier detection and intervention, saving money and improving facility performance.

---

## Fine-Tuning Approach

### Model
- Base Model: **Qwen2.5-VL-7B-Instruct**  
- Method: **QLoRA (4-bit quantization)**  
- Environment: **Nautilus GPU Pod + PVC**  
- Training method uses custom YAML job files and adapter-based fine-tuning.

### Dataset
- Proprietary dataset 

### Capabilities After Fine-Tuning
- Detect jam-causing waste  
- Predict bounding boxes  
- Output class names + reasoning  
- Support multimodal prompts (image + text)  
- Operate under MRF environmental variability

---

## Goals of This Model
1. **Detect jam-causing items early** to prevent machinery damage.  
2. **Assist operators** with real-time bounding-box visualization.  
3. **Support digital twin simulations** of waste flow.  
4. **Reduce maintenance and downtime costs** for MRFs.  
5. **Increase throughput and recovery rates** by preventing jams.  

## Acknowledgments
This work is part of the **Recykool Project**, which aims to improve recycling systems, reduce contamination, and enhance environmental sustainability using AI and digital twin technologies.  
Training infrastructure is supported by the **Pacific Research Platform (PRP) Nautilus**.

Project webiste:
- [ARCS](https://arcs.center/workers-and-technology-together-watt-and-recykool-reduce-methane-landfill-emissions-through-citizen-science-community-outreach/)
