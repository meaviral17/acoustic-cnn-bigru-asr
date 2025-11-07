# Hybrid ASR with KenLM and LLM Repair

> A multi-stage Automatic Speech Recognition (ASR) pipeline that fuses deep acoustic modeling (CNN–BiGRU), probabilistic decoding (KenLM 5-gram), and semantic refinement (Gemini LLM). This system bridges low-level signal processing with high-level language understanding to produce contextually accurate and human-like transcriptions.

---

## Overview

This project implements a hybrid ASR system that progressively enhances transcriptions through three complementary stages:

1. **Acoustic Encoding** – CNN–BiGRU encoder trained on LibriSpeech to extract spectral and temporal speech features.  
2. **Language Modeling** – Beam/greedy decoding with KenLM fusion for grammatical fluency.  
3. **LLM Refinement** – Context-aware correction using Gemini LLM with teacher-forced guidance.

Each stage contributes to a structured improvement in transcription quality, from noisy phoneme-level hypotheses to coherent, semantically meaningful sentences.

---

## Repository Structure

```

Hybrid-ASR-KenLM-LLM/
├── Hybrid_ASR_with_KenLM_and_LLM_Repair.ipynb   # Main notebook
├── /models/                                     # CNN–BiGRU checkpoints
├── /kenlm/                                      # 5-gram language model binaries
├── /outputs/                                   # Raw and refined transcripts
├── /plots/                                     # Visualizations and heatmaps
└── README.md

```

---

## System Architecture

### (1) Overall ASR Pipeline
*(Insert Diagram 1: Overall Architecture)*  
Flow from **LibriSpeech → Mel Spectrograms → CNN–BiGRU Encoder → CTC + KenLM → LLM Refinement**.

---

### (2) Stage-Wise Model Flow
*(Insert Diagram 2: Two-Level Architecture)*  
Illustrates **Stage 1** (Acoustic & Language Modeling) and **Stage 2** (Post-Processing & Semantic Refinement).

---

### (3) Decoding and Refinement Workflow
*(Insert Diagram 3: Decoding Flow)*  
Shows **CTC/Beam outputs** cleaned, fused via **KenLM**, and refined through **Gemini LLM** with teacher-forced semantic alignment.

---

## Key Highlights

- **CNN–BiGRU Encoder:** Robust bidirectional temporal modeling of speech.  
- **KenLM Fusion:** Enforces linguistic and syntactic consistency.  
- **Gemini LLM Correction:** Converts noisy CTC outputs into fluent English.  
- **Tracked Metrics:** CTC Loss, Word Error Rate (WER), Character Error Rate (CER).  
- **Visual Analysis:** Activation heatmaps, word clouds, and progressive decoding comparisons.

---

## Results Snapshot

| Stage | Description | Example |
|:--|:--|:--|
| CTC Output | Raw sequence with blanks | `<blk><blk>hhhehhihhha...` |
| After KenLM | Decoded hypothesis | `hehofhihehha` |
| After LLM | Refined output | “He hoped there would be stew for dinner.” |

---

## Technical Stack

- **Language:** Python 3  
- **Frameworks:** PyTorch, Torchaudio  
- **Language Model:** KenLM 5-gram  
- **LLM API:** Google Gemini 2.5 Flash  
- **Visualization:** Matplotlib, Seaborn, WordCloud  
- **Dataset:** LibriSpeech (train-clean-100 / test-clean)

---

## Visual Placeholders

| Figure |  |
|:--|:--|
| <img width="1135" height="363" alt="image" src="https://github.com/user-attachments/assets/deb2c2e1-1a8f-47d8-8226-87ec1fb2c663" />
 | Overall Hybrid ASR Architecture |
| <img width="563" height="1077" alt="image" src="https://github.com/user-attachments/assets/f426b366-419f-45fa-bf08-77e1f39c43a5" />
 | Stage-Wise Acoustic–Semantic Flow |
| <img width="1194" height="252" alt="image" src="https://github.com/user-attachments/assets/e33ad68e-a12b-4ede-84c7-21086f16e901" />
 | Decoding and LLM Refinement Flow |


