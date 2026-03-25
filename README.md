<div align="center">

# SynSym: A Synthetic Data Generation Framework for Psychiatric Symptom Identification

[![KDD](https://img.shields.io/badge/KDD-2026-blue.svg)](https://kdd.org/kdd2026/)
[![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg)](https://arxiv.org/abs/TODO)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

[Migyeong Kang](https://github.com/gyeong707)¹, Jihyun Kim¹, Hyolim Jeon¹, Sunwoo Hwang¹, Jihyun An², Yonghoon Kim³, Haewoon Kwak⁴, Jisun An⁴, Jinyoung Han¹†

¹Sungkyunkwan University, ²Samsung Medical Center, ³Omnicns, ⁴Indiana University

</div>

---

## ✅ Overview

**SynSym** is an LLM-based synthetic data generation framework for constructing generalizable datasets for psychiatric symptom identification. 
Given only a list of symptom classes and their brief descriptions, SynSym automatically produces diverse, clinically relevant linguistic expressions that reflect the presence of each symptom.

Psychiatric symptom identification on social media remains challenging due to the difficulty of constructing high-quality training datasets: expert annotation is costly, labeling protocols lack standardization, and most existing datasets are collected from a single platform, limiting model generalizability. 
SynSym addresses these limitations by leveraging LLMs to generate synthetic training data that is both clinically valid and stylistically diverse.


<p align="center">
  <img width="1463" height="523" alt="Image" src="https://github.com/user-attachments/assets/e03f34d8-8ddf-48dd-a526-02fe90853863" />
  <br>
  <em>Figure 1: The SynSym framework consists of four stages: (1) Symptom Concept Expansion, (2) Single-Symptom Expression Generation, (3) Multi-Symptom Expression Generation, and (4) Synthetic Data Evaluation.</em>
</p>


## 🔧 Framework

SynSym consists of four sequential stages:

### 1. Symptom Concept Expansion
Given a high-level symptom keyword and its brief clinical description, SynSym prompts an LLM to generate fine-grained sub-concepts that capture diverse real-world manifestations of the symptom. Outputs are manually reviewed to eliminate redundancy and ensure clear separation between related symptoms.

### 2. Single-Symptom Expression Generation
For each sub-concept, SynSym generates synthetic expressions in two distinct styles:
- **Clinical Style**: Formal, objective statements incorporating clinical terminology
- **Colloquial Style**: Informal, first-person descriptions reflecting how individuals express symptoms on social media

### 3. Multi-Symptom Expression Generation
To reflect the co-occurrence of multiple symptoms within a single post, SynSym generates symptom combinations guided by clinical knowledge of comorbidity patterns in depression. Expressions are then generated that naturally describe these combinations in both clinical and colloquial styles.

### 4. Synthetic Data Evaluation
Each generated expression is scored by an LLM on a 5-point scale for alignment with its intended symptom label. Expressions below a quality threshold are removed to ensure label reliability.

---

## 🗂️ Dataset Statistics

| Statistic | PsySym | PRIMATE | D2S | **SynSym** |
|-----------|--------|---------|-----|------------|
| # Samples | 1,433 | 2,003 | 1,717 | **18,254** |
| # Symptom Classes | 14 | 9 | 9 | **14** |
| Avg. Post Length | 18.5 | 299.3 | 17.9 | **23.2** |
| Avg. Symptoms per Sample | 1.4 | 3.5 | 1.2 | **1.6** |
| Avg. Samples per Class | 143.5 | 779.3 | 227.4 | **2,119.5** |
| Min. Samples per Class | 51 | 195 | 46 | **1,186** |

The synthetic dataset comprises **12,621 single-symptom** and **5,633 multi-symptom** entries across 14 DSM-5–based symptom categories.

---

## 👨‍⚕️ Expert Validation

Clinical validity was assessed by two licensed psychiatrists on a 5-point Likert scale:

| Component | Expert 1 | Expert 2 | Agreement |
|-----------|----------|----------|-----------|
| Expanded Sub-Concepts (253 items) | 4.61 | 4.57 | 94.86% |
| Synthetic Expressions (300 items) | 4.99 | 5.00 | 99.66% |

---

## 📊 Key Results

### Overall Performance
Models trained solely on SynSym-generated data (**SynSym w/o Real**) achieve performance comparable to models trained on real benchmark datasets. Further fine-tuning on real data (**SynSym with Real**) consistently outperforms all baselines across most metrics.

| Model | PsySym Macro-F1 | PRIMATE Macro-F1 | D2S Macro-F1 |
|-------|:---------:|:----------:|:-------:|
| BERT | 0.814 | 0.646 | 0.600 |
| DeBERTa | 0.796 | 0.614 | 0.563 |
| MentalBERT | 0.811 | 0.643 | 0.603 |
| GPT-4o ZSL | 0.808 | 0.567 | 0.588 |
| GPT-4o CoT | 0.807 | 0.568 | 0.592 |
| SynSym w/o Real | 0.778 | 0.557 | 0.518 |
| **SynSym with Real** | **0.830** | **0.650** | **0.614** |

---

## 🚀 Getting Started

### Repository Structure

| Directory | Description |
|-----------|-------------|
| `synthetic-data-generation/` | Synthetic data generation pipeline (Stages S1–S5) |
| `symptom-identification/` | Symptom identification model training and evaluation |
| `resources/` | Shared resources (e.g., symptom dictionary) |

---

### 1. Synthetic Data Generation

The synthetic data generation pipeline proceeds **stage by stage**. Each stage directory contains a dedicated README with detailed instructions on inputs, outputs, and execution. Follow the order below and refer to the corresponding README for specific commands.

| Stage | Directory | Description |
|-------|-----------|-------------|
| **S1** Symptom Concept Expansion | [`synthetic-data-generation/s1_symptom_concept_expansion/`](synthetic-data-generation/s1_symptom_concept_expansion/README.md) | Decompose each symptom into fine-grained sub-concepts |
| **S2** Synthetic Generation (Single) | [`synthetic-data-generation/s2_synthetic_generation_single/`](synthetic-data-generation/s2_synthetic_generation_single/README.md) | Generate single-symptom expressions |
| **S3** Co-occurrence Calculation | [`synthetic-data-generation/s3_calcuate_occurrence/`](synthetic-data-generation/s3_calcuate_occurrence/README.md) | Compute symptom co-occurrence patterns |
| **S4** Synthetic Generation (Multi) | [`synthetic-data-generation/s4_synthetic_generation_multi/`](synthetic-data-generation/s4_synthetic_generation_multi/README.md) | Generate multi-symptom expressions |
| **S5** Quality Evaluation | [`synthetic-data-generation/s5_quality_evaluation/`](synthetic-data-generation/s5_quality_evaluation/README.md) | Evaluate and filter generated expressions |

> API key setup, batch job configuration, and input JSON formats are described in each stage's README.

---

### 2. Symptom Identification

Experiment code for multi-label symptom identification is located in `symptom-identification/`. Navigate to the directory before running any scripts.
```bash
cd symptom-identification
```

#### 2.1 Pre-training on Synthetic Data

To pre-train a MentalBERT multi-label classifier on the SynSym-generated synthetic dataset:
```bash
bash script/run_symptom_identification_mlc_synth.sh
```

- Internally calls `train.py` with `-mode pretrain`.
- Modify `MODEL_NAME`, `OPTION_NAME_1` / `OPTION_NAME_2`, `SEED`, and `CKP_FOLD` at the top of the script to run different configurations.


#### 2.2 Fine-tuning and Evaluation on Real Data

To fine-tune on real labeled data and evaluate on the test set:
```bash
bash script/run_symptom_identification_test_real.sh
```

- By default, evaluation is done via `test.py`.
- For datasets with predefined train/valid/test splits (e.g., PsySym), use `test_psysym.py` instead, and reports are generated via `auto_report.py`.
- Modify `MODEL_PATH`, `DATA_NAME`, `CONFIG_PATH`, `CKP_FOLD`, `SEED`, and `FOLD` as needed. If a fine-tuned checkpoint already exists, pass its path to `-ckp`.

---

#### 2.3 Benchmark Datasets

This repository does **not** include the original benchmark datasets. Please obtain each dataset directly from the original authors in accordance with their respective terms of use.

| Dataset | Reference | Config Directory |
|---------|-----------|-----------------|
| **PsySym** | Zhang et al., *Symptom Identification for Interpretable Detection of Multiple Mental Disorders on Social Media*, EMNLP 2022 | `config/Synth/depression/Psysym/` |
| **PRIMATE** | Gupta et al., *Learning to Automate Follow-Up Question Generation Using Process Knowledge for Depression Triage on Reddit Posts*, CLPsych 2022 | `config/Synth/depression/Primate/` |
| **D2S** | Yadav et al., *Identifying Depressive Symptoms from Tweets: Figurative Language Enabled Multitask Learning Framework*, COLING 2020 | `config/Synth/depression/D2S/` |

Once obtained, update the `train_dir`, `valid_dir`, `test_dir`, and `dict_dir` fields in the corresponding JSON config files to point to your local data paths.

## 📝 Citation

> 📌 Citation information will be updated upon official publication.

---


