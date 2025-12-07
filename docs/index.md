---
title: ""
hide:
  - title
  - navigation
  - toc
  - path
---

<p align="center">
  <img src="assets/OmniTrust-full.png" width="1600" alt="OmniTrust Logo">
</p>


<p align="center">
  <a href="https://github.com/S3IC-Lab/OmniTrust" class="md-button md-button--primary">GitHub</a>
  <a href="https://arxiv.org/" class="md-button">Arxiv (Coming Soon)</a>
</p>

---

## 🚀 What is OmniTrust?

**OmniTrust** is a unified, modular, and reproducible evaluation framework designed to assess the trustworthiness of Large Generative Models (LLMs / VLMs / Multimodal Models).  
The platform offers **end-to-end benchmarking pipelines**, enabling standardized evaluation across **six major trustworthiness dimensions**:

- 🔒 **Safety** — jailbreak, harmful response generation, adversarial prompting  
- 🕵️ **Privacy** — PII leakage, memorization, unintended data exposure  
- 🧬 **Detectability** — watermark embedding & detection  
- 🧠 **Hallucination** — factual consistency, grounding, evidence scoring  
- ⚖️ **Fairness** — stereotyping, demographic bias, representational harm  
- 🎯 **Fidelity** — stability, robustness, alignment, answer consistency  

OmniTrust supports **black-box** and **white-box** evaluation settings, with extensible APIs for enterprise and research use.

---

## ✨ Key Features

### 🔧 Modular Evaluation Pipelines  
Each module contains **plug-and-play** evaluation scripts, datasets, metrics, and reporting templates.

### 📦 Black-box & White-box Model Support  
Works with API-based LLMs, local open-source models, and multimodal models.

### 🧪 Reproducible Experiments  
All evaluated methods are versioned and logged with unified interfaces.

### 📈 Enterprise-ready Reporting  
Pluggable scoring system and automated risk grading.

---

## 🏗 System Overview


---

## 🧭 Explore Modules

- 🔒 [Safety Module](safety/index.md)  
- 🕵️ [Privacy Module](privacy/index.md)  
- 🧬 [Detectability Module](detectability/index.md)  
- 🧠 [Hallucination Module](hallucination/index.md)  
- ⚖️ [Fairness Module](fairness/index.md)  
- 🎯 [Fidelity Module](fidelity/index.md)  

---

## 💡 For Developers

Refer to the [developer guideline](develop/index.md) for internal contribution rules, component registration, and API usage.

---

## 📝 Citation

If you use OmniTrust in your research, please cite (Coming Soon):

```
@article{omni2025,
  title={OmniTrust: A Unified Platform for Evaluation of Trustworthy Generative Models},
  author={...},
  year={2025},
  journal={arXiv preprint}
}
```

---

<p align="center">
  <sub>Maintained by S3IC-Lab · 2025</sub>
</p>