# 🕌 As-Sunnah Foundation — Voice-Enabled RAG Assistant

> **AI Engineer Assessment Task** · Retrieval-Augmented Generation · Voice I/O · LSTM Memory · Confidence Scoring

[![Platform](https://img.shields.io/badge/Platform-Kaggle%20Notebook-20BEFF?logo=kaggle)](https://kaggle.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://python.org)
[![Gradio](https://img.shields.io/badge/Interface-Gradio-FF7C00?logo=gradio)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green)](#)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [Features](#-features)
4. [Setup Instructions](#-setup-instructions)
5. [How to Run](#-how-to-run)
6. [Sample Queries & Outputs](#-sample-queries--outputs)
7. [Confidence Scoring Explained](#-confidence-scoring-explained)
8. [Assumptions & Limitations](#-assumptions--limitations)
9. [Project Structure](#-project-structure)

---

## 🎯 Project Overview

A **fully open-source**, voice-enabled RAG (Retrieval-Augmented Generation) assistant built for the As-Sunnah Foundation knowledge base. The system:

- Accepts **voice or text** input
- Retrieves relevant knowledge using **hybrid search** (semantic + keyword)
- Generates **grounded answers** using Gemma 2B-IT LLM
- Outputs both **text and audio** responses
- Maintains **conversation context** using an LSTM memory module
- Provides **confidence scoring** with hallucination risk indicators

**All models are free and open-source. Runs entirely on Kaggle's free GPU tier.**

---

## 🏗 Architecture

![RAG Architecture Diagram](docs/architecture.svg)

### Pipeline Flow

```
Voice/Text Input
       │
       ▼
[Whisper STT]  ←── Audio only
       │
       ▼
[LSTM Memory] ──► Query Expansion (follow-up detection)
       │
       ▼
┌─────────────────────────────────────┐
│         HYBRID RETRIEVAL            │
│  FAISS (semantic, α=0.6)            │
│    +  BM25 (keyword,  α=0.4)        │
│    → Score merge → top-12           │
│    → CrossEncoder rerank → top-6    │
└─────────────────────────────────────┘
       │
       ▼
[Context Builder]  ← original_chunk metadata
       │
       ▼
[Prompt Builder]   ← anti-hallucination system prompt
       │              + LSTM history
       ▼
[Gemma 2B-IT]      ← generates answer
       │
       ▼
[Confidence Scorer] → Faithfulness · Precision · Hallucination Risk
       │
       ├──► Text Answer + Sources (Gradio Markdown)
       └──► Audio Answer (gTTS → MP3 autoplay)
```

### Key Components

| Component | Model / Library | Purpose |
|-----------|----------------|---------|
| **STT** | `whisper base.en` | Voice → Text (English) |
| **Embeddings** | `intfloat/multilingual-e5-base` | Semantic search vectors |
| **Vector Store** | `FAISS (faiss-cpu)` | Fast approximate nearest-neighbour |
| **Keyword Search** | `BM25Okapi (rank_bm25)` | Exact keyword matching |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder reranking |
| **LLM** | `google/gemma-2-2b-it` | Answer generation |
| **Memory** | Custom LSTM (PyTorch) | Conversation context |
| **TTS** | `gTTS` | Text → Audio output |
| **UI** | `Gradio 4.x` | Web interface |

---

## ✨ Features

### Core (Mandatory)
- ✅ **RAG System** — semantic chunking, hybrid retrieval, grounded generation
- ✅ **Voice Input** — Whisper STT, mic recording + file upload
- ✅ **Grounded Answering** — answers only from retrieved context, source references shown
- ✅ **Gradio Interface** — clean, usable web UI
- ✅ **Fail-safe** — returns `"I don't know based on the provided data"` when answer not found

### Bonus
- ✅ **Text-to-Speech** — gTTS English audio, autoplay
- ✅ **Conversation History** — LSTM sliding-window memory (last 6 turns)
- ✅ **Confidence Scoring** — faithfulness, precision, retrieval confidence, hallucination risk

---

## ⚙️ Setup Instructions

### Prerequisites
- Kaggle account (free)
- GPU accelerator: T4 × 1 (Settings → Accelerator)
- HuggingFace account with `HF_TOKEN` added to Kaggle Secrets

### Step 1 — Clone / Upload the Notebook

Upload all `.py` parts (Part 1–12) as a **single Kaggle Notebook**, or import from this repo.

### Step 2 — Add Dataset

Upload `as_sunnah_foundation_extract.txt` to Kaggle as a dataset named `maindataset`, so the path resolves to:
```
/kaggle/input/datasets/mdfaishalahmedrudroo/maindataset/as_sunnah_foundation_extract.txt
```

### Step 3 — Add HuggingFace Token

In Kaggle Notebook → Add-ons → Secrets → add secret named `HF_TOKEN` with your HuggingFace token.

> Required to download `google/gemma-2-2b-it` (gated model — accept terms on HF first).

### Step 4 — Install Dependencies

Part 1 of the notebook handles all installs:

```bash
pip install gradio --upgrade
pip install langchain langchain-community langchain-core \
            langchain-text-splitters langchain-huggingface
pip install faiss-cpu sentence-transformers
pip install "transformers>=4.38.0" accelerate
pip install rank_bm25 rouge-score numpy pandas
pip install openai-whisper gTTS
```

Or from the requirements file:
```bash
pip install -r requirements.txt
```

### Step 5 — Accept Gemma License

Go to https://huggingface.co/google/gemma-2-2b-it and accept the model license with your HuggingFace account before running.

---

## 🚀 How to Run

Run all cells **in order** (Part 1 → Part 12). Each part builds on the previous:

| Part | What it does |
|------|-------------|
| Part 1 | Install packages, load secrets |
| Part 2 | Load dataset, semantic chunking |
| Part 3 | Build FAISS + BM25 + CrossEncoder retrieval |
| Part 4 | Load Gemma 2B-IT |
| Part 5 | Load Whisper STT |
| Part 6 | Initialize LSTM memory |
| Part 7 | Confidence scoring functions |
| Part 8 | RAG chain + generate_answer() |
| Part 9 | Response formatter |
| Part 10 | gTTS text-to-speech |
| Part 11 | Main process_query() pipeline |
| Part 12 | Launch Gradio UI (`share=True`) |

After Part 12 runs, Gradio will print a **public share URL** — open it in any browser.

---

## 🧪 Sample Queries & Outputs

### Query 1 — Factual (High Confidence)
**Input:** `"What are the core principles and values of As-Sunnah Foundation?"`

**Output:**
```
The core principles and values of As-Sunnah Foundation are guided by the 
Qur'an and Sunnah, emphasize unity and brotherhood, promote good deeds and 
discourage wrongdoing, avoid extremism, and emphasize compassion and welfare 
for all people regardless of religion or race. They also maintain 
transparency through proper accounting and auditing when using donations.
```
| Metric | Score |
|--------|-------|
| Overall | 🟢 0.8725 (High) |
| Faithfulness | 1.0000 |
| Hallucination Risk | 0.0000 |

---

### Query 2 — Statistics (High Confidence)
**Input:** `"Details about Dawah initiatives, number of videos and live sessions"`

**Output:**
```
The foundation produces over 256 da'wah videos which have reached more than 
30 million viewers. The foundation offers weekly live shariah Q&A with 267 episodes.
```
| Metric | Score |
|--------|-------|
| Overall | 🟢 0.8027 (High) |
| Faithfulness | 1.0000 |
| Hallucination Risk | 0.0000 |

---

### Query 3 — Follow-up / Conversation Memory
**Turn 1:** `"Founder of As-Sunnah Foundation."` → *"Shaikh Ahmadullah is the founder."*

**Turn 2:** `"When was it founded?"` → Query expanded to: *"Founder of As-Sunnah Foundation. — When was it founded?"*

> Note: LSTM memory correctly identifies this as a follow-up and expands the query. However, if the founding year is not present in the dataset, confidence will be Very Low — indicating a potential hallucination.

---

### Query 4 — Education Goals (High Confidence)
**Input:** `"What are the education-related goals of As-Sunnah Foundation?"`

**Output:**
```
• Spreading pure Islamic knowledge
• Developing qualified scholars
• Establishing research centres and institutions based on Qur'an and Sunnah
• Operating schools and universities
• Training 100,000 unemployed youths annually by 2030
• Empowering poor women through technical training
• Providing scholarships to meritorious students
```
| Metric | Score |
|--------|-------|
| Overall | 🟢 0.8430 (High) |
| Faithfulness | 0.8750 |
| Hallucination Risk | 0.1250 |

---

### Query 5 — Fail-safe Test
**Input:** `"What is the price of rice in Bangladesh?"`

**Output:** `"I don't know based on the provided data."`

| Metric | Score |
|--------|-------|
| Is Fallback | ✅ True |
| Hallucination Risk | 0.0 |

---

### Query 6 — Self-Reliance Project (High Confidence)
**Input:** `"What is self-reliance project and who benefits from it?"`

**Output:**
```
The Self-Reliance Project provides income-generating tools and training to 
capable poor individuals to ensure sustainable self-reliance. Beneficiaries 
are selected through a screening process and receive profession-appropriate 
tools (sewing machines, vans, shop items, agricultural equipment) along with 
religious and moral training. Between 2021 and 2024, 5,140 families benefited.
```
| Metric | Score |
|--------|-------|
| Overall | 🟢 0.8132 (High) |
| Faithfulness | 1.0000 |
| Precision | 0.8723 |
| Hallucination Risk | 0.0000 |

---

## 📊 Confidence Scoring Explained

Each answer gets a 5-metric confidence dashboard:

```
Overall = 0.45 × Faithfulness + 0.30 × Precision + 0.25 × Retrieval Confidence
```

| Metric | What it measures |
|--------|-----------------|
| **Faithfulness** | % of answer sentences supported by retrieved context |
| **Precision** | Token overlap between answer and context |
| **Retrieval Confidence** | Sigmoid-normalised CrossEncoder scores |
| **Hallucination Risk** | `1 - Faithfulness` |
| **Overall** | Weighted combination above |

| Score Range | Label |
|-------------|-------|
| ≥ 0.75 | 🟢 High |
| 0.50–0.74 | 🟡 Medium |
| 0.25–0.49 | 🟠 Low |
| < 0.25 | 🔴 Very Low |

---

## ⚠️ Assumptions & Limitations

### Assumptions
- The dataset (`as_sunnah_foundation_extract.txt`) is a clean UTF-8 text extraction from the As-Sunnah Foundation website
- All questions are asked in **English** (Whisper base.en is English-only)
- Kaggle T4 GPU is available; CPU fallback works but is significantly slower
- The user has accepted the Gemma 2B-IT license on HuggingFace

### Known Limitations

| Limitation | Details |
|------------|---------|
| **English only** | Whisper base.en and Gemma 2B-IT are optimised for English. Bangla input will give poor results |
| **Founding year hallucination** | If the dataset does not explicitly state the founding year, Gemma may infer "2017" — confidence score will be Very Low, indicating this risk |
| **Broad queries** | Very general questions (e.g. "tell me everything about disaster relief") retrieve lower-scoring chunks — answer quality degrades |
| **No persistent memory** | LSTM memory is session-only; cleared when notebook restarts |
| **Audio length** | Whisper input limited to 60 seconds; very long audio is rejected |
| **TTS quality** | gTTS is cloud-based (requires internet); audio quality is basic compared to neural TTS |
| **Model size** | Gemma 2B-IT loads ~5GB GPU RAM; may OOM on smaller GPUs |
| **No Bangla support** | Bangla language is listed as a bonus but not implemented in this version |

### Design Decisions
- **Summary page skipped** during chunking — summary sections caused faithfulness to drop to 0 because they're too generic and paraphrase other sections
- **`original_chunk` metadata** stored separately from e5-prefixed text — ensures LLM never sees "passage: " prefix in its context
- **CrossEncoder on clean text** — reranking done on original text, not on e5-prefixed versions, for accurate relevance scoring
- **LSTM memory** uses exponential time-decay so older turns have less influence on query expansion

---

## 📁 Project Structure

rag-assistant/
│
├── README.md                    ← This file
├── requirements.txt             ← All Python dependencies
├── .gitignore
│
├── docs/
│   └── architecture.svg        ← System architecture diagram
│
├── screenshots/
│   ├── gradio_ui.png            ← Main interface screenshot
│   ├── high_confidence.png      ← Sample high-confidence answer
│   ├── fallback_response.png    ← Fail-safe behaviour
│   └── confidence_dashboard.png ← Confidence metrics panel
│
└── notebook/
    └── as_sunnah_rag.ipynb      ← Full Kaggle Notebook (Parts 1–12)


---

*Built for the As-Sunnah Foundation AI Engineer Assessment. All models are free and open-source.*
