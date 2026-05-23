# On-Device Adaptive Mathematics Tutoring Engine

An end-to-end, hardware-efficient pipeline designed to train and deploy a lightweight, cloud-independent AI math tutor for primary education (2nd and 3rd grade). Inspired by the **Kumon method**, this platform delivers structured, high-frequency, and progressively difficult arithmetic practice on edge devices without relying on external commercial cloud APIs.

## 🌟 Key Features

- **Automated Synthetic Data Generation Pipeline:** Structured generation of balanced training examples across 4 progressive math domains: addition/subtraction, multiplication, division, and mixed operations.
- **Zero-Hallucination Programmatic Validation:** Multi-stage validation framework running custom execution loops and regex filters to eliminate hallucinated targets and secure exact ground-truth tokens before training[cite: 1, 2].
- **Optimized PEFT Training Configuration:** Fine-tuned via Unsloth + LoRA on consumer-tier cloud hardware to optimize base-model execution and accelerate training throughput.
- **4-Bit Local Quantization:** Weights are merged and compiled into highly quantized GGUF binaries (`Q4_K_M`) for lightning-fast, zero-latency inference on lower-spec local consumer platforms.
- **Interactive Student-Centric Interface:** A full-stack prototype application delivering structured progression tracking, adaptive difficulty scaling, and dynamic hint generation driven completely natively by the edge model[cite: 5].

---

## 🛠️ Architecture & Pipeline Workflows

### 1. Synthetic Data Engineering & Clean Room Validation
To build a highly accurate domain-specific model, the pipeline generates an exhaustive, specialized dataset formatted to the Alpaca instruction style[cite: 1, 4].
- **Curriculum Structuring:** Math problems are generated progressively across 4 incremental difficulty tiers to mirror a child's natural learning curve[cite: 1, 4].
- **Algorithmic Filtering:** To ensure completely safe and accurate educational outputs, the pipeline strips fractional remainders, filters out negative constraints, and rigorously sanitizes incoming generation batches using algorithmic validation loops[cite: 1, 2].

### 2. Parameter-Efficient Fine-Tuning (PEFT)
The training phase aligns a lightweight **Llama 3.2 1B Instruct** base model with specialized mathematical instruction targets[cite: 5].
- **Framework Optimization:** Leveraging **Unsloth** and **LoRA** configuration mappings, training targets are constrained strictly to performance-critical attention projection matrices[cite: 5].
- **Training Parameters:** Configured with an effective batch size of 16 through gradient accumulation steps, 8-bit optimization primitives, and cross-entropy sequence checking—driving training convergence gracefully down to a validation loss of $\sim$0.135[cite: 5].

### 3. Edge Compilation & Deployment Runtime
For real-world on-device deployments, reducing inference latency and local physical footprints is paramount.
- **Weights Fusion & Quantization:** Model adapters are seamlessly compiled and consolidated via `llama.cpp` hooks directly into **4-bit Q4_K_M GGUF format**[cite: 5].
- **Deterministic Steering:** System prompt architectures inside custom Modelfiles lock the edge model's generation scope to output clean, numeric evaluations and contextual hints with near-zero generation temperatures[cite: 4, 5].

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch (with CUDA support for training)
- Ollama (installed locally for edge inference support)

### 1. Setup & Installation
Clone the repository and install the standard dependencies:
```bash
git clone [https://github.com/sambitkarmakar03/your-repository-name.git](https://github.com/sambitkarmakar03/your-repository-name.git)
cd your-repository-name
pip install -r requirements.txt



# Convert to GGUF format and load the model into local orchestration layers
ollama create math-tutor -f Modelfile

streamlit run app.py
