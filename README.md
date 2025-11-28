# LLM-Agents-for-Non-Equilibrium

This repository contains the code used in our paper to simulate a competitive market with:
- A fully **equation-based baseline** (rule-based buyers and sellers),
- **LLM-driven sellers** with rule-based buyers, and
- **LLM-driven buyers** with rule-based sellers.

The goal is to study how LLM-driven agents, endowed with boundedly rational and impression-based decision processes, can reproduce non-equilibrium market phenomena such as alternating monopolies and phase transitions.

The code is written for clarity and reproducibility so that reviewers can easily rerun the main experiments.

---

## 1. Environment Setup

### 1.1 Python version
We recommend:

- **Python ≥ 3.9**

### 1.2 Install dependencies

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate      # on Linux / macOS
# venv\Scripts\activate     # on Windows
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. OpenAI / LLM Configuration

### 2.1 API Key
Set environment variable:

```bash
export OPENAI_API_KEY="sk-..."
# Windows PowerShell:
# setx OPENAI_API_KEY "sk-..."
```

### 2.2 Base URL
Default:

```
https://api.openai.com/v1
```

Override if using local LLM:

```bash
--base_url http://your-host:port/v1
```

---

## 3. Repository Structure

```
.
├── equation_based.py
├── equation_based.sh
├── LLM_seller.py
├── LLM_buyer.py 
├── LLM_seller.sh
├── LLM_buyer.sh
├── requirements.txt
└── README.md
```

Outputs:

```
exp*/      # experiment results
logs/      # log files
```

---

## 4. Running the Experiments

Before running:

```bash
mkdir -p logs
```

---

### 4.1 Baseline: Equation-Based Market

```bash
python equation_based.py --J 0.4 --rho 0.8 --Nt 2000
```

---

### 4.2 LLM-Driven Sellers

```bash
python LLM_seller.py --model gpt-4o --J 0.4 --rho 0.8 --Nt 400 --fig 1
```

---

### 4.3 LLM-Driven Buyers

```bash
python LLM_buyer.py --model gpt-4o --J 0.4 --rho 0.8 --ratio 0.5 --Nt 400 --fig 1
```

---

## 5. Reproducibility Notes

- Baseline uses fixed seed.
- LLM experiments include randomness but macro patterns are stable.

---

## 6. Contact

Please contact the authors if you need assistance.

