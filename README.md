# FinVERIFY: Multi-Aspect Retrieval-Augmented Financial Fact-Checking

Financial fact-checking system using MAINRAG's multi-aspect retrieval framework.

## Team
- Shashank Dugad (sd5957)
- Utkarsh Arora (ua2152)
- Shivam Balikondwar (ssb10002)
- Surbhi (xs2682)

## Project Structure
```
finverify/
├── data/          # Datasets and indexes
├── src/           # Source code
├── scripts/       # Executable scripts
├── models/        # Model checkpoints
├── outputs/       # Results and logs
└── demo/          # Gradio demo
```

## Setup
```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/build_index.py
```

## Usage
```bash
python scripts/run_evaluation.py
```

## Paper
Based on "Multi-Aspect Integration for Enhanced RAG" (Wang et al., 2025)
