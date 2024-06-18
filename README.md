# Fine-Tuning GPT-2 with WikiText for Enhanced Causal Language Modeling

This project focuses on fine-tuning the GPT-2 language model with the WikiText dataset for enhanced causal language modeling. Additionally, it explores unstructured pruning techniques to optimize the model by reducing parameters, thus speeding up inference while maintaining performance and coherence.

## Project Overview

- **Fine-Tuning:** Leverage the comprehensive and diverse textual data from the WikiText dataset to enhance GPT-2's text generation capabilities and improve its understanding of natural language.
- **Model Optimization:** Implement unstructured pruning to experiment with 20-50% parameter reduction, aiming to speed up inference and maintain model performance.

## File Structure

- `main.py`: The main script for fine-tuning GPT-2 with the WikiText dataset and applying unstructured pruning techniques.

## Requirements

To run this project, you'll need the following dependencies:

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- WikiText dataset
- NumPy
- Pandas

You can install the required packages using:

```bash
pip install torch transformers numpy pandas
