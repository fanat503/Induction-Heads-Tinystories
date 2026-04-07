# Induction Heads on TinyStories

GPT-2 Small (124M parameters) trained from scratch on TinyStories (473M tokens).
Experiment: tracking Induction Heads formation dynamics during training + 
verification via Sparse Autoencoder.

## Results

- Previous Token Heads: L0H3, score 0.20 — stable formation in layers 0-4
- Induction Heads: L6H4, score 0.05 — weak, dataset-dependent
- SAE (Layer 6): no clean Induction features found

## Structure

gpt2.py — model architecture  
train.py — training script  
induction_heads.py — compute_induction_score, compute_previous_token_score  
sae.py — SAE architecture, train_sae

## Dataset

TinyStories — https://huggingface.co/datasets/roneneldan/TinyStories

## Article

[Habr — link after publication]
