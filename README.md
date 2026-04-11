# Induction Heads on TinyStories

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fanat503/Induction-Heads-Tinystories/blob/main/induction_heads_tinystories.ipynb)

Training GPT-2 Small from scratch and tracking the formation of 
Induction Heads throughout training. Verified with a custom 
Sparse Autoencoder on Layer 6 activations.

**Key finding:** Previous Token Heads form reliably on any dataset. 
Induction Heads are dataset-dependent — on TinyStories they barely 
form at all (max score 0.05), because the dataset is too simple 
to require in-context learning.

## Results

| Mechanism | Head | Score | Notes |
|---|---|---|---|
| Previous Token | L0H3 | 0.20 | Stable from step 14K |
| Induction | L6H4 | 0.05 | Peak at 16K, slow decay |

**SAE analysis (Layer 6):**
- Active features: 20-35 per token
- Dead features: ~3% (228/8192)
- No clean Induction features found
- F#8107 → syntactically predictable context
- F#635 → general contextual pattern

## Why TinyStories?

To isolate the effect of dataset complexity on mechanistic formation.
TinyStories is intentionally simple — short texts, basic grammar, 
low repetition. The hypothesis was that Induction Heads would 
struggle to form. It was confirmed.

Next experiment: TinyStories vs OpenWebText across model sizes 
(10M → 350M) on TPU. To verify that dataset complexity is the 
key factor.

## Setup

```bash
git clone https://github.com/yourusername/induction-heads-tinystories
cd induction-heads-tinystories
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

## Compute induction scores on checkpoints

```bash
python induction_heads.py
```

## Train SAE on Layer 6

```bash
python sae.py
```

## Architecture

**GPT-2 Small** — 124M parameters, implemented from scratch in PyTorch
- 12 layers, 12 heads, 768 dim
- Gradient accumulation, fp16
- Trained for 20,000 steps on TinyStories (473M tokens, ~5.5 epochs)

**SAE** — trained on Layer 6 residual stream activations
- Encoder: 768 → 8192
- Decoder: 8192 → 768
- L1 coefficient: 3.9
- 25 epochs

## Structure

```
gpt2.py             — model architecture
train.py            — training script
induction_heads.py  — compute_induction_score, compute_previous_token_score
sae.py              — SAE architecture + training
```

## Dataset

TinyStories — https://huggingface.co/datasets/roneneldan/TinyStories

## References

- Olsson et al. — [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895) (2022)
- Elhage et al. — [Toy Models of Superposition](https://arxiv.org/abs/2209.11405) (2022)
- Cunningham et al. — [Sparse Autoencoders](https://arxiv.org/abs/2309.08600) (2023)
