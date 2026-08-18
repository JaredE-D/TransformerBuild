# Transformer From Scratch

A from-scratch PyTorch implementation of the original Transformer architecture
("Attention Is All You Need", https://arxiv.org/pdf/1706.03762), built to
understand the internals rather than to reuse `nn.Transformer` or a library
implementation. Notes on intent and reasoning are written alongside the code
in [Transformer.ipynb](Transformer.ipynb).

## What's implemented

- Sinusoidal positional encoding + scaled token embedding
- Multi-head attention (single-head scaled dot-product attention composed
  into multi-head), built from raw `nn.Linear` projections — no
  `nn.MultiheadAttention`
- Pre-norm residual blocks (`LayerNorm` before the sublayer, not after, as in
  the original paper's Figure 1)
- Position-wise feed-forward network (`d_model` → `4*d_model` → `d_model`)
- Full Encoder / Decoder stacks with source padding masks, target causal
  masks, and target padding masks
- Adam optimizer with the Noam warmup/decay learning-rate schedule
  (AIAYN §5.3): `lr = d_model^-0.5 · min(step^-0.5, step · warmup^-1.5)`
- Checkpointing (model/optimizer/scheduler state) and resume-from-checkpoint
- Greedy autoregressive decoding for inference

## Task and data

Trained on **English→German translation** using the
[Multi30k](https://huggingface.co/datasets/bentrevett/multi30k) dataset
(29,000 train / 1,014 validation sentence pairs), tokenized with GPT-2 BPE
via `tiktoken` plus three appended special tokens (`PAD`, `BOS`, `EOS`;
vocab size 50,260).

A toy reversal-sequence config (`cfg_quick`) is also included in the
notebook as a fast (~1 min) pipeline sanity check before running the full
translation training.

## Final training configuration

| Param | Value |
|---|---|
| `d_model` | 256 |
| heads (`h`) | 8 |
| encoder/decoder layers | 6 |
| dropout | 0.2 |
| batch size | 32 |
| warmup steps | 4000 |
| planned epochs | 100 |
| parameters | ~36.9M |

## Results

Training was run for 87/100 planned epochs (mixed-precision, on GPU).
Validation loss bottomed out early and the model then overfit:

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 4.38 | 2.55 |
| 8 | 1.29 | 1.37 |
| **19** | **0.75** | **1.23 (best)** |
| 40 | 0.39 | 1.41 |
| 87 (last run) | 0.16 | 1.71 |

**Takeaway:** with this config, generalization peaks around epoch ~19-20;
everything after that is the model memorizing the 29k-sentence training set
(train loss → ~0.16 while val loss climbs back to ~1.7). The checkpoint
worth using for inference is the epoch ~19-20 one, not the final one.

## Quantitative result

Greedy decoding with the `epoch_039.pt` checkpoint (epoch 40 — past the
val-loss minimum, see above) against the full 1,000-sentence Multi30k test
split:

**BLEU: 26.34** (`59.9/32.2/20.0/12.5` 1/2/3/4-gram precisions, brevity
penalty 1.000), computed with `sacrebleu`.

Sample translations (English source → German reference → model output):

| EN | DE reference | DE predicted |
|---|---|---|
| A man in an orange hat starring at something. | Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt. | Ein Mann mit orangefarbenem Hut starrt auf etwas. |
| People are fixing the roof of a house. | Leute Reparieren das Dach eines Hauses. | Leute reparieren das Dach eines Hauses. |
| A group of people standing in front of an igloo. | Eine Gruppe von Menschen steht vor einem Iglu. | Eine Gruppe von Personen steht vor einem Innenraum. |
| *(custom, not in dataset)* A dog runs across the field. | — | Ein Hund rennt über das Feld. |

Shorter/simpler sentences translate cleanly; longer sentences with multiple
clauses keep the right vocabulary but lose some structure 

## Checkpoints

Checkpoints are no longer committed to git 
- `checkpoints/translation_layers6_drop0.2/epoch_039.pt` — full run (model +
  optimizer + scheduler state) matching the config above, saved at epoch 40.
  Used for the BLEU result above.
- `checkpoints/epoch_043.pt` — an earlier training run from a prior config
  iteration, kept for reference only; it does not match the current model
  config and should not be loaded with it.

The training loop now tracks the best validation loss and saves it to
`best.pt` (instead of every epoch), and stops automatically after 10 epochs
with no improvement —

## Recent improvements

- **Early stopping + best-checkpoint saving**: the training loop tracks
  best validation loss and saves only `best.pt` on improvement, stopping
  after `patience=10` epochs with no improvement — addresses the
  overfitting shown in Results without wasting compute on epochs 20-100.
- **Inference loads `best.pt`** when present, falling back to the latest
  per-epoch checkpoint otherwise.
- **BLEU evaluation cell** added, computing corpus BLEU over the full
  Multi30k test split with `sacrebleu`.


## Known issues / next steps

- BLEU is measured with greedy decoding only; beam search would likely add
  a couple of points.
- The GPT-2 BPE tokenizer wasn't designed for German — a German-aware or
  joint-vocabulary tokenizer would likely improve results.

## Repo structure

```
Transformer.ipynb        # implementation, training, and inference, with notes
requirements.txt         # pinned environment
checkpoints/              # saved model/optimizer/scheduler states
```
