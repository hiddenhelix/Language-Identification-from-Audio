# Spoken Language Identification for 22 Indian Languages

Fine-tuning `facebook/mms-300m` with audio augmentation and adversarial debiasing (DANN) to identify spoken language across India's 22 constitutionally recognized languages, while explicitly suppressing speaker-identity shortcuts.

> Full writeup: [`Project_report_NNTI__MAIN_.pdf`](./Project_report_NNTI__MAIN_.pdf) · Task spec: [`NNTI_Term_Description.pdf`](./NNTI_Term_Description.pdf)

## Problem

Spoken language ID (LID) across India's 22 languages is hard for two reasons: many languages are acoustically close (especially within the Indo-Aryan family), and the available data is speaker-sparse — the [`badrex/nnti-dataset-full`](https://huggingface.co/datasets/badrex/nnti-dataset-full) dataset has only 400 samples per language from just **5 speakers per language** (110 speakers total). With that little speaker diversity, a model can easily learn to recognize *voices* instead of *languages* — we call this **speaker bias**.

## Approach

Two complementary debiasing strategies on top of a fine-tuned `MMS-300m` encoder:

1. **Audio augmentation** (data-centric) — 70% of training samples get randomized pitch shift (±4 semitones) and speed perturbation (0.85×–1.15×) via `torchaudio`, artificially expanding speaker/acoustic diversity while preserving linguistic content.
2. **Domain-Adversarial Neural Network (DANN)** (model-centric) — a gradient reversal layer sits between the shared encoder and a speaker-classification head. The encoder is trained to make speaker identity *undecodable* while a linear head keeps it language-discriminative. Reversal strength ramps up over training on a sigmoid schedule.

Both are evaluated together and separately against a tuned baseline.

## Results

| System | Accuracy | Macro F1 |
|---|---|---|
| Chance (random) | 4.54% | – |
| Baseline (no aug, no DANN) | 33.39% | 0.321 |
| Augmentation only | 37.58% | 0.370 |
| **DANN + Augmentation** | **41.09%** | **0.400** |

DANN + Augmentation improves **+7.7 points** over baseline. Embedding analysis (t-SNE + 5-fold linear probing) shows the two methods work for different reasons:
- **Augmentation** directly destroys speaker-identifying acoustic variation at the input (lowest speaker-probe score).
- **DANN** pushes the encoder toward richer, more language-discriminative representations (highest language-probe score), though it doesn't by itself minimize speaker information in the shared encoder as directly as augmentation does.

Gains are concentrated in acoustically distinctive languages (Kashmiri, Manipuri, Assamese). Closely related Indo-Aryan pairs (e.g. Hindi/Urdu) remain hard to separate, and DANN can even worsen confusion there — suppressing speaker cues sometimes strips out subtle phonetic signal needed to tell them apart. Full per-language breakdown, confusion matrices, and t-SNE plots are in the report.

## Repo contents

| File | Description |
|---|---|
| `train_model.py` | End-to-end training/eval script (cell-marked with `#%%` for notebook conversion) |
| `Project_report_NNTI__MAIN_.pdf` | Full write-up: methodology, results, embedding analysis |
| `NNTI_Term_Description.pdf` | Original course task specification |

## Method details

- **Backbone**: `facebook/mms-300m`. Also swappable for `w2v-bert-2.0`, `mHuBERT-147`, `wav2vec2-xls-r-300m`.
- **Classifier head**: last 4 transformer layers pooled into a 1024-dim embedding, fed to (a) a linear language classifier and (b) a 2-layer MLP speaker adversary (1024→256→110) behind a gradient reversal layer.
- **Training**: batch size 8, gradient accumulation ×4 (effective batch 32), 15 epochs, LR 5e-5 with cosine decay, weight decay 0.01, dropout 0.1, bf16 precision, early stopping (patience 7), max audio length 7s.
- **Evaluation**: classification report + confusion matrices; t-SNE (perplexity 30) on utterance-level embeddings, colored by language and by speaker; 5-fold cross-validated logistic-regression probes to quantify residual speaker-identity vs. language-identity information in the embeddings.
- Logged via Weights & Biases.

## Running it

The script is written as a cell-marked (`#%%`) script meant for Google Colab.

```bash
# 1. Convert to a notebook
pip install jupytext
jupytext --to notebook train_model.py -o train_model.ipynb

# 2. Upload train_model.ipynb to Colab

# 3. Install the one extra dependency (rest is preinstalled on Colab)
!pip install evaluate

# 4. Run all cells
```

Set your `HF_TOKEN` and `WANDB_API_KEY` as environment variables before running (do **not** hardcode them). Results are logged to your W&B project by default; alternatively, copy artifacts off the Colab runtime disk before the session ends.

## Limitations

- Only 5 speakers per language — stochastic augmentation can't fully substitute for genuine speaker diversity.
- Constrained to ≤600M-parameter models; larger architectures (e.g. `w2v-BERT-2.0`) might suppress speaker bias more naturally without explicit adversarial training — untested here.
- Results are on a single validation split; cross-validating over multiple speaker splits would give more robust estimates.
- Not intended for deployment: language misclassification could disproportionately affect speakers of underrepresented languages.

## References

- Ganin & Lempitsky (2014). *Unsupervised Domain Adaptation by Backpropagation.* [arXiv:1409.7495](https://arxiv.org/abs/1409.7495)
- Elazar & Goldberg (2018). *Adversarial Removal of Demographic Attributes from Text Data.* [arXiv:1808.06640](https://arxiv.org/abs/1808.06640)
- Meta AI. [`facebook/mms-300m`](https://huggingface.co/facebook/mms-300m)
- [`badrex/nnti-dataset-full`](https://huggingface.co/datasets/badrex/nnti-dataset-full)
