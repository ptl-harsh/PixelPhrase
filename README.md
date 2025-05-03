# PixelPhrase

## Image Captioning with Transformer and EfficientNet

## Overview

This project implements an image captioning pipeline using a pre-trained EfficientNet-B0 backbone combined with custom Transformer encoder–decoder blocks. It generates natural-language descriptions for images in the Flickr8k dataset.

---

## Features

- EfficientNet-B0 (frozen) for robust image feature extraction
- Custom TransformerEncoderBlock with LayerNorm → Dense(ReLU) → MultiHeadAttention → residual connections
- Custom TransformerDecoderBlock with positional + token embeddings, causal self-attention, cross-attention, and feed-forward residuals
- `ImageCaptioningModel` orchestrating multi-caption training and evaluation
- Text standardization and vectorization via Keras `TextVectorization`
- Data augmentation (random flip, rotation, contrast)
- BLEU score evaluation with smoothing
- Early stopping and warmup learning-rate scheduler
- Plotting utilities for dataset distribution and training metrics

---

## Dependencies / Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x (includes Keras)
- NumPy
- Matplotlib
- NLTK (for BLEU score)
- wget, unzip (or similar download tools)

Install required Python packages with:

```bash
pip install tensorflow numpy matplotlib nltk
```

---

## Dataset

The Flickr8k dataset, which consists of images and captions:

```
Flicker8k_Dataset/       #  Flickr8k images
└── Flickr8k.token.txt    # Captions file
```

---

## Model Architecture

1. **EfficientNetB0** (frozen) extracts spatial feature maps from input images.
2. **TransformerEncoderBlock**:
   - LayerNormalization → Dense(ReLU)
   - MultiHeadAttention
   - Residual connection
3. **TransformerDecoderBlock**:
   - Token + positional embeddings
   - Causal self-attention → cross-attention → feed-forward → residuals
4. **ImageCaptioningModel**:
   - Integrates CNN features, encoder, decoder
   - Implements custom training (`train_step`) and testing (`test_step`) loops for handling multiple captions per image

---

### Attention Mechanisms

This model leverages two key attention types within the Transformer blocks:

#### Multi-Head Self-Attention (in both encoder and decoder)

- Allows each position in the input sequence (tokens or spatial features) to attend to all other positions.
- Calculates attention weights by projecting inputs into queries, keys, and values, then computing scaled dot-product attention across multiple heads.
- Heads learn diverse representation subspaces, enhancing model capacity.

#### Encoder–Decoder Cross-Attention (in decoder only)

- Enables the decoder to focus on relevant parts of the image representation when generating each output token.
- The decoder’s queries come from the previous layer’s output, while keys and values come from the encoder’s output.
- Guides token generation using visual context.

---

## Notebook Walkthrough

The Jupyter notebook (`notebooks/train.ipynb`) demonstrates the end-to-end process:

1. Set `KERAS_BACKEND` to `tensorflow`
2. Download and extract the Flickr8k dataset
3. Load and preprocess captions and images
4. Split data into train/val/test sets (70%/15%/15%)
5. Build `tf.data.Dataset` pipelines with augmentation and batching
6. Define CNN, Transformer blocks, and the captioning model
7. Compile with a warmup learning-rate schedule and early stopping
8. Train the model and plot loss/accuracy per epoch
9. Evaluate performance with BLEU scores

---

## Training Instructions

Run the training script:

```bash
python src/train.py
```

Key hyperparameters:

- **Image size:** 299 × 299
- **Vocabulary size:** 10,000 tokens
- **Sequence length:** 25 tokens
- **Embedding dimension:** 512
- **Feed-forward dimension:** 512
- **Batch size:** 64
- **Epochs:** 30
- **Warmup steps:** total_training_steps / 15
- **Optimizer:** Adam with custom `LearningRateSchedule`

**Callbacks:**

- `EarlyStopping` (`patience=3`, `monitor='val_loss'`)
- Custom `PlotCallback` for live metric visualization

---

## Evaluation

- Compute BLEU scores on validation and test splits
- Generate stochastic captions and average results across multiple runs
- Plot line charts and histograms of BLEU score distributions

---

## Results

After training, you should obtain:

- Loss and accuracy curves for training vs. validation per epoch
- BLEU scores across multiple caption generations for sample images
- Distribution statistics (mean, min, max, std) of BLEU scores over 100 validation images
- Saved plots and metrics (optionally stored under `outputs/`)

---
