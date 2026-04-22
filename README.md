# Speech Language Model

A 522 million parameter joint speech-text language model built from scratch. The core idea is simple: treat audio the same way GPT treats text. Instead of building separate systems for speech and language, this model learns both in a single unified vocabulary and a single transformer.

No pretrained speech encoders. No fine-tuning on top of an existing model. Everything from the architecture to the training pipeline was written from scratch.

**Model weights:** [HuggingFace](https://huggingface.co/kirangowda3101/speech-language-model) (5.9GB)
**Code:** [GitHub](https://github.com/kirangowda3101/Speech-Language-Model)

---

## The core idea

Text tokens from GPT-2's BPE tokenizer cover words and subwords. Audio tokens come from Meta's EnCodec neural codec, which compresses raw audio into discrete codes at 75 frames per second using 8 codebooks, producing 600 tokens per second of speech. Both live in the same flat vocabulary. A training sequence looks like this:

```
[BOS] the quick brown fox [AUDIO_START] ...600 audio tokens per second... [AUDIO_END] [EOS]
```

The transformer learns to predict the next token at every position, whether that token is a word or a piece of audio. This gives the model a joint understanding of speech and language in a single forward pass.

---

## Architecture

The transformer follows a GPT-style decoder-only design with two upgrades over the original GPT-2.

Rotary positional encoding (RoPE) replaces learned absolute positions. RoPE encodes position by rotating query and key vectors, which generalizes better to sequence lengths not seen during training compared to fixed embeddings.

SwiGLU replaces the standard ReLU feed-forward block. It uses a gating mechanism where one linear projection controls how much information from another projection passes through. In practice this converges faster and reaches lower loss than ReLU at the same parameter count.

| Setting | Value |
|---|---|
| Parameters | 522M |
| Layers | 24 |
| Attention heads | 16 |
| d_model | 1024 |
| FFN hidden dim | 4096 |
| Context window | 2048 tokens |
| Text vocabulary | 50,257 (GPT-2 BPE) |
| Audio vocabulary | 8,192 (8 codebooks x 1024) |
| Special tokens | 5 (BOS, EOS, PAD, AUDIO_START, AUDIO_END) |
| Total vocabulary | 58,454 |

---

## Training

**Data:** LibriSpeech train-clean-100 (100 hours of read English speech)

Each audio file is preprocessed once before training. EnCodec encodes it to discrete codes, the tokenizer flattens those codes to integers, and the result is saved as a `.npy` file on disk. The training loop loads these files directly so no GPU time is spent on audio decoding during training.

**Infrastructure:** One A100 GPU on Northeastern University's Explorer HPC cluster, managed through SLURM. Jobs were chained automatically so training continued across the cluster's 55-minute time limit without any manual intervention.

**Optimizer:** AdamW with weight decay 0.1, gradient clipping at 1.0. Learning rate follows a cosine decay schedule with 2,000 step linear warmup, starting at 3e-4 and decaying to 3e-5.

**Result:**

| Step | Val Loss |
|---|---|
| 0 | 11.00 |
| 1,000 | 4.90 |
| 10,000 | 3.38 |
| 31,000 | 2.80 |
| 88,000 | 2.13 |
| 100,000 | 3.49 |

A randomly initialized model on this vocabulary would score log(58454) = 10.98. Reaching 3.49 confirms the model is learning real structure in both speech and language.

---

## JAX benchmark

The attention block was re-implemented in JAX/Flax to compare throughput against PyTorch on the same hardware. JAX uses XLA compilation through `jax.jit` which fuses operations and eliminates Python overhead after the first forward pass.

Both frameworks ran on the same L40S GPU.

| Sequence length | PyTorch (ms) | JAX (ms) | JAX speedup |
|---|---|---|---|
| 64 | 0.14 | 0.11 | 1.24x |
| 128 | 0.23 | 0.16 | 1.47x |
| 256 | 0.43 | 0.26 | 1.63x |
| 512 | 0.85 | 1.00 | 0.85x |

JAX is faster for shorter sequences where XLA kernel fusion pays off the most. PyTorch recovers at sequence length 512, likely because cuDNN's attention kernels are more optimized at that size. The crossover is somewhere between 256 and 512 tokens.

---

## Running the demo locally

Clone the repo and download the weights:

```bash
git clone https://github.com/kirangowda3101/Speech-Language-Model
cd Speech-Language-Model

pip install torch torchaudio encodec tiktoken fastapi uvicorn python-multipart scipy

# Download model weights from HuggingFace
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('kirangowda3101/speech-language-model', 'best.pt', local_dir='.')"
```

Run the server:

```bash
python -m uvicorn app:app --reload
```

Open `http://localhost:8000` in your browser. The demo has two modes: type text and hear the model speak it, or record your voice and hear the model respond in its own voice.

Note: inference on CPU takes 30-60 seconds per generation for this model size. Running on a GPU brings that down to 5-15 seconds.

---

## What the output sounds like right now

At val loss 3.49 the output is structured noise, not intelligible speech. The model generates valid EnCodec tokens that decode to real audio, but the tokens do not yet form coherent words. This is the honest state of the model at 100K training steps on 100 hours of data.

For context, well-trained speech synthesis models typically reach val loss below 2.0 and train on 1,000 to 10,000 hours. This is a proof of concept that the architecture and pipeline work end to end.

---

## Scaling up

The compute available for this project was limited to 100 hours of LibriSpeech and 100K training steps on a single A100. The architecture scales directly to much larger datasets.

Training on GigaSpeech XL (10,000 hours of diverse speech including audiobooks, podcasts, and YouTube) would expose the model to far more acoustic variation and speaker diversity. Combined with Open Assistant or a similar instruction-following dataset, the model could learn to respond to spoken questions rather than just continue audio sequences.

With 10x more data and 5x more training steps the val loss would likely drop below 2.0 and the model would start producing recognizable speech. The code, architecture, and training pipeline are all ready to scale. The bottleneck is compute.

---

## Repository structure

```
config.py          vocabulary layout and model hyperparameters
model.py           transformer (RoPE, SwiGLU, causal self-attention)
tokenizer.py       text and audio tokenization
encodec_wrapper.py wrapper around Meta's EnCodec model
audio_utils.py     audio loading, normalization, chunking utilities
dataset.py         PyTorch Dataset for preprocessed token files
dataloader.py      DataLoader construction with DDP support
preprocess.py      one-time preprocessing of LibriSpeech to .npy files
train.py           DDP training loop with checkpointing
checkpoint.py      atomic checkpoint save and load with auto-cleanup
jax_attention.py   attention block re-implemented in JAX
flax_model.py      full model re-implemented in Flax
benchmark.py       PyTorch vs JAX throughput comparison script
app.py             FastAPI inference server
static/index.html  browser demo UI with text and voice input
```

---

## Setup for JAX benchmark

```bash
pip install jax[cuda12] flax
python benchmark.py
```

---

Kiran Gowda
[LinkedIn](https://www.linkedin.com/in/kirangowda3101/) | [GitHub](https://github.com/kirangowda3101)
