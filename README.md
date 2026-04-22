# Speech Language Model

A 522 million parameter joint speech-text language model built from scratch. The model learns to generate speech by treating audio as a sequence of discrete tokens, the same way GPT treats text. No pretrained speech encoder, no fine-tuning on top of Whisper — everything is trained from the ground up.

## What it does

You type a sentence, the model generates audio. Or you speak into a microphone and the model responds in its own voice. The output at this training stage sounds like structured noise — the model has learned patterns in speech but hasn't converged to intelligible words yet. That is expected and honest for 100K training steps on 100 hours of data.

## How it works

Text and audio live in the same vocabulary. GPT-2's BPE tokenizer covers text (50,257 tokens). Meta's EnCodec neural audio codec converts raw audio into discrete codes at 75 frames per second with 8 codebooks, producing 600 audio tokens per second. Both token types are concatenated into a single sequence and fed into a transformer that predicts the next token — whether that next token is a word or a piece of audio.

Training sequences look like this:

```
[BOS] the quick brown fox [AUDIO_START] ...600 audio tokens... [AUDIO_END] [EOS]
```

The model learns to continue audio after seeing text, and continue text after seeing audio.

## Architecture

The transformer follows a GPT-style decoder-only design with a few modern improvements over the original GPT-2 paper.

Rotary positional encoding (RoPE) replaces learned absolute positions. Unlike the original sinusoidal or learned embeddings, RoPE encodes position by rotating the query and key vectors, which generalizes better to sequence lengths not seen during training.

SwiGLU replaces the standard ReLU feed-forward network. It uses a gating mechanism where one linear projection controls how much information from another projection passes through. In practice this trains faster and reaches lower loss than ReLU at the same parameter count.

The vocabulary has three zones: text tokens (0 to 50,256), audio tokens (50,257 to 58,448), and five special tokens for BOS, EOS, PAD, AUDIO_START, and AUDIO_END. The embedding layer handles all three zones with separate learned tables that get summed at each position.

| Parameter | Value |
|---|---|
| Total parameters | 522M |
| Layers | 24 |
| Attention heads | 16 |
| d_model | 1024 |
| FFN hidden dim | 4096 |
| Context window | 2048 tokens |
| Total vocabulary | 58,454 |

## Training

Data is 100 hours of LibriSpeech train-clean-100. Each audio file gets preprocessed once: EnCodec encodes it to codes, the tokenizer flattens those codes to integers, and the result is saved as a `.npy` file. Training loads these files directly so no GPU time is wasted on audio decoding.

Training used PyTorch DDP across one A100 GPU on Northeastern University's Explorer HPC cluster, managed through SLURM. Jobs were chained automatically so training resumed across the cluster's 55-minute time limit without manual intervention.

The learning rate follows a cosine decay schedule with linear warmup over 2,000 steps, starting at 3e-4 and decaying to 3e-5. AdamW with weight decay 0.1 and gradient clipping at 1.0.

Loss went from 11.0 at initialization (random weights, log(58454) ≈ 10.98) to 3.49 at 100K steps.

## JAX Benchmark

The attention block was re-implemented in JAX/Flax to compare throughput against PyTorch. Both versions run on the same L40S GPU. JAX uses XLA compilation via `jax.jit` which fuses operations and eliminates Python overhead after the first call.

| Sequence length | PyTorch (ms) | JAX (ms) | JAX speedup |
|---|---|---|---|
| 64 | 0.14 | 0.11 | 1.24x |
| 128 | 0.23 | 0.16 | 1.47x |
| 256 | 0.43 | 0.26 | 1.63x |
| 512 | 0.85 | 1.00 | 0.85x |

JAX is faster for shorter sequences where XLA's kernel fusion pays off. PyTorch wins at sequence length 512, likely because cuDNN's attention kernels are more optimized at that size. The crossover point sits somewhere between 256 and 512 tokens.

## Demo

The demo runs as a FastAPI server. It accepts either typed text or recorded audio, runs inference on the model, decodes the generated audio tokens back to a waveform using EnCodec, and returns a WAV file.

To run it on the cluster with GPU:

```bash
conda activate speechlm
cd ~/speech_lm
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Then from your local machine:

```bash
ssh -L 8000:NODE_NAME:8000 username@login.explorer.northeastern.edu
```

Open `http://localhost:8000` in your browser.

## Repository structure

```
config.py          vocabulary layout and model hyperparameters
model.py           transformer architecture (RoPE, SwiGLU, causal attention)
tokenizer.py       text and audio tokenization
encodec_wrapper.py thin wrapper around Meta's EnCodec model
audio_utils.py     audio loading, normalization, chunking
dataset.py         PyTorch Dataset for preprocessed token files
dataloader.py      DataLoader construction with DDP support
preprocess.py      one-time preprocessing of LibriSpeech to .npy files
train.py           DDP training loop with checkpointing and LR scheduling
checkpoint.py      atomic checkpoint save/load with auto-cleanup
jax_attention.py   attention block re-implemented in JAX
flax_model.py      full model re-implemented in Flax
benchmark.py       PyTorch vs JAX throughput comparison
app.py             FastAPI inference server
static/index.html  browser demo UI
```

## What's next

The obvious next step is more training. 100K steps on 100 hours is a proof of concept. Intelligible speech typically requires 500K+ steps and ideally the full 960-hour LibriSpeech split. KV-cache would make inference fast enough to be interactive. LoRA fine-tuning on a specific speaker could produce a consistent voice with much less compute.

## Setup

```bash
pip install torch torchaudio encodec tiktoken fastapi uvicorn scipy
pip install jax[cuda12] flax  # for JAX benchmark only
```

The model checkpoint (`best.pt`, 5.9GB) is not included in this repository due to file size. Contact me if you want the weights.
