# Speech Language Model

A 522 million parameter joint speech-text language model built from scratch. The core idea is simple: treat audio the same way GPT treats text. Instead of building separate systems for speech and language, this model learns both in a single unified vocabulary and a single transformer.

No pretrained speech encoders. No fine-tuning on top of an existing model. Everything from the architecture to the training pipeline was written from scratch.

**Model weights:** [HuggingFace](https://huggingface.co/kirangowda3101/speech-language-model) (5.9GB) **Code:** [GitHub](https://github.com/kirangowda3101/Speech-Language-Model)

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

| Setting          | Value                                       |
| ---------------- | ------------------------------------------- |
| Parameters       | 522M                                        |
| Layers           | 24                                          |
| Attention heads  | 16                                          |
| d_model          | 1024                                        |
| FFN hidden dim   | 4096                                        |
| Context window   | 2048 tokens                                 |
| Text vocabulary  | 50,257 (GPT-2 BPE)                          |
| Audio vocabulary | 8,192 (8 codebooks x 1024)                  |
| Special tokens   | 5 (BOS, EOS, PAD, AUDIO_START, AUDIO_END)   |
| Total vocabulary | 58,454                                      |

---

## Training

Training happened in two phases. Phase 1 pre-trained the model on LibriSpeech from a random initialization. Phase 2 continued training on GigaSpeech, a much larger and more diverse corpus, to test how the model scales with more data.

Each audio file is preprocessed once before training. EnCodec encodes it to discrete codes, the tokenizer flattens those codes to integers, and the result is saved as a `.npy` file on disk. The training loop loads these files directly so no GPU time is spent on audio decoding during training.

**Optimizer:** AdamW with weight decay 0.1, gradient clipping at 1.0. Learning rate follows a cosine decay schedule with linear warmup.

### Phase 1 — LibriSpeech (100 hours)

LibriSpeech train-clean-100, 100 hours of clean read English speech. Trained from scratch for 100K steps on a single GPU.

| Step    | Val Loss |
| ------- | -------- |
| 0       | 11.00    |
| 1,000   | 4.90     |
| 10,000  | 3.38     |
| 100,000 | 3.49     |

A randomly initialized model on this vocabulary scores log(58454) = 10.98. Reaching 3.49 confirms the model learned real structure in both speech and language.

### Phase 2 — GigaSpeech (1,000 hours)

To test scaling, training continued on GigaSpeech Medium (`gigaspeech-m`), roughly 1,000 hours of far more diverse audio: podcasts, YouTube, audiobooks, and spontaneous conversational speech. 910,140 utterances were tokenized to EnCodec codes. Training ran from step 100K to step 250K — 150K additional steps.

**Infrastructure:** A single V100-SXM2 GPU on Northeastern University's Explorer HPC cluster, managed through SLURM on the 8-hour-per-job partition. Because a full run far exceeds one job's time limit, jobs were chained automatically: each job pre-submits its successor with a SLURM dependency before training begins, so the run continues unattended across dozens of jobs until it reaches the target step, then cancels the chain.

**Memory:** Fitting a 522M model plus activations on a 32GB V100 required a batch size of 2 with 16 gradient accumulation steps (effective batch 32), fp16 mixed precision, and gradient checkpointing. DDP ran with `find_unused_parameters=True`.

**Result:** validation loss over the GigaSpeech phase, sampled across the run:

| Step (approx) | Val Loss |
| ------------- | -------- |
| 100,000       | 3.88     |
| 130,000       | 3.78     |
| 160,000       | 3.70     |
| 190,000       | 3.66     |
| 210,000       | 3.62     |
| 230,000       | 3.59     |
| 250,000       | 3.54     |

Two things about this curve are worth reading honestly.

First, val loss *jumped up* to 3.88 when training switched to GigaSpeech, then declined steadily to 3.54 over 150K steps. The jump is expected: GigaSpeech is far harder than LibriSpeech, so a model tuned on clean audiobooks initially predicts diverse spontaneous speech worse. The steady decline that follows is the real signal — the model kept learning and the curve shows convergence behavior, flattening near the end as it approached what this data and compute budget could give.

Second, the final GigaSpeech val loss (3.54) is slightly higher than the LibriSpeech number (3.49). This is not a regression. The two numbers are measured on different, non-comparable data: 3.49 on narrow, clean read speech versus 3.54 on broad, noisy, spontaneous speech. A 3.54 on hard, diverse audio represents a more general model than a 3.49 on easy audio. The final training loss reached ~2.99, confirming the model was still actively fitting the data.

---

## What the output sounds like right now

A sample generated from the GigaSpeech checkpoint is in [`samples/sample_long1.wav`](samples/sample_long1.wav) (10 seconds, 24kHz, prompted with "Hello, how are you doing today? I hope you are well.").

At val loss 3.54 the output is **speech-shaped but not yet intelligible**. It is not random noise, and a quick spectral analysis shows why: the energy concentrates in the 164–1054 Hz band, exactly where human speech lives, and there is weak periodic structure around 250 Hz consistent with vocal pitch. In other words, the model has learned the acoustic *texture* of speech — the frequency distribution and rough rhythm — without yet forming coherent words. The generation also degrades over long horizons: past a few seconds it tends to collapse toward silence, a known failure mode when sampling well beyond the content the model has learned to sustain.

This is the honest state of the model, and it matches expectation. Well-trained speech synthesis models typically reach val loss below 2.0 and train on 1,000 to 10,000 hours with far more compute. This project reached 3.54 and demonstrates that the architecture and full pipeline work end to end, that the model learns genuine acoustic structure, and that it continues to improve with more data. The remaining gap to intelligible speech is a data-and-compute limitation, not an architecture one.

---

## JAX benchmark

The attention block was re-implemented in JAX/Flax to compare throughput against PyTorch on the same hardware. JAX uses XLA compilation through `jax.jit` which fuses operations and eliminates Python overhead after the first forward pass.

Both frameworks ran on the same L40S GPU.

| Sequence length | PyTorch (ms) | JAX (ms) | JAX speedup |
| --------------- | ------------ | -------- | ----------- |
| 64              | 0.14         | 0.11     | 1.24x       |
| 128             | 0.23         | 0.16     | 1.47x       |
| 256             | 0.43         | 0.26     | 1.63x       |
| 512             | 0.85         | 1.00     | 0.85x       |

JAX is faster for shorter sequences where XLA kernel fusion pays off the most. PyTorch recovers at sequence length 512, likely because cuDNN's attention kernels are more optimized at that size. The crossover is somewhere between 256 and 512 tokens.

---

## Engineering notes

A few real problems solved along the way, kept here because the debugging is part of the work:

- **torchcodec / FFmpeg on the cluster.** The GigaSpeech loader failed to decode audio because a required FFmpeg shared library was missing on the compute nodes. Bypassed by streaming the dataset with decoding disabled and decoding manually via `soundfile` from in-memory bytes.
- **Out-of-memory on the V100.** A 522M model does not fit on 32GB at a naive batch size. Solved with a small per-step batch, gradient accumulation to preserve effective batch size, fp16, and gradient checkpointing.
- **DDP unused parameters.** Distributed training raised a gradient-reduction error until DDP was initialized with `find_unused_parameters=True`.
- **Checkpoint key mismatch.** Checkpoints saved under DDP / `torch.compile` carry an `_orig_mod.` prefix on state-dict keys; inference strips it automatically.
- **Disk quota management.** The home directory quota required active cleanup of accumulated step checkpoints (keeping only the best and latest) and relocating raw datasets.

---

## Running the demo locally

Clone the repo and download the weights:

```
git clone https://github.com/kirangowda3101/Speech-Language-Model
cd Speech-Language-Model

pip install torch torchaudio encodec tiktoken fastapi uvicorn python-multipart scipy

# Download model weights from HuggingFace
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('kirangowda3101/speech-language-model', 'best.pt', local_dir='.')"
```

Generate an audio sample from a checkpoint:

```
python generate.py \
  --ckpt best.pt \
  --text "The speaker said hello" \
  --out sample.wav \
  --max_tokens 600
```

Token math: 75 frames/s x 8 codebooks = 600 tokens per second, so `--max_tokens 600` is about 1 second of audio.

Or run the browser demo server:

```
python -m uvicorn app:app --reload
```

Open `http://localhost:8000`. The demo has two modes: type text and hear the model speak it, or record your voice and hear the model respond in its own voice.

Note: inference on CPU takes 30-60 seconds per generation for this model size. Running on a GPU brings that down to 5-15 seconds.

---

## Scaling further

The bottleneck throughout this project was compute: a single V100 with 8-hour job limits. The architecture and pipeline scale directly to much larger runs.

GigaSpeech XL (10,000 hours) with several times more training steps would very likely push val loss below 2.0, the range where speech synthesis models start producing recognizable words. Combined with an instruction-following dataset, the model could learn to respond to spoken input rather than only continue audio. The code, architecture, and training pipeline are all ready for that; the only missing ingredient is compute.

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
preprocess.py      preprocessing of LibriSpeech / GigaSpeech to .npy files
train.py           DDP training loop with checkpointing
checkpoint.py      atomic checkpoint save and load with auto-cleanup
generate.py        standalone inference: checkpoint + prompt -> .wav
jax_attention.py   attention block re-implemented in JAX
flax_model.py      full model re-implemented in Flax
benchmark.py       PyTorch vs JAX throughput comparison script
app.py             FastAPI inference server
static/index.html  browser demo UI with text and voice input
samples/           generated audio samples
```

---

## Setup for JAX benchmark

```
pip install jax[cuda12] flax
python benchmark.py
```

---

Kiran Gowda [LinkedIn](https://www.linkedin.com/in/kirangowda3101/) | [GitHub](https://github.com/kirangowda3101)
