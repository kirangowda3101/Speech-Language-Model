"""
generate.py — Standalone inference script for SpeechLM.

Modes:
  Text-to-audio  : give --text "some prompt"
  Audio-to-audio : give --wav path/to/input.wav

Examples:
  python generate.py --ckpt ~/checkpoints/speechlm_medium_giga/best.pt \
                     --text "The quick brown fox" --out sample.wav

  python generate.py --ckpt ~/checkpoints/speechlm_medium_giga/best.pt \
                     --wav prompt.wav --out continuation.wav \
                     --max_tokens 600 --temperature 0.8
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import medium_config
from model import SpeechLM
from tokenizer import SpeechLMTokenizer
from encodec_wrapper import EnCodecWrapper


def load_checkpoint(ckpt_path: str, device: str):
    cfg = medium_config()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = SpeechLM(cfg)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys (e.g. {missing[0]})")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys")

    model.to(device).eval()

    loss = ckpt.get("loss", "?")
    step = ckpt.get("step", ckpt.get("iter", "?"))
    print(f"[info] Loaded checkpoint: step={step}, loss={loss}")

    tok = SpeechLMTokenizer(cfg.vocab)
    enc = EnCodecWrapper(bandwidth=6.0, device=device)

    return model, tok, enc, cfg


def build_text_prompt(tok, text: str, device: str) -> torch.Tensor:
    text_ids = tok.encode_text(text)
    prompt = [tok.bos_id] + text_ids + [tok.audio_start_id]
    return torch.tensor([prompt], dtype=torch.long, device=device)


def build_audio_prompt(tok, enc, wav_path: str, device: str) -> torch.Tensor:
    waveform, sr = torchaudio.load(wav_path)
    if sr != 24000:
        waveform = torchaudio.transforms.Resample(sr, 24000)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    waveform = waveform[:, :24000 * 2]  # cap at 2s input
    waveform = waveform.to(device)

    codes = enc.encode(waveform, sample_rate=24000)
    audio_ids = tok.encode_audio_codes(codes.cpu().numpy())
    print(f"[info] Input audio: {len(audio_ids)} tokens from {waveform.shape[-1]/24000:.1f}s")

    prompt = [tok.bos_id, tok.audio_start_id] + audio_ids + [tok.audio_end_id]
    return torch.tensor([prompt], dtype=torch.long, device=device)


def sample(model, tok, ids: torch.Tensor, cfg, max_tokens: int, temperature: float, top_p: float):
    AUD_OFF = tok.cfg.audio_token_offset
    AUD_END = tok.cfg.special_token_offset
    new_ids = []

    with torch.no_grad():
        for step in range(max_tokens):
            # Crop to max_seq_len
            ctx = ids[:, -cfg.model.max_seq_len:]
            logits, _ = model(ctx)
            lg = logits[0, -1, :].float()

            # Mask to audio tokens only
            mask = torch.full_like(lg, float("-inf"))
            mask[AUD_OFF:AUD_END] = lg[AUD_OFF:AUD_END]
            mask = mask / max(temperature, 1e-8)

            probs = torch.softmax(mask, dim=-1)

            # Top-p (nucleus) sampling
            sp, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, 0)
            sp[cum - sp > top_p] = 0.0
            s = sp.sum()
            if s > 0:
                probs = torch.zeros_like(probs).scatter_(0, si, sp / s)

            next_tok = torch.multinomial(probs, 1)
            new_ids.append(next_tok.item() - AUD_OFF)
            ids = torch.cat([ids, next_tok.unsqueeze(0)], dim=1)

            if (step + 1) % 100 == 0:
                print(f"[info] {step+1}/{max_tokens} tokens generated", flush=True)

    return new_ids


def tokens_to_wav(new_ids, enc, K=8) -> np.ndarray:
    T = len(new_ids) // K
    if T == 0:
        return np.zeros(24000, dtype=np.float32)
    flat = np.array(new_ids[:T * K], dtype=np.int64)
    grid = np.clip(flat.reshape(T, K).T, 0, 1023)  # (K, T)
    wave = enc.decode(torch.from_numpy(grid).long())
    return wave.cpu().numpy()


def save_wav(audio: np.ndarray, out_path: str, sr: int = 24000):
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9
    waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
    torchaudio.save(out_path, waveform, sr)
    duration = len(audio) / sr
    print(f"[info] Saved {out_path} ({duration:.2f}s)")


def main():
    parser = argparse.ArgumentParser(description="SpeechLM inference")
    parser.add_argument("--ckpt",        required=True,       help="Path to best.pt checkpoint")
    parser.add_argument("--text",        default=None,        help="Text prompt (text-to-audio)")
    parser.add_argument("--wav",         default=None,        help="Audio prompt (audio-to-audio)")
    parser.add_argument("--out",         default="output.wav",help="Output .wav path")
    parser.add_argument("--max_tokens",  type=int,   default=600,  help="Tokens to generate (600 ≈ 1s)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p",       type=float, default=0.9)
    parser.add_argument("--device",      default=None,        help="cpu / cuda / mps (auto-detect if omitted)")
    parser.add_argument("--seed",        type=int,   default=None)
    args = parser.parse_args()

    if args.text is None and args.wav is None:
        parser.error("Provide --text or --wav (or both)")

    # Device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    print(f"[info] Device: {args.device}")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, tok, enc, cfg = load_checkpoint(args.ckpt, args.device)

    if args.wav:
        ids = build_audio_prompt(tok, enc, args.wav, args.device)
    else:
        ids = build_text_prompt(tok, args.text, args.device)

    print(f"[info] Prompt length: {ids.shape[1]} tokens")
    print(f"[info] Generating {args.max_tokens} audio tokens "
          f"(≈{args.max_tokens / (75 * enc.num_codebooks):.1f}s at 75fps × {enc.num_codebooks} codebooks)...")

    new_ids = sample(model, tok, ids, cfg, args.max_tokens, args.temperature, args.top_p)
    audio = tokens_to_wav(new_ids, enc, K=enc.num_codebooks)
    save_wav(audio, args.out)


if __name__ == "__main__":
    main()
