"""
app.py  —  SpeechLM FastAPI Demo
User speaks → model generates audio tokens → plays back in model voice.

Run:
    python -m uvicorn app:app --reload --limit-concurrency 1
Then open: http://localhost:8000
"""

import io
import sys
import time
import tempfile
import subprocess
import numpy as np
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="SpeechLM Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Globals ───────────────────────────────────────────────────────────────────
_model     = None
_tokenizer = None
_encodec   = None
_device    = "cpu"          # safe default for Mac
CKPT       = ROOT / "best.pt"
AUDIO_DIR  = ROOT / "static" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


class GenerateResponse(BaseModel):
    audio_url:              str
    duration_seconds:       float
    tokens_generated:       int
    inference_time_seconds: float


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    global _model, _tokenizer, _encodec, _device

    # Pick device
    if torch.cuda.is_available():
        _device = "cuda"
    elif torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"
    print(f"[INFO] Device: {_device}")

    if not CKPT.exists():
        print(f"[WARNING] No checkpoint at {CKPT} — mock mode")
        return

    print(f"[INFO] Loading {CKPT}")
    from config          import medium_config
    from model           import SpeechLM
    from tokenizer       import SpeechLMTokenizer
    from encodec_wrapper import EnCodecWrapper

    cfg  = medium_config()
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)

    _model = SpeechLM(cfg)

    # Strip torch.compile / DDP prefix  (_orig_mod.)
    state = {}
    for k, v in ckpt["model_state"].items():
        state[k.replace("_orig_mod.", "")] = v
    missing, _ = _model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARNING] {len(missing)} missing keys (first: {missing[0]})")

    _model.to(_device)
    _model.eval()

    _tokenizer = SpeechLMTokenizer(cfg.vocab)
    _encodec   = EnCodecWrapper(bandwidth=6.0, device=_device)

    n = sum(p.numel() for p in _model.parameters())
    print(f"[INFO] Model ready — {n/1e6:.1f}M params | loss={ckpt.get('loss','?')}")


@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        import traceback; traceback.print_exc()


# ── Upload endpoint ───────────────────────────────────────────────────────────
@app.post("/generate-from-audio", response_model=GenerateResponse)
async def generate_from_audio(
    request:     Request,
    file:        UploadFile = File(...),
    max_tokens:  int   = 200,
    temperature: float = 0.8,
    top_p:       float = 0.9,
):
    # Allow up to 10 MB upload
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 10 MB)")

    print(f"[INFO] Received audio: {len(raw)/1024:.1f} KB, type={file.content_type}")
    t0 = time.time()

    try:
        audio_out, n_tokens = _generate(raw, max_tokens, temperature, top_p)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"Generation error: {e}")

    wav = _save_wav(audio_out)
    elapsed = time.time() - t0
    print(f"[INFO] Done in {elapsed:.1f}s — {n_tokens} tokens, {len(audio_out)/24000:.1f}s audio")

    return GenerateResponse(
        audio_url              = f"/audio/{wav.name}",
        duration_seconds       = round(len(audio_out) / 24000, 2),
        tokens_generated       = n_tokens,
        inference_time_seconds = round(elapsed, 2),
    )


def _load_audio(raw_bytes: bytes):
    """
    Try multiple methods to decode incoming browser audio (webm, ogg, wav).
    Returns (waveform_tensor, sample_rate).
    """
    import torchaudio

    # Method 1: torchaudio directly
    try:
        buf = io.BytesIO(raw_bytes)
        waveform, sr = torchaudio.load(buf)
        print(f"[INFO] torchaudio loaded: shape={waveform.shape}, sr={sr}")
        return waveform, sr
    except Exception as e1:
        print(f"[INFO] torchaudio direct failed: {e1}")

    # Method 2: write to temp file and load (helps with webm)
    try:
        import tempfile, os
        suffix = ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(raw_bytes)
            tmp = f.name
        waveform, sr = torchaudio.load(tmp)
        os.unlink(tmp)
        print(f"[INFO] torchaudio file load: shape={waveform.shape}, sr={sr}")
        return waveform, sr
    except Exception as e2:
        print(f"[INFO] torchaudio file failed: {e2}")

    # Method 3: ffmpeg → wav → torchaudio
    try:
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as fin:
            fin.write(raw_bytes)
            inp = fin.name
        out = inp.replace(".webm", ".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", inp, "-ar", "24000", "-ac", "1", out],
            capture_output=True, check=True
        )
        waveform, sr = torchaudio.load(out)
        os.unlink(inp); os.unlink(out)
        print(f"[INFO] ffmpeg converted: shape={waveform.shape}, sr={sr}")
        return waveform, sr
    except Exception as e3:
        print(f"[INFO] ffmpeg failed: {e3}")

    # Method 4: soundfile
    try:
        import soundfile as sf
        buf = io.BytesIO(raw_bytes)
        data, sr = sf.read(buf, dtype="float32")
        if data.ndim == 1:
            data = data[np.newaxis, :]
        else:
            data = data.T
        waveform = torch.from_numpy(data)
        print(f"[INFO] soundfile loaded: shape={waveform.shape}, sr={sr}")
        return waveform, sr
    except Exception as e4:
        print(f"[INFO] soundfile failed: {e4}")

    raise RuntimeError(
        "Could not decode audio. Install ffmpeg: brew install ffmpeg"
    )


def _generate(raw_bytes: bytes, max_tokens=200, temperature=0.8, top_p=0.9):
    """Audio bytes in → generated audio numpy array out."""

    if _model is None:
        # Mock: return 2s of silence
        return np.zeros(24000 * 2, dtype=np.float32), 0

    import torchaudio

    # 1. Load audio
    waveform, sr = _load_audio(raw_bytes)

    # 2. Normalise: 24 kHz mono
    if sr != 24000:
        waveform = torchaudio.transforms.Resample(sr, 24000)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    waveform = waveform.to(_device)
    MAX_INPUT_SAMPLES = 24000 * 2
    if waveform.shape[-1] > MAX_INPUT_SAMPLES:
        waveform = waveform[:, :MAX_INPUT_SAMPLES]

    # 3. EnCodec encode
    with torch.no_grad():
        codes = _encodec.encode(waveform, sample_rate=24000)   # (K, T)
    input_audio_ids = _tokenizer.encode_audio_codes(codes.cpu().numpy())
    print(f"[INFO] Input audio: {len(input_audio_ids)} tokens from {waveform.shape[-1]/24000:.1f}s")

    # 4. Build prompt
    prompt = (
        [_tokenizer.bos_id, _tokenizer.audio_start_id]
        + input_audio_ids
        + [_tokenizer.audio_end_id]
    )
    ids = torch.tensor([prompt], dtype=torch.long, device=_device)

    # 5. Autoregressive generation
    AUD_OFF = _tokenizer.cfg.audio_token_offset    # 50257
    AUD_END = _tokenizer.cfg.special_token_offset  # 58449
    new_ids = []

    with torch.no_grad():
        for step in range(max(max_tokens, 600)):
            logits, _ = _model(ids)
            lg = logits[0, -1, :].float()

            # Audio tokens only
            mask = torch.full_like(lg, float("-inf"))
            mask[AUD_OFF:AUD_END] = lg[AUD_OFF:AUD_END]
            mask = mask / max(temperature, 1e-8)

            probs = torch.softmax(mask, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, 0)
            sp[cum - sp > top_p] = 0.0
            s = sp.sum()
            if s > 0:
                probs = torch.zeros_like(probs).scatter_(0, si, sp / s)

            tok = torch.multinomial(probs, 1)
            new_ids.append(tok.item() - AUD_OFF)
            ids = torch.cat([ids, tok.unsqueeze(0)], dim=1)

            if (step + 1) % 50 == 0:
                print(f"[INFO] Generated {step+1}/{max_tokens} tokens")

    if not new_ids:
        return np.zeros(24000, dtype=np.float32), 0

    # 6. Reshape → (K, T)
    K = _encodec.num_codebooks   # 8
    T = len(new_ids) // K
    if T == 0:
        return np.zeros(24000, dtype=np.float32), 0

    flat = np.array(new_ids[:T * K], dtype=np.int64)
    grid = flat.reshape(T, K).T        # interleaved → (K, T)
    grid = np.clip(grid, 0, 1023)

    # 7. Decode
    wave = _encodec.decode(torch.from_numpy(grid).long())
    return wave.cpu().numpy(), len(new_ids)


def _save_wav(audio: np.ndarray, sr: int = 24000) -> Path:
    import scipy.io.wavfile as wio
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9
    tmp = tempfile.NamedTemporaryFile(
        suffix=".wav", prefix="sl_", dir=AUDIO_DIR, delete=False
    )
    wio.write(tmp.name, sr, (audio * 32767).astype(np.int16))
    return Path(tmp.name)


# ── Static files ──────────────────────────────────────────────────────────────
@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    p = AUDIO_DIR / filename
    if not p.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(str(p), media_type="audio/wav")

@app.get("/health")
async def health():
    return {"model_loaded": _model is not None, "device": _device}
@app.post("/generate-from-text")
async def generate_from_text(request: Request):
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(400, "No text provided")
    t0 = time.time()
    try:
        text_ids = _tokenizer.encode_text(text)
        prompt = [_tokenizer.bos_id] + text_ids + [_tokenizer.audio_start_id]
        ids = torch.tensor([prompt], dtype=torch.long, device=_device)
        AUD_OFF = _tokenizer.cfg.audio_token_offset
        AUD_END = _tokenizer.cfg.special_token_offset
        new_ids = []
        with torch.no_grad():
            for _ in range(int(body.get("max_tokens", 600))):
                logits, _ = _model(ids)
                lg = logits[0, -1, :].float()
                mask = torch.full_like(lg, float("-inf"))
                mask[AUD_OFF:AUD_END] = lg[AUD_OFF:AUD_END]
                mask = mask / 0.8
                probs = torch.softmax(mask, dim=-1)
                tok = torch.multinomial(probs, 1)
                new_ids.append(tok.item() - AUD_OFF)
                ids = torch.cat([ids, tok.unsqueeze(0)], dim=1)
        K = _encodec.num_codebooks
        T = len(new_ids) // K
        flat = np.array(new_ids[:T*K], dtype=np.int64)
        grid = np.clip(flat.reshape(T, K).T, 0, 1023)
        wave = _encodec.decode(torch.from_numpy(grid).long())
        wav = _save_wav(wave.cpu().numpy())
        return {"audio_url": f"/audio/{wav.name}", "tokens": len(new_ids), "time": round(time.time()-t0,2)}
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

@app.get("/")
async def root():
    return FileResponse(str(ROOT / "static" / "index.html"))
