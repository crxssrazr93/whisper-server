#!/usr/bin/env python3
"""Drop-in faster-whisper transcription server for the LAN clients.

Rebuilt to match the exact contract used by transcribe-remote.py /
transcribe-remote-diarize.py:

  GET  /health     -> status JSON (model, device, gpu, loaded, idle, batch, parallel, diarize)
  POST /transcribe -> multipart: file + form(formats, word_timestamps, vad_filter,
                      diarize, job_id, [language], [diarize_model])
                      returns {duration, language, formats:{fmt:text}, diarized,
                      num_speakers, diarize_seconds}; HTTP 499 {"cancelled":true} on cancel
  POST /cancel      -> form(job_id) cancels an in-flight job

Backend: faster-whisper (CTranslate2), CUDA/float16, lazy load + idle unload,
batched inference, bounded parallelism, optional pyannote diarization.

Config via environment (see whisper-server.env):
  WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, WHISPER_IDLE_TIMEOUT,
  WHISPER_BATCH_SIZE, WHISPER_MAX_PARALLEL, WHISPER_DIARIZE_MODEL,
  WHISPER_HOST, WHISPER_PORT, HF_TOKEN
"""
from __future__ import annotations

import os
import threading
import time
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

# faster_whisper / torch / pyannote are imported lazily (inside functions) so a
# socket-activated cold start can answer /health before the heavy libs load.

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MODEL_NAME    = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
DEVICE        = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE  = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
IDLE_TIMEOUT  = int(os.environ.get("WHISPER_IDLE_TIMEOUT", "600"))   # seconds; 0 disables
BATCH_SIZE    = int(os.environ.get("WHISPER_BATCH_SIZE", "8"))
MAX_PARALLEL  = int(os.environ.get("WHISPER_MAX_PARALLEL", "1"))
DIARIZE_MODEL = os.environ.get("WHISPER_DIARIZE_MODEL", "pyannote/speaker-diarization-community-1")
HOST          = os.environ.get("WHISPER_HOST", "0.0.0.0")
PORT          = int(os.environ.get("WHISPER_PORT", "9000"))
HF_TOKEN      = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
SAMPLE_RATE   = 16000

ALL_FORMATS = ("txt", "srt", "vtt", "json")

# --------------------------------------------------------------------------- #
# Model state (lazy load + idle unload)
# --------------------------------------------------------------------------- #
_model: WhisperModel | None = None
_batched: BatchedInferencePipeline | None = None
_diar_pipelines: dict[str, Any] = {}
_model_lock = threading.Lock()
_last_used: float | None = None
_last_activity: float | None = None   # time of last HTTP request (drives idle exit)
_gpu_name = "?"

_sem = threading.Semaphore(MAX_PARALLEL)
_active = 0
_active_lock = threading.Lock()

# job_id -> threading.Event (set => cancel requested)
_jobs: dict[str, threading.Event] = {}
_jobs_lock = threading.Lock()


class Cancelled(Exception):
    pass


def _gpu_info() -> dict:
    """GPU descriptor the client expects: {device, total_mb, free_mb, used_mb}."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            mb = 1024 * 1024
            return {
                "device": name,
                "total_mb": total // mb,
                "free_mb": free // mb,
                "used_mb": (total - free) // mb,
            }
    except Exception:
        pass
    return {"device": DEVICE}


def _get_model():
    global _model, _batched, _last_used, _gpu_name
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel, BatchedInferencePipeline
            _model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE, num_workers=MAX_PARALLEL)
            _batched = BatchedInferencePipeline(model=_model)
            _gpu_name = _gpu_info().get("device", DEVICE)
        _last_used = time.monotonic()
        return _model, _batched


def _idle_exit_watchdog(server) -> None:
    """Exit the whole process after IDLE_TIMEOUT with no requests.

    Under systemd socket activation this frees all RAM/VRAM; the socket unit
    relaunches the service on the next incoming connection.
    """
    if IDLE_TIMEOUT <= 0:
        return
    # Wait until uvicorn is actually serving.
    for _ in range(1200):
        if getattr(server, "started", False):
            break
        time.sleep(0.1)
    while True:
        time.sleep(5)
        with _active_lock:
            busy = _active > 0
        last = _last_activity
        if busy or last is None:
            continue
        if time.monotonic() - last >= IDLE_TIMEOUT:
            server.should_exit = True
            return


def _get_diar_pipeline(model_id: str):
    """Load and cache a pyannote diarization pipeline."""
    if model_id in _diar_pipelines:
        return _diar_pipelines[model_id]
    from pyannote.audio import Pipeline
    import torch
    try:
        # pyannote.audio 4.x
        pipe = Pipeline.from_pretrained(model_id, token=HF_TOKEN)
    except TypeError:
        # pyannote.audio 3.x
        pipe = Pipeline.from_pretrained(model_id, use_auth_token=HF_TOKEN)
    if pipe is None:
        raise RuntimeError(f"could not load diarization pipeline {model_id!r} (check HF_TOKEN / gated terms)")
    if DEVICE == "cuda":
        try:
            pipe.to(torch.device("cuda"))
        except Exception:
            pass
    _diar_pipelines[model_id] = pipe
    return pipe


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _ts(seconds: float, sep: str = ",") -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _label(seg: dict) -> str:
    spk = seg.get("speaker")
    text = seg["text"].strip()
    return f"[{spk}] {text}" if spk else text


def build_txt(segments: list[dict], diarized: bool) -> str:
    return "\n".join(_label(s) for s in segments).strip() + "\n"


def build_srt(segments: list[dict], diarized: bool) -> str:
    out = []
    for i, s in enumerate(segments, 1):
        out.append(str(i))
        out.append(f"{_ts(s['start'], ',')} --> {_ts(s['end'], ',')}")
        out.append(_label(s))
        out.append("")
    return "\n".join(out)


def build_vtt(segments: list[dict], diarized: bool) -> str:
    out = ["WEBVTT", ""]
    for s in segments:
        out.append(f"{_ts(s['start'], '.')} --> {_ts(s['end'], '.')}")
        out.append(_label(s))
        out.append("")
    return "\n".join(out)


def build_json(segments: list[dict], language: str, duration: float,
               diarized: bool, word_timestamps: bool) -> str:
    import json
    payload = {
        "language": language,
        "duration": round(duration, 3),
        "diarized": diarized,
        "segments": [],
    }
    for s in segments:
        item = {"start": round(s["start"], 3), "end": round(s["end"], 3),
                "text": s["text"].strip()}
        if s.get("speaker"):
            item["speaker"] = s["speaker"]
        if word_timestamps and s.get("words"):
            item["words"] = [
                {"start": round(w["start"], 3), "end": round(w["end"], 3),
                 "word": w["word"], **({"speaker": w["speaker"]} if w.get("speaker") else {})}
                for w in s["words"]
            ]
        payload["segments"].append(item)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_formats(requested: list[str], segments: list[dict], language: str,
                   duration: float, diarized: bool, word_timestamps: bool) -> dict[str, str]:
    out: dict[str, str] = {}
    for fmt in requested:
        if fmt == "txt":
            out["txt"] = build_txt(segments, diarized)
        elif fmt == "srt":
            out["srt"] = build_srt(segments, diarized)
        elif fmt == "vtt":
            out["vtt"] = build_vtt(segments, diarized)
        elif fmt == "json":
            out["json"] = build_json(segments, language, duration, diarized, word_timestamps)
    return out


# --------------------------------------------------------------------------- #
# Diarization
# --------------------------------------------------------------------------- #
def _to_annotation(diarization):
    """Normalize a pyannote diarization result to an Annotation with .itertracks().

    pyannote 3.x returns the Annotation directly; 4.x community-1 returns a
    DiarizeOutput whose .speaker_diarization holds the Annotation.
    """
    if hasattr(diarization, "itertracks"):
        return diarization
    for attr in ("speaker_diarization", "diarization", "annotation"):
        ann = getattr(diarization, attr, None)
        if ann is not None and hasattr(ann, "itertracks"):
            return ann
    raise RuntimeError(f"unexpected diarization output type: {type(diarization).__name__}")


def _assign_speakers(segments: list[dict], diarization) -> int:
    """Tag each transcript segment with the dominant overlapping speaker."""
    ann = _to_annotation(diarization)
    turns = [(t.start, t.end, spk) for t, _, spk in ann.itertracks(yield_label=True)]
    # True speaker count comes from the diarization itself, not from how many
    # distinct speakers happen to win segment-level overlap (coarse Whisper
    # segments can otherwise collapse the count).
    all_speakers = {spk for _, _, spk in turns}
    for seg in segments:
        best, best_ov = None, 0.0
        for ts, te, spk in turns:
            ov = max(0.0, min(seg["end"], te) - max(seg["start"], ts))
            if ov > best_ov:
                best_ov, best = ov, spk
        if best is not None:
            seg["speaker"] = best
            if seg.get("words"):
                for w in seg["words"]:
                    w["speaker"] = best
    return len(all_speakers)


# --------------------------------------------------------------------------- #
# Core transcription
# --------------------------------------------------------------------------- #
def _do_transcribe(audio: np.ndarray, *, language: str | None, vad_filter: bool,
                   word_timestamps: bool, diarize: bool, diarize_model: str,
                   cancel: threading.Event) -> dict:
    _, batched = _get_model()

    def collect(seg_iter) -> list[dict]:
        out: list[dict] = []
        for seg in seg_iter:
            if cancel.is_set():
                raise Cancelled()
            item = {"start": float(seg.start), "end": float(seg.end), "text": seg.text}
            if word_timestamps and seg.words:
                item["words"] = [{"start": float(w.start), "end": float(w.end), "word": w.word}
                                 for w in seg.words]
            out.append(item)
        return out

    # Anti-repetition / anti-hallucination safeguards. The batched pipeline is fast but
    # otherwise omits these, so silence/noise in phone voice notes makes Whisper loop on
    # the last phrase. temperature[] enables the fallback re-decode, triggered when a
    # segment's compression_ratio or avg_logprob crosses the thresholds below; the
    # penalties directly discourage verbatim repeats.
    decode_opts = dict(
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=False,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
    )

    # Pass 1: raw transcript in the source language (auto-detected when language is None).
    seg_iter, info = batched.transcribe(
        audio,
        task="transcribe",
        language=language,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
        batch_size=BATCH_SIZE,
        **decode_opts,
    )
    segments = collect(seg_iter)

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    language_code = getattr(info, "language", None) or (language or "?")
    is_english = str(language_code).lower().startswith("en")

    # Pass 2: only when the source is NOT English, also produce an English translation.
    translated_segments: list[dict] | None = None
    if not is_english:
        if cancel.is_set():
            raise Cancelled()
        t_iter, _ = batched.transcribe(
            audio,
            task="translate",
            language=language_code if language_code != "?" else language,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            batch_size=BATCH_SIZE,
            **decode_opts,
        )
        translated_segments = collect(t_iter)

    diarized = False
    num_speakers = 0
    diarize_seconds = 0.0
    if diarize:
        if cancel.is_set():
            raise Cancelled()
        import torch
        t0 = time.monotonic()
        pipe = _get_diar_pipeline(diarize_model)
        wav = torch.from_numpy(audio).unsqueeze(0)  # (1, n) float32
        diarization = pipe({"waveform": wav, "sample_rate": SAMPLE_RATE})
        # Speaker turns are time-based, so the same annotation labels both passes.
        num_speakers = _assign_speakers(segments, diarization)
        if translated_segments is not None:
            _assign_speakers(translated_segments, diarization)
        diarize_seconds = round(time.monotonic() - t0, 1)
        diarized = True

    with _model_lock:
        global _last_used
        _last_used = time.monotonic()

    return {
        "segments": segments,
        "translated_segments": translated_segments,
        "duration": duration,
        "language": language_code,
        "is_english": is_english,
        "diarized": diarized,
        "num_speakers": num_speakers,
        "diarize_seconds": diarize_seconds,
    }


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(title="whisper-server")


@app.middleware("http")
async def _track_activity(request, call_next):
    global _last_activity
    _last_activity = time.monotonic()
    try:
        return await call_next(request)
    finally:
        _last_activity = time.monotonic()


@app.get("/health")
def health() -> dict:
    with _model_lock:
        loaded = _model is not None
        last = _last_used
    with _active_lock:
        active = _active
    idle_seconds = int(time.monotonic() - last) if (loaded and last is not None) else None
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "model_loaded": loaded,
        "idle_seconds": idle_seconds,
        "idle_timeout": IDLE_TIMEOUT,
        "gpu": _gpu_info(),
        "batch_size_config": BATCH_SIZE,
        "batch_size_now": BATCH_SIZE,
        "max_parallel_config": MAX_PARALLEL,
        "max_parallel_now": active,
        "diarize_available": bool(HF_TOKEN),
        "diarize_model": DIARIZE_MODEL,
    }


@app.post("/cancel")
def cancel(job_id: str = Form(...)) -> dict:
    with _jobs_lock:
        ev = _jobs.get(job_id)
    if ev is not None:
        ev.set()
        return {"cancelled": True, "job_id": job_id}
    return {"cancelled": False, "job_id": job_id, "reason": "unknown or finished"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    formats: str = Form("txt"),
    word_timestamps: str = Form("false"),
    vad_filter: str = Form("true"),
    diarize: str = Form("false"),
    job_id: str = Form(""),
    language: str = Form(""),
    diarize_model: str = Form(""),
):
    req_formats = [f.strip() for f in formats.split(",") if f.strip() in ALL_FORMATS] or ["txt"]
    want_words = word_timestamps.lower() == "true"
    want_vad = vad_filter.lower() == "true"
    want_diar = diarize.lower() == "true"
    lang = language.strip() or None
    diar_model = diarize_model.strip() or DIARIZE_MODEL

    if want_diar and not HF_TOKEN:
        return JSONResponse(status_code=400,
                            content={"error": "diarization requested but server has no HF_TOKEN"})

    cancel_ev = threading.Event()
    if job_id:
        with _jobs_lock:
            _jobs[job_id] = cancel_ev

    # Stream upload to a temp file, then decode once to 16k mono float32.
    suffix = Path(file.filename or "audio").suffix or ".bin"
    tmp = tempfile.NamedTemporaryFile(prefix="ws-up-", suffix=suffix, delete=False)
    try:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            tmp.write(chunk)
        tmp.flush()
        tmp.close()

        def work() -> dict:
            global _active
            with _sem:
                with _active_lock:
                    _active += 1
                try:
                    if cancel_ev.is_set():
                        raise Cancelled()
                    from faster_whisper.audio import decode_audio
                    audio = decode_audio(tmp.name, sampling_rate=SAMPLE_RATE)
                    return _do_transcribe(
                        audio, language=lang, vad_filter=want_vad,
                        word_timestamps=want_words, diarize=want_diar,
                        diarize_model=diar_model, cancel=cancel_ev,
                    )
                finally:
                    with _active_lock:
                        _active -= 1

        try:
            result = await run_in_threadpool(work)
        except Cancelled:
            return JSONResponse(status_code=499, content={"cancelled": True, "job_id": job_id})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"{type(e).__name__}: {e}"})

        fmt_out = render_formats(req_formats, result["segments"], result["language"],
                                 result["duration"], result["diarized"], want_words)

        # English source => raw transcript only. Non-English => also an English translation.
        fmt_translated = None
        if result.get("translated_segments") is not None:
            fmt_translated = render_formats(req_formats, result["translated_segments"], "en",
                                            result["duration"], result["diarized"], want_words)

        return {
            "duration": result["duration"],
            "language": result["language"],
            "is_english": result.get("is_english", True),
            "formats": fmt_out,
            "translated": fmt_translated is not None,
            "formats_translated": fmt_translated,
            "diarized": result["diarized"],
            "num_speakers": result["num_speakers"],
            "diarize_seconds": result["diarize_seconds"],
        }
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        if job_id:
            with _jobs_lock:
                _jobs.pop(job_id, None)


def _systemd_socket_fd() -> int | None:
    """Return the inherited socket fd if launched via systemd socket activation."""
    if os.environ.get("LISTEN_PID") == str(os.getpid()) and os.environ.get("LISTEN_FDS"):
        return 3  # SD_LISTEN_FDS_START
    return None


def main() -> None:
    global _last_activity
    import uvicorn

    fd = _systemd_socket_fd()
    cfg_kwargs = dict(app=app, log_level="info")
    if fd is not None:
        cfg_kwargs["fd"] = fd          # serve the socket systemd handed us
    else:
        cfg_kwargs["host"] = HOST       # standalone run (no socket activation)
        cfg_kwargs["port"] = PORT

    server = uvicorn.Server(uvicorn.Config(**cfg_kwargs))
    _last_activity = time.monotonic()   # start the idle clock at boot
    threading.Thread(target=_idle_exit_watchdog, args=(server,), daemon=True).start()
    server.run()


if __name__ == "__main__":
    main()
