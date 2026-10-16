# whisper-server

A self-hosted speech to text server for a home LAN. It runs **faster-whisper large-v3** on an NVIDIA GPU, exposes a small HTTP API on port 9000, and is **socket activated** by systemd, so it holds zero VRAM while idle and wakes on the first request.

Any machine on the network can send it audio or video and get back a transcript (txt, srt, vtt, json with word timestamps), an automatic English translation when the source is not English, and optional speaker labels.

This repo is a **replication kit**: the exact server code and configuration that run on the reference box, plus a step by step SOP for rebuilding it from a bare machine.

| Doc | What it covers |
|---|---|
| [SOP.md](SOP.md) | Full rebuild procedure, verification, maintenance, troubleshooting, and the list of things that do not work |
| [PLATFORMS.md](PLATFORMS.md) | Running the server or the clients on Windows (WSL2 or native) and macOS |
| [API-MODE.md](API-MODE.md) | Running the same server with no local GPU by calling a hosted speech API instead, with feature parity notes |

## Reference box

| Item | Value |
|---|---|
| OS | CachyOS (Arch based), kernel 7.0 |
| CPU / GPU | AMD Ryzen 5 PRO 3400GE, NVIDIA RTX 3060 12 GB |
| Driver | nvidia-open 595.71.05 |
| Python | 3.12 in a `uv` venv (system Python is 3.14, too new) |
| Key packages | faster-whisper 1.2.1, ctranslate2 4.7.2, pyannote.audio 4.0.4, torch 2.12.0, fastapi 0.136.3 |
| Models | `Systran/faster-whisper-large-v3`, `pyannote/speaker-diarization-community-1`, `Revai/reverb-diarization-v2` |

## Quick start

Assumes the NVIDIA driver, `uv` and `ffmpeg` are installed (SOP steps 1 and 2).

```bash
git clone https://github.com/crxssrazr93/whisper-server.git
cd whisper-server
bash install.sh                       # pinned versions, systemd socket, linger
$EDITOR ~/whisper-server/whisper-server.env   # add HF_TOKEN for diarization
bash ~/whisper-server/download-models.sh      # about 3.5 GB
curl -s http://localhost:9000/health | python3 -m json.tool
```

From another machine:

```bash
WHISPER_SERVER=http://<box-ip>:9000 python3 client/transcribe-remote.py clip.m4a
```

## Repo layout

| Path | Purpose |
|---|---|
| `install.sh` | One shot installer (copies files to `~/whisper-server`, builds venv, enables socket) |
| `server/server.py` | FastAPI server: transcription, translation, diarization, idle exit |
| `server/run.sh` | Launch wrapper (puts the pip NVIDIA libs on `LD_LIBRARY_PATH`) |
| `server/setup.sh` | Unpinned dependency install (fallback to the lock file) |
| `server/requirements.lock.txt` | Exact package set from the working box |
| `server/download-models.sh` | Pre downloads all model weights with progress bars |
| `server/whisper-server.env.example` | Config template |
| `systemd/` | `whisper-server.socket` and `whisper-server.service` user units |
| `client/` | CLI clients (`transcribe-remote.py`, `transcribe-remote-diarize.py`) |
| `desktop/` | File manager right click launchers (plain, diarize, Reverb diarize) |

No secrets live in this repo. The real `whisper-server.env` (which holds the Hugging Face token) is git ignored.
