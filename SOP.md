# SOP: Rebuild the LAN whisper-server

Standard operating procedure for recreating the transcription server from a bare machine. Follow the steps in order. Each step ends with a check; do not move on until it passes.

Time required: about 45 minutes plus model download time (about 3.5 GB).

## Contents

0. [Decide the box](#0-decide-the-box)
1. [NVIDIA driver](#1-nvidia-driver)
2. [Base tools](#2-base-tools)
3. [Install the server](#3-install-the-server)
4. [Configure](#4-configure)
5. [Hugging Face access for diarization](#5-hugging-face-access-for-diarization)
6. [Download models](#6-download-models)
7. [systemd socket activation](#7-systemd-socket-activation)
8. [Network: fixed IP and firewall](#8-network-fixed-ip-and-firewall)
9. [Verify end to end](#9-verify-end-to-end)
10. [Clients and right click launchers](#10-clients-and-right-click-launchers)
11. [How the server behaves](#11-how-the-server-behaves)
12. [Maintenance](#12-maintenance)
13. [Troubleshooting](#13-troubleshooting)
14. [Known dead ends](#14-known-dead-ends)
15. [Backup and restore checklist](#15-backup-and-restore-checklist)

## 0. Decide the box

| Requirement | Minimum | Reference box |
|---|---|---|
| GPU | NVIDIA with 6 GB free VRAM for large-v3 float16 | RTX 3060 12 GB |
| Disk | 10 GB (venv about 6 GB, models about 3.5 GB) | NVMe |
| OS | Any systemd Linux | CachyOS |
| Network | Reachable from the clients on TCP 9000 | Home LAN, Wi-Fi |

No GPU at all? See [API-MODE.md](API-MODE.md). The same server can call a hosted speech API instead of a local model.

Server on Windows or macOS, or clients on Windows or macOS? See [PLATFORMS.md](PLATFORMS.md). This SOP is the Linux path, which is the tested reference.

## 1. NVIDIA driver

Install the vendor driver for your distro.

```bash
# CachyOS: the hardware detection tool picks the right package
sudo chwd -a
# Plain Arch alternative
sudo pacman -S nvidia-open nvidia-utils
# Ubuntu / Debian alternative
sudo ubuntu-drivers install
```

Reboot, then check:

```bash
nvidia-smi        # must list the GPU and a driver version
```

You do **not** need the system CUDA toolkit or system cuDNN. The venv ships its own CUDA libraries as pip wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) and `run.sh` puts them on the library path.

## 2. Base tools

```bash
# uv (Python and venv manager), installs to ~/.local/bin
curl -LsSf https://astral.sh/uv/install.sh | sh
# ffmpeg is needed on every CLIENT (it transcodes before upload); git on the server
sudo pacman -S --needed ffmpeg git          # or: sudo apt install ffmpeg git
```

Check: `uv --version` and `ffmpeg -version` both print a version.

## 3. Install the server

```bash
git clone https://github.com/crxssrazr93/whisper-server.git
cd whisper-server
bash install.sh
```

`install.sh` does the following, so you can do it by hand if it fails part way:

1. Copies `server/*` and `client/*.py` into `~/whisper-server` (the systemd units and scripts expect exactly this path).
2. Creates `~/whisper-server/whisper-server.env` from the example with mode 600, only if it does not already exist.
3. Builds `.venv` with **Python 3.12** (`uv venv --python 3.12`). uv downloads 3.12 by itself if the system does not have it.
4. Installs the exact package set from `requirements.lock.txt`. Use `bash install.sh --latest` to run the unpinned `setup.sh` instead.
5. Copies the two systemd units, enables the socket, and turns on linger.

Check:

```bash
cd ~/whisper-server && source .venv/bin/activate
python -c "import torch, ctranslate2, faster_whisper; print(torch.cuda.is_available(), ctranslate2.get_cuda_device_count())"
```

Expected output: `True 1`.

## 4. Configure

Edit `~/whisper-server/whisper-server.env`:

| Variable | Reference value | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | Must be the full large-v3. `large-v3-turbo` cannot translate. |
| `WHISPER_DEVICE` | `cuda` | Inference device |
| `WHISPER_COMPUTE_TYPE` | `float16` | Use `int8_float16` on a smaller GPU |
| `WHISPER_IDLE_TIMEOUT` | `120` | Seconds idle before the process exits and frees VRAM. `0` keeps it loaded forever. |
| `WHISPER_BATCH_SIZE` | `8` | Batched inference size. 8 to 16 fits 12 GB. |
| `WHISPER_MAX_PARALLEL` | `3` | Concurrent jobs. Also passed to the model as `num_workers`. |
| `WHISPER_DIARIZE_MODEL` | `pyannote/speaker-diarization-community-1` | Default diarization pipeline |
| `WHISPER_HOST` / `WHISPER_PORT` | `0.0.0.0` / `9000` | Only used when run outside systemd |
| `HF_TOKEN` | your token | Needed for the gated diarization models |
| `HF_HUB_DISABLE_XET` | `1` | Keep this. See dead ends. |

Keep the file at mode 600. It is the only place the token lives.

## 5. Hugging Face access for diarization

Skip this if you never use speaker labels.

1. Create a read token at https://huggingface.co/settings/tokens and put it in `HF_TOKEN`.
2. While logged in to that same account, open each page and accept the terms:
   * https://huggingface.co/pyannote/speaker-diarization-community-1
   * https://huggingface.co/Revai/reverb-diarization-v2

A 401 or 403 during download means step 2 was not done for the account that owns the token.

## 6. Download models

Download before the first request. Otherwise the first transcription stalls for minutes while weights download inside the service.

```bash
bash ~/whisper-server/download-models.sh
```

It fetches `Systran/faster-whisper-large-v3` (about 3 GB) and the two diarization pipelines into `~/.cache/huggingface/hub`, with progress bars. Re-running skips files already cached.

Check: `ls ~/.cache/huggingface/hub` shows `models--Systran--faster-whisper-large-v3`.

## 7. systemd socket activation

`install.sh` already did this. By hand:

```bash
mkdir -p ~/.config/systemd/user
cp systemd/whisper-server.{socket,service} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now whisper-server.socket
sudo loginctl enable-linger "$USER"
```

Rules:

* Enable the **socket**, never the service. The service has no `[Install]` section on purpose.
* Linger is required, or the user units stop when you log out and the box serves nothing after a reboot until someone logs in.
* `Restart=on-failure` (not `always`), because a clean idle exit returns code 0 and must not restart. The socket relaunches it on the next request.

Check:

```bash
systemctl --user status whisper-server.socket   # active (listening)
loginctl show-user "$USER" -p Linger            # Linger=yes
```

## 8. Network: fixed IP and firewall

* Give the box a **DHCP reservation** on the router so its IP never changes. The clients default to that IP.
* If a firewall is active, allow TCP 9000 from the LAN only:

```bash
sudo ufw allow from 192.168.0.0/16 to any port 9000 proto tcp     # ufw
sudo firewall-cmd --permanent --add-port=9000/tcp && sudo firewall-cmd --reload   # firewalld
```

* Do **not** port forward 9000 to the internet. The API has no authentication.

Check from another machine: `curl -s http://<box-ip>:9000/health`.

## 9. Verify end to end

Run all four.

```bash
# 1. Health (first call cold starts the process in about 2 s)
curl -s http://<box-ip>:9000/health | python3 -m json.tool
#    expect "gpu": {"device": "NVIDIA ...", ...} and "model": "large-v3"

# 2. English file produces one transcript
WHISPER_SERVER=http://<box-ip>:9000 python3 client/transcribe-remote.py english.m4a
#    expect english.txt and NO english.translated.txt

# 3. Non English file produces raw plus English translation
WHISPER_SERVER=http://<box-ip>:9000 python3 client/transcribe-remote.py spanish.m4a
#    expect spanish.txt (Spanish) and spanish.translated.txt (English)

# 4. Diarization
WHISPER_SERVER=http://<box-ip>:9000 python3 client/transcribe-remote-diarize.py --diarize interview.mp3
#    expect lines prefixed with SPEAKER_00, SPEAKER_01 ...
```

Then confirm idle exit: wait `WHISPER_IDLE_TIMEOUT` seconds and run `systemctl --user is-active whisper-server.service`. It should print `inactive`, and `nvidia-smi` should show no python process. The socket stays `listening`.

## 10. Clients and right click launchers

The clients are single file Python scripts with no dependencies beyond the standard library and `ffmpeg`. They transcode to 64 kbps mono 16 kHz MP3 before upload (about 380 times smaller than raw video, near lossless for speech), upload, and write outputs next to each source file.

```bash
mkdir -p ~/whisper-server
cp client/*.py ~/whisper-server/
# Point at the server. Either export this, or edit DEFAULT_SERVER at the top of each client.
set -Ux WHISPER_SERVER http://<box-ip>:9000        # fish
export WHISPER_SERVER=http://<box-ip>:9000         # bash/zsh, put in your rc file
```

Right click launchers (Dolphin, Nautilus, Nemo, any XDG file manager). They accept files or folders (recursive):

```bash
mkdir -p ~/.local/share/applications
cp desktop/*.desktop ~/.local/share/applications/
sed -i "s#/home/USER#$HOME#" ~/.local/share/applications/Transcribe-Remote*.desktop
update-desktop-database ~/.local/share/applications 2>/dev/null || true
```

| Launcher | Runs |
|---|---|
| Transcribe (Remote) | plain transcription, plus translation when non English |
| Transcribe with Speakers (Remote) | adds `--diarize` (pyannote Community-1) |
| Transcribe with Speakers, Reverb (Remote) | diarization with `Revai/reverb-diarization-v2` |

Other tools can use the same API. Anything that speaks the contract below works, for example the `transcribe.py` script in [video-ultracut](https://github.com/crxssrazr93/video-ultracut), which asks for `formats=json,srt` with `word_timestamps=true` and falls back to in process faster-whisper when the server is unreachable.

## 11. How the server behaves

### HTTP contract

Clients depend on these exact shapes. Keep them stable.

* `GET /health` returns `{model, device, compute_type, model_loaded, idle_seconds, idle_timeout, gpu, batch_size_config, batch_size_now, max_parallel_config, max_parallel_now, diarize_available, diarize_model}`. `gpu` **must be an object** `{device, total_mb, free_mb, used_mb}`, because clients call `gpu.get("device")`.
* `POST /transcribe` is multipart. File field `file`. Form fields: `formats` (csv of `txt,srt,vtt,json`), `word_timestamps`, `vad_filter`, `diarize` (each `true`/`false`), `job_id` (hex), optional `language`, optional `diarize_model`. Returns `{duration, language, is_english, formats{fmt: text}, translated, formats_translated{fmt: text} or null, diarized, num_speakers, diarize_seconds}`.
* `POST /cancel` (form field `job_id`) cancels that job. The cancelled request returns HTTP **499** `{"cancelled": true}`.

The `json` format holds `{language, duration, segments: [{start, end, text, speaker?, words: [{start, end, word, speaker?}]}]}`.

### Pipeline per request

1. Pass 1, `task="transcribe"`, language auto detected. This is always the raw transcript in the spoken language.
2. If the detected language is not English, pass 2, `task="translate"`, produces English.
3. If `diarize=true`, pyannote runs once and its labels are applied to both passes. `num_speakers` comes from the diarization's own label set, not from how many speakers won a segment.

### Anti repetition settings

The batched pipeline (`BatchedInferencePipeline`, about 10 times realtime) skips Whisper's repetition safeguards by default, which made noisy phone voice notes loop the last phrase. Both passes therefore pass:

```
temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
compression_ratio_threshold=2.4
log_prob_threshold=-1.0
no_speech_threshold=0.6
condition_on_previous_text=False
repetition_penalty=1.1
no_repeat_ngram_size=3
```

Cost: speed dropped from about 11.8x to 9.1x realtime. If loops return, raise `repetition_penalty` or lower `compression_ratio_threshold`, or switch to sequential `model.transcribe`.

### Cold start and idle exit

Only fastapi, uvicorn and numpy load at process start, so the socket answers in about 2 seconds. faster-whisper, torch and pyannote import lazily, and the model loads on the first `/transcribe`. A watchdog thread sets `uvicorn.Server.should_exit` after `WHISPER_IDLE_TIMEOUT` seconds without a request. Under systemd the server takes its listening socket from `LISTEN_FDS` (fd 3); run by hand it binds `WHISPER_HOST:WHISPER_PORT`.

## 12. Maintenance

| Task | Command |
|---|---|
| Live logs | `journalctl --user -u whisper-server -f` |
| Apply a `server.py` or env change | copy the file into `~/whisper-server`, then `systemctl --user restart whisper-server.socket` (also stops a running instance) |
| Change model | edit `WHISPER_MODEL`, `hf download` the new repo, restart the socket |
| Upgrade packages | `bash ~/whisper-server/setup.sh`, run section 9 again, then refresh the lock with `uv pip freeze --python .venv/bin/python > requirements.lock.txt` |
| Back up before edits | `cp server.py server.py.bak-$(date +%F)` |

## 13. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Could not load library libcudnn_ops.so` or `libcublas.so.12` | NVIDIA pip libs not on the library path | Start through `run.sh`, never `python server.py` directly |
| `/health` works, first `/transcribe` hangs for minutes | Model downloading inside the service | Run `download-models.sh` first |
| Download stuck at 0% | Hugging Face Xet backend stall | `HF_HUB_DISABLE_XET=1` |
| Diarization 401 / 403 | Terms not accepted, or wrong token | Section 5 |
| `AttributeError` on the diarization result | pyannote 4.x returns `DiarizeOutput` | Annotation is at `.speaker_diarization` (already handled) |
| `unexpected keyword use_auth_token` | pyannote 4.x renamed it | Use `token=` (already handled, with 3.x fallback) |
| Translation missing for non English audio | Running `large-v3-turbo` | Use `large-v3` |
| Server gone after reboot until login | Linger off | `sudo loginctl enable-linger $USER` |
| Service restarts in a loop after idle | `Restart=always` | Must be `on-failure` |
| Output repeats one phrase many times | Batched pipeline without safeguards | Section 11 settings |
| Clients time out from the laptop, fine locally | Firewall, or the box moved IP | Section 8 |
| CUDA OOM | Batch too large, or another GPU job running (llama.cpp etc.) | Lower `WHISPER_BATCH_SIZE`, or stop the other job |

## 14. Known dead ends

Recorded so nobody walks them again.

* **System Python 3.14.** Several dependencies had no wheels. Pin the venv to 3.12.
* **Re-pinning torch after installing pyannote.** pyannote.audio 4.0.4 pulls torch up to 2.12 (CUDA 13 build). It looks like it should break CTranslate2, but the venv keeps `nvidia-cublas-cu12` (`libcublas.so.12`), so GPU inference still works. Do not fight it; the lock file captures the working combination.
* **`large-v3-turbo` as default.** Faster, but transcription only. It silently cannot translate.
* **Xet downloads.** `hf` 1.x with Xet stalled at 0% on this network every time. Classic HTTP is reliable.
* **Counting speakers from segment overlap.** Coarse Whisper segments collapsed the count. Count labels in the diarization itself.
* **An always on daemon.** Held about 4 GB of VRAM all day, which starved local LLMs. Socket activation plus idle exit replaced it.
* **Assuming the server source survives a reinstall.** The original was lost when the box was reformatted, and had to be rebuilt from the client contract. That is why this repo exists. Keep it updated after every server change.

## 15. Backup and restore checklist

What must survive to rebuild in under an hour:

* [ ] This repo, with `server/server.py` matching the live box (`md5sum` both)
* [ ] `whisper-server.env` values (the token can be regenerated, the rest is in the example)
* [ ] The box's DHCP reservation on the router
* [ ] Optional: `~/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3` and the two diarization folders, to skip a 3.5 GB download

Restore order: section 1, 2, 3, 4, 5, 6, 9.
