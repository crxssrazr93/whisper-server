#!/usr/bin/env bash
# Pre-download all model weights for whisper-server (run this yourself when ready).
#   bash ~/whisper-server/download-models.sh
#
# Uses the `hf` CLI so you get live per-file progress bars (speed + ETA).
# Downloads into the shared HF cache (~/.cache/huggingface/hub) that the
# server reads from.
#
#   1. Systran/faster-whisper-large-v3                  ~3 GB (default model, supports translate)
#   2. pyannote/speaker-diarization-community-1 (gated)  ~hundreds of MB
#   3. Revai/reverb-diarization-v2 (gated)               ~hundreds of MB
#
# The two diarization repos are GATED: with your HF account, accept their terms first:
#   https://huggingface.co/pyannote/speaker-diarization-community-1
#   https://huggingface.co/Revai/reverb-diarization-v2
set -uo pipefail
cd "$HOME/whisper-server"
# shellcheck disable=SC1091
source .venv/bin/activate
# Load HF_TOKEN from the service env file so `hf` can fetch gated repos.
set -a; source ./whisper-server.env; set +a
export HF_TOKEN
# The Xet storage backend (hf_xet) stalls at 0% on this network; force classic
# HTTP/LFS downloads which are reliable here.
export HF_HUB_DISABLE_XET=1

dl () {  # repo, label
  echo
  echo "==================================================================="
  echo "==> $2"
  echo "==================================================================="
  if hf download "$1"; then
    echo "    OK: $1"
  else
    echo "    !! FAILED: $1"
    echo "       If gated (401/403): accept the terms on huggingface.co with"
    echo "       this token's account, then re-run (cached files are skipped)."
  fi
}

dl "Systran/faster-whisper-large-v3"               "[1/3] Whisper large-v3"
dl "pyannote/speaker-diarization-community-1"     "[2/3] pyannote speaker-diarization-community-1 (gated)"
dl "Revai/reverb-diarization-v2"                  "[3/3] Revai reverb-diarization-v2 (gated)"

echo
echo "DONE"
