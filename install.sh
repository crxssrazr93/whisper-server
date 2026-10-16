#!/usr/bin/env bash
# One-shot installer for whisper-server on a fresh NVIDIA Linux box.
# Copies server files into ~/whisper-server, builds the Python 3.12 venv from the
# pinned lock file, and installs + enables the systemd --user socket.
#
#   bash install.sh            # pinned, known-good versions (recommended)
#   bash install.sh --latest   # unpinned install via server/setup.sh
#
# Safe to re-run. It never overwrites an existing whisper-server.env.
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
DEST="$HOME/whisper-server"
MODE="${1:-pinned}"

command -v uv >/dev/null || { echo "uv not found. Install it first (see SOP.md step 2)."; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found. Install the NVIDIA driver first (SOP.md step 1)."; exit 1; }

mkdir -p "$DEST"
cp "$REPO"/server/{server.py,run.sh,setup.sh,download-models.sh,requirements.lock.txt,whisper-server.env.example} "$DEST/"
cp "$REPO"/client/*.py "$DEST/"
chmod +x "$DEST"/{run.sh,setup.sh,download-models.sh}

if [ ! -f "$DEST/whisper-server.env" ]; then
    cp "$DEST/whisper-server.env.example" "$DEST/whisper-server.env"
    chmod 600 "$DEST/whisper-server.env"
    echo ">> Created $DEST/whisper-server.env. Put your HF_TOKEN in it before using diarization."
fi

cd "$DEST"
[ -d .venv ] || uv venv --python 3.12 .venv
if [ "$MODE" = "--latest" ]; then
    bash setup.sh
else
    uv pip install --python .venv/bin/python -r requirements.lock.txt
fi

mkdir -p "$HOME/.config/systemd/user"
cp "$REPO"/systemd/whisper-server.{service,socket} "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable --now whisper-server.socket
loginctl enable-linger "$USER" || echo "!! enable-linger failed. Run: sudo loginctl enable-linger $USER"

echo
echo "Installed. Next: download models (SOP.md step 6), then: curl -s http://localhost:9000/health"
