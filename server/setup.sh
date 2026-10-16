#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/whisper-server"
source .venv/bin/activate
echo "=== $(date) starting install ==="
echo "--- torch (CUDA 12.4) ---"
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
echo "--- core server deps (PyPI) ---"
uv pip install "faster-whisper>=1.1.0" "fastapi>=0.110" "uvicorn[standard]>=0.29" python-multipart
echo "--- diarization deps (PyPI) ---"
uv pip install "pyannote.audio>=3.3"
echo "=== $(date) install DONE ==="
python - <<'PY'
import torch, ctranslate2, faster_whisper, fastapi
print("torch", torch.__version__, "cuda?", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("ctranslate2", ctranslate2.__version__, "fw", faster_whisper.__version__, "fastapi", fastapi.__version__)
PY
echo "=== verify done ==="
