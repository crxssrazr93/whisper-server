#!/usr/bin/env bash
# Launch wrapper for whisper-server.
# Resolves the pip-installed NVIDIA lib dirs (cuDNN/cuBLAS/etc) so CTranslate2
# can dlopen them, then execs the server inside the venv.
set -euo pipefail
cd "$HOME/whisper-server"
# shellcheck disable=SC1091
source .venv/bin/activate

# Build LD_LIBRARY_PATH from every nvidia/*/lib dir in the venv (version-agnostic).
NV_LIBS="$(python - <<'PY'
import os, glob
try:
    import nvidia
    base = os.path.dirname(nvidia.__file__)
    print(":".join(sorted(glob.glob(os.path.join(base, "*", "lib")))))
except Exception:
    print("")
PY
)"
if [ -n "$NV_LIBS" ]; then
    export LD_LIBRARY_PATH="${NV_LIBS}:${LD_LIBRARY_PATH:-}"
fi

exec python server.py
