# Windows and macOS

The tested reference is the Linux build in [SOP.md](SOP.md). This page covers the other two platforms, for the **server** and for the **clients** separately. Read the status column before relying on a path.

| Role | Platform | Path | Status |
|---|---|---|---|
| Server | Windows + NVIDIA | **WSL2** (recommended) | Same code as Linux, unchanged. Not yet run on the reference setup. |
| Server | Windows + NVIDIA | Native Python | Works in principle, needs manual CUDA DLL setup. Not yet tested. |
| Server | macOS (Apple Silicon or Intel) | Native Python, CPU | Works with no code changes, but slow. Not yet tested. |
| Server | macOS | MLX engine, or API mode | Fast, needs an engine adapter. See the end of this page and [API-MODE.md](API-MODE.md). |
| Client | Windows | Python + ffmpeg | Clients already handle Windows (`msvcrt` key press, no Linux only calls). |
| Client | macOS | Python + ffmpeg | Same scripts, no changes. |

What the server code already handles on every platform:

* No systemd: when `LISTEN_FDS` is absent, `server.py` binds `WHISPER_HOST:WHISPER_PORT` itself.
* No GPU: `/health` returns `"gpu": {"device": "cpu"}` and clients print it without VRAM numbers.
* Audio decoding uses faster-whisper's bundled PyAV, so the server needs no system ffmpeg. Diarization receives an in memory waveform, so pyannote needs no ffmpeg either.

What is Linux only: `run.sh` (bash, NVIDIA lib discovery), the systemd units, socket activation, `requirements.lock.txt` (it pins Linux CUDA wheels), and the `.desktop` launchers.

> **Idle exit rule.** Outside systemd nothing relaunches the process after it exits. On native Windows and macOS set `WHISPER_IDLE_TIMEOUT=0` so it stays up, unless you also set up an on demand launcher.

## Windows server, option A: WSL2 (recommended)

WSL2 runs the Linux build unchanged, including systemd and socket activation, with full CUDA through the Windows driver.

1. **Driver on Windows only.** Install the normal NVIDIA Game Ready or Studio driver on Windows. Do **not** install any NVIDIA driver inside WSL. Inside WSL, `nvidia-smi` should already work.
2. **Install WSL and Ubuntu** (PowerShell as admin):
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
3. **Enable systemd** inside Ubuntu, then restart WSL:
   ```bash
   sudo tee /etc/wsl.conf >/dev/null <<'EOF'
   [boot]
   systemd=true
   EOF
   ```
   ```powershell
   wsl --shutdown
   ```
4. **Let LAN clients reach it.** Create `%UserProfile%\.wslconfig` (Windows 11 22H2 or newer):
   ```ini
   [wsl2]
   networkingMode=mirrored
   vmIdleTimeout=-1
   ```
   Then allow inbound traffic to the WSL VM and open the port (PowerShell as admin):
   ```powershell
   Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
   New-NetFirewallRule -DisplayName "whisper-server 9000" -Direction Inbound -Protocol TCP -LocalPort 9000 -Action Allow -Profile Private
   wsl --shutdown
   ```
   With mirrored networking the server answers on the **Windows** machine's LAN IP.
   On Windows 10, mirrored mode does not exist; use `netsh interface portproxy add v4tov4 listenport=9000 connectport=9000 connectaddress=<wsl-ip>` instead, and redo it whenever the WSL IP changes.
5. **Follow SOP.md from step 2** inside Ubuntu (`sudo apt install ffmpeg git`, uv, `install.sh`, models, verify). Skip SOP step 1.
6. **Start WSL at boot.** WSL only runs while something holds it open. In Task Scheduler create a task, trigger "At log on" (or "At startup" with "Run whether user is logged on or not"), action:
   ```
   Program:   C:\Windows\System32\wsl.exe
   Arguments: -d Ubuntu-24.04 --exec sleep infinity
   ```
   Systemd then brings up `whisper-server.socket` by itself.

Check from another machine: `curl http://<windows-lan-ip>:9000/health`.

## Windows server, option B: native Python

Use this only if WSL2 is not an option. CTranslate2 has Windows CUDA wheels, but it needs the cuBLAS 12 and cuDNN 9 DLLs on `PATH`, which `run.sh` would normally handle.

1. Install the NVIDIA driver, then the tools (PowerShell):
   ```powershell
   winget install astral-sh.uv Git.Git
   ```
2. Clone and lay out the folder:
   ```powershell
   git clone https://github.com/crxssrazr93/whisper-server.git
   mkdir $HOME\whisper-server
   copy whisper-server\server\* $HOME\whisper-server\
   copy whisper-server\client\*.py $HOME\whisper-server\
   cd $HOME\whisper-server
   copy whisper-server.env.example whisper-server.env
   ```
3. Build the venv. Do **not** use `requirements.lock.txt` (Linux wheels). The default PyPI torch on Windows is CPU only, so install the CUDA build of torch last:
   ```powershell
   uv venv --python 3.12 .venv
   uv pip install --python .venv\Scripts\python.exe faster-whisper fastapi "uvicorn[standard]" python-multipart pyannote.audio nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"
   uv pip install --python .venv\Scripts\python.exe --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```
   torch only matters for diarization on the GPU and for the VRAM figures in `/health`. Whisper itself runs through CTranslate2.
4. Save this as `$HOME\whisper-server\run.ps1`. It is the Windows equivalent of `run.sh` plus the systemd `EnvironmentFile`:
   ```powershell
   Set-Location $PSScriptRoot
   # Load whisper-server.env into the process environment
   Get-Content .\whisper-server.env | Where-Object { $_ -match '^\s*[A-Z_]+=' } | ForEach-Object {
       $k, $v = $_ -split '=', 2
       [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), 'Process')
   }
   # Put the pip shipped CUDA DLLs on PATH so CTranslate2 can load them
   $nv = Get-ChildItem .\.venv\Lib\site-packages\nvidia -Directory -ErrorAction SilentlyContinue |
         ForEach-Object { Join-Path $_.FullName 'bin' } | Where-Object { Test-Path $_ }
   $env:PATH = (($nv + $env:PATH) -join ';')
   & .\.venv\Scripts\python.exe server.py
   ```
5. In `whisper-server.env` set `WHISPER_IDLE_TIMEOUT=0`, add `HF_TOKEN`, and keep `HF_HUB_DISABLE_XET=1`.
6. Download models:
   ```powershell
   $env:HF_HUB_DISABLE_XET=1; $env:HF_TOKEN="hf_..."
   .venv\Scripts\hf.exe download Systran/faster-whisper-large-v3
   .venv\Scripts\hf.exe download pyannote/speaker-diarization-community-1
   .venv\Scripts\hf.exe download Revai/reverb-diarization-v2
   ```
7. Run at boot with Task Scheduler: trigger "At startup", "Run whether user is logged on or not", action `powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\Users\<you>\whisper-server\run.ps1`. Alternatively wrap the same command as a service with NSSM.
8. Open the firewall:
   ```powershell
   New-NetFirewallRule -DisplayName "whisper-server 9000" -Direction Inbound -Protocol TCP -LocalPort 9000 -Action Allow -Profile Private
   ```

Check: `.venv\Scripts\python.exe -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"` prints `1`, then run SOP section 9 from a client.

If CTranslate2 still reports a missing `cublas64_12.dll` or `cudnn_ops64_9.dll`, the DLL folder is not on `PATH`. Run `Get-ChildItem -Recurse .venv -Filter cudnn*.dll` to find it and add that folder.

## macOS server

There is no CUDA on macOS, and CTranslate2 has no Metal backend, so faster-whisper runs on the **CPU**. It works with zero code changes. Expect large-v3 to be several times slower than on the RTX 3060; Apple Silicon with int8 is the usable case, Intel Macs are not.

1. Tools:
   ```bash
   xcode-select --install
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   brew install uv git
   ```
2. Lay out the folder and build the venv (unpinned; the lock file is Linux only, and `setup.sh` points torch at a CUDA index that has no macOS wheels):
   ```bash
   git clone https://github.com/crxssrazr93/whisper-server.git
   mkdir -p ~/whisper-server
   cp whisper-server/server/* whisper-server/client/*.py ~/whisper-server/
   cd ~/whisper-server
   cp whisper-server.env.example whisper-server.env && chmod 600 whisper-server.env
   uv venv --python 3.12 .venv
   uv pip install --python .venv/bin/python torch torchaudio faster-whisper fastapi "uvicorn[standard]" python-multipart pyannote.audio
   ```
3. Edit `whisper-server.env`:
   ```
   WHISPER_DEVICE=cpu
   WHISPER_COMPUTE_TYPE=int8
   WHISPER_IDLE_TIMEOUT=0
   WHISPER_BATCH_SIZE=4
   WHISPER_MAX_PARALLEL=1
   ```
   For speed, `WHISPER_MODEL=medium` still supports translation (only the `turbo` variants cannot translate) at some accuracy cost.
4. Download models: `bash ~/whisper-server/download-models.sh` (works unchanged on macOS).
5. Run at login with launchd. Save as `~/Library/LaunchAgents/lan.whisper-server.plist`, replacing `YOU`:
   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
   <plist version="1.0">
   <dict>
     <key>Label</key><string>lan.whisper-server</string>
     <key>ProgramArguments</key>
     <array>
       <string>/bin/bash</string><string>-c</string>
       <string>cd /Users/YOU/whisper-server &amp;&amp; set -a &amp;&amp; source ./whisper-server.env &amp;&amp; set +a &amp;&amp; exec ./run.sh</string>
     </array>
     <key>RunAtLoad</key><true/>
     <key>KeepAlive</key><true/>
     <key>StandardOutPath</key><string>/Users/YOU/whisper-server/server.log</string>
     <key>StandardErrorPath</key><string>/Users/YOU/whisper-server/server.log</string>
   </dict>
   </plist>
   ```
   ```bash
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/lan.whisper-server.plist
   ```
   `run.sh` works on macOS as is: the NVIDIA lookup finds nothing and is skipped.
   launchd also supports socket activation, but it hands over sockets through `launch_activate_socket`, not `LISTEN_FDS`, so `server.py` would need a small change to use it. Keeping it always on is simpler.
6. Allow incoming connections for Python when macOS asks, or in System Settings, Network, Firewall.

### Faster on Apple Silicon: an MLX engine

For real speed on a Mac, run Whisper on the GPU through MLX (`mlx-whisper`, which supports language detection, the translate task and word timestamps) instead of CTranslate2. That is a change of engine inside `server.py`, not of the HTTP contract, and it uses exactly the same adapter seam described in [API-MODE.md](API-MODE.md#the-adapter-seam): replace the model load and the two transcribe passes, keep everything else. MLX does not ship the anti repetition options faster-whisper has, so check noisy voice notes for loops.

## Clients on Windows

1. Install Python and ffmpeg:
   ```powershell
   winget install Python.Python.3.12 Gyan.FFmpeg
   ```
2. Copy `client\transcribe-remote.py` and `client\transcribe-remote-diarize.py` to `%UserProfile%\whisper-server\`.
3. Point at the server once (new terminals pick it up):
   ```powershell
   setx WHISPER_SERVER http://<box-ip>:9000
   ```
4. Use it:
   ```powershell
   py $HOME\whisper-server\transcribe-remote.py "C:\path\clip.m4a"
   py $HOME\whisper-server\transcribe-remote-diarize.py --diarize "C:\path\interview.mp3"
   ```
5. Right click: Windows has no `.desktop` launchers, so use **Send to**. Press Win+R, open `shell:sendto`, and create a shortcut with this target:
   ```
   C:\Windows\py.exe "C:\Users\<you>\whisper-server\transcribe-remote.py"
   ```
   Name it "Transcribe (Remote)". Make a second one for diarization with `--diarize` appended after the script path, and a third with `--diarize --diarize-model Revai/reverb-diarization-v2`. Selected files or folders are appended as arguments, and the console window stays open at the end with "Press any key to close".

## Clients on macOS

1. `brew install python ffmpeg`
2. Copy the two client scripts to `~/whisper-server/`, and add `export WHISPER_SERVER=http://<box-ip>:9000` to `~/.zshrc`.
3. Use it: `python3 ~/whisper-server/transcribe-remote.py clip.m4a`
4. Right click in Finder: open **Automator**, new **Quick Action**, "Workflow receives current **files or folders** in **Finder**", add **Run Shell Script** with shell `/bin/zsh` and "Pass input: **as arguments**", and paste:
   ```zsh
   f="$(mktemp -d)/transcribe.command"
   {
     echo '#!/bin/zsh -l'
     echo 'export WHISPER_SERVER=http://<box-ip>:9000'
     printf 'python3 ~/whisper-server/transcribe-remote.py'
     for a in "$@"; do printf ' %q' "$a"; done
     echo
   } > "$f"
   chmod +x "$f"
   open "$f"
   ```
   Save it as "Transcribe (Remote)". It opens a Terminal window so you can watch progress. Duplicate it with `--diarize` (and the Reverb model flag) for the speaker label variants.
