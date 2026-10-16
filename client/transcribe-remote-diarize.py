#!/usr/bin/env python3
"""Remote batch transcription client for whisper-server.

Standard-library only. No pip install needed. Copy this single file to any
machine with Python 3.9+ and run it.

Usage:
    python3 transcribe-remote.py audio1.mp3 audio2.wav ...
    python3 transcribe-remote.py --server http://192.168.2.186:9000 *.mp3
    python3 transcribe-remote.py --formats txt,srt --language en podcast.m4a

The script:
  * Pings the server's /health endpoint and displays status.
  * Uploads each file, shows a live spinner with elapsed time.
  * Writes txt/srt/vtt/json outputs next to each source (or --output-dir).
  * Prints a summary and waits for a keypress before closing.

Set the WHISPER_SERVER env var to avoid passing --server every time.
"""
from __future__ import annotations

import argparse
import http.client
import itertools
import json
import mimetypes
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
from pathlib import Path

DEFAULT_SERVER = os.environ.get("WHISPER_SERVER", "http://192.168.2.186:9000")
#ALL_FORMATS = ["txt", "srt", "vtt", "json"]
ALL_FORMATS = ["txt"]

# Extensions that are already compressed speech-friendly audio. Everything else
# (video, wav, flac, etc.) gets re-encoded to 64 kbps mono 16 kHz MP3 before
# upload. Whisper runs at 16 kHz internally, so this is near-lossless for ASR.
AUDIO_PASSTHROUGH_EXTS = {".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wma", ".mka"}

# Extensions we'll pick up when a folder is dropped on the launcher.
MEDIA_EXTS = {
    # audio
    ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wma", ".mka",
    ".wav", ".flac", ".aiff", ".aif", ".ape", ".wv", ".amr",
    # video
    ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".wmv", ".flv",
    ".mpeg", ".mpg", ".mts", ".m2ts", ".ts", ".3gp", ".3g2", ".vob", ".ogv",
}


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    return True


USE_COLOR = _supports_color()
CSI = "\033["


def c(code: str, text: str) -> str:
    if not USE_COLOR:
        return text
    return f"{CSI}{code}m{text}{CSI}0m"


BOLD = "1"
DIM = "2"
RED = "31"
GREEN = "32"
YELLOW = "33"
BLUE = "34"
CYAN = "36"
GREY = "90"


def term_width(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except OSError:
        return default


def hr(char: str = "-", width: int | None = None) -> str:
    return char * (width or term_width())


def press_any_key(prompt: str = "Press any key to close...") -> None:
    sys.stdout.write("\n" + c(DIM, prompt) + "\n")
    sys.stdout.flush()
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return
    except Exception:
        pass
    try:
        import msvcrt  # type: ignore[import-not-found]

        msvcrt.getch()
        return
    except Exception:
        pass
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass


# ---------------------------------------------------------------------------
# Minimal HTTP (stdlib only)
# ---------------------------------------------------------------------------

def _parse_url(url: str) -> tuple[str, str, int]:
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https"):
        raise ValueError(f"bad URL scheme: {u.scheme!r} (use http or https)")
    if not u.hostname:
        raise ValueError(f"bad URL host: {url!r}")
    port = u.port or (443 if u.scheme == "https" else 80)
    return u.scheme, u.hostname, port


def _connect(scheme: str, host: str, port: int, timeout: float | None) -> http.client.HTTPConnection:
    if scheme == "https":
        return http.client.HTTPSConnection(host, port, timeout=timeout)
    return http.client.HTTPConnection(host, port, timeout=timeout)


def http_get_json(url: str, path: str, timeout: float = 10.0) -> dict:
    scheme, host, port = _parse_url(url)
    conn = _connect(scheme, host, port, timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {body[:200].decode(errors='replace')}")
        return json.loads(body)
    finally:
        conn.close()


def http_post_json(url: str, path: str, timeout: float = 10.0) -> dict:
    scheme, host, port = _parse_url(url)
    conn = _connect(scheme, host, port, timeout)
    try:
        conn.request("POST", path)
        resp = conn.getresponse()
        body = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {body[:200].decode(errors='replace')}")
        return json.loads(body)
    finally:
        conn.close()


def http_post_form(url: str, path: str, fields: dict[str, str], timeout: float = 5.0) -> dict:
    scheme, host, port = _parse_url(url)
    body = urllib.parse.urlencode(fields).encode("utf-8")
    conn = _connect(scheme, host, port, timeout)
    try:
        conn.request("POST", path, body=body, headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Content-Length": str(len(body)),
            "Accept": "application/json",
        })
        resp = conn.getresponse()
        data = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {data[:200].decode(errors='replace')}")
        return json.loads(data) if data else {}
    finally:
        conn.close()


def http_post_multipart_file(
    url: str,
    path: str,
    fields: dict[str, str],
    file_path: Path,
) -> dict:
    """Stream a file upload as multipart/form-data without loading it into memory."""
    scheme, host, port = _parse_url(url)
    filename = file_path.name
    file_size = file_path.stat().st_size
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    boundary = f"----WS{uuid.uuid4().hex}"

    pre = ""
    for k, v in fields.items():
        pre += f"--{boundary}\r\n"
        pre += f'Content-Disposition: form-data; name="{k}"\r\n\r\n'
        pre += f"{v}\r\n"
    file_header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    )
    trailer = f"\r\n--{boundary}--\r\n"

    pre_b = pre.encode("utf-8")
    head_b = file_header.encode("utf-8")
    trail_b = trailer.encode("utf-8")
    content_length = len(pre_b) + len(head_b) + file_size + len(trail_b)

    conn = _connect(scheme, host, port, timeout=None)
    try:
        conn.putrequest("POST", path, skip_host=False, skip_accept_encoding=True)
        conn.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        conn.putheader("Content-Length", str(content_length))
        conn.putheader("Accept", "application/json")
        conn.endheaders()
        conn.send(pre_b)
        conn.send(head_b)
        with file_path.open("rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                conn.send(chunk)
        conn.send(trail_b)

        resp = conn.getresponse()
        body = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {body[:400].decode(errors='replace')}")
        return json.loads(body)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _prep_upload(src: Path) -> tuple[Path, Path | None, str]:
    """Return (upload_path, temp_dir_to_cleanup, note).

    If the source is already a compressed speech audio format, upload the
    original. Otherwise, if ffmpeg is available, re-encode to 64 kbps mono
    16 kHz MP3 in a temp dir (same basename, new extension) and return that
    path plus the temp dir to clean up later. If ffmpeg is missing or the
    conversion fails, fall back to uploading the original.
    """
    ext = src.suffix.lower()
    if ext in AUDIO_PASSTHROUGH_EXTS:
        return src, None, ""

    if not shutil.which("ffmpeg"):
        return src, None, "no ffmpeg on client, uploading original"

    tmp_dir = Path(tempfile.mkdtemp(prefix="ws-"))
    out_path = tmp_dir / f"{src.stem}.mp3"
    orig_size = src.stat().st_size
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(src),
                "-vn",                     # drop any video streams
                "-ac", "1",                # mono
                "-ar", "16000",            # 16 kHz, matches Whisper's native rate
                "-b:a", "64k",
                "-codec:a", "libmp3lame",
                str(out_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return src, None, "ffmpeg missing, uploading original"
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or b"").decode(errors="replace").strip().splitlines()[-1:] or ["unknown"]
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return src, None, f"ffmpeg failed ({msg[0][:80]}), uploading original"

    new_size = out_path.stat().st_size
    ratio = orig_size / new_size if new_size else 1.0
    note = f"compressed {_fmt_bytes(orig_size)} -> {_fmt_bytes(new_size)} ({ratio:.0f}x smaller)"
    return out_path, tmp_dir, note


def print_header(server_url: str, health: dict) -> None:
    width = min(term_width(), 80)
    title = " whisper-server "
    bar = "=" * ((width - len(title)) // 2)
    print(c(BOLD + ";" + CYAN, bar + title + bar))

    model = health.get("model", "?")
    device = health.get("device", "?")
    compute = health.get("compute_type", "?")
    model_loaded = bool(health.get("model_loaded"))
    idle = health.get("idle_seconds")
    gpu = health.get("gpu") or {}
    gpu_name = gpu.get("device", "unknown")
    total = gpu.get("total_mb")
    used = gpu.get("used_mb")
    free = gpu.get("free_mb")

    state_txt = c(GREEN, "hot") if model_loaded else c(YELLOW, "cold (will warm up on first file)")
    idle_txt = f" idle {idle}s" if (model_loaded and idle is not None) else ""

    def row(k: str, v: str) -> None:
        print(f"  {c(DIM, k.ljust(9))} {v}")

    row("URL:", server_url)
    row("Model:", model)
    row("Device:", f"{gpu_name} ({device}, {compute})")
    if total is not None:
        row("VRAM:", f"{free} MB free / {total} MB total  ({used} MB used)")
    row("State:", state_txt + idle_txt)
    print(c(CYAN, "=" * width))
    print()


def print_summary(ok: int, total: int, failed: list[tuple[Path, str]], audio_s: float, wall_s: float, out_dirs: set[Path]) -> None:
    width = min(term_width(), 80)
    print()
    print(c(CYAN, "-" * width))
    tag = c(GREEN, "OK") if ok == total else c(YELLOW if ok > 0 else RED, "DONE" if ok > 0 else "FAIL")
    print(f"  {tag}  {ok}/{total} files transcribed", end="")
    if wall_s > 0 and audio_s > 0:
        print(f"  ({audio_s:.1f}s audio in {wall_s:.1f}s, {audio_s / wall_s:.1f}x realtime)")
    else:
        print()
    if out_dirs:
        loc = ", ".join(str(p) for p in sorted(out_dirs))
        print(f"  {c(DIM, 'Outputs:')} {loc}")
    if failed:
        print(c(RED, "  Failures:"))
        for src, msg in failed:
            print(f"    {src.name}: {msg}")
    print(c(CYAN, "-" * width))


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------

class Spinner:
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start = 0.0

    def _run(self) -> None:
        frames = itertools.cycle(self.FRAMES)
        while not self._stop.is_set():
            elapsed = time.monotonic() - self._start
            sys.stdout.write(f"\r{self.prefix} {c(CYAN, next(frames))} {elapsed:5.1f}s ")
            sys.stdout.flush()
            self._stop.wait(0.1)

    def __enter__(self) -> "Spinner":
        self._start = time.monotonic()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join()
        sys.stdout.write("\r" + CSI + "2K")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_formats(value: str) -> list[str]:
    if value.strip().lower() in {"all", "*"}:
        return list(ALL_FORMATS)
    out = [f.strip().lower() for f in value.split(",") if f.strip()]
    unknown = [f for f in out if f not in ALL_FORMATS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown formats: {unknown}; allowed: {ALL_FORMATS}")
    return out


def _print_status(server_url: str, health: dict) -> None:
    width = min(term_width(), 80)
    gpu = health.get("gpu") or {}
    loaded = bool(health.get("model_loaded"))
    print(c(BOLD + ";" + CYAN, "=" * width))
    print(f"  {c(DIM, 'URL:'.ljust(9))} {server_url}")
    print(f"  {c(DIM, 'Model:'.ljust(9))} {health.get('model', '?')}")
    print(f"  {c(DIM, 'Device:'.ljust(9))} {gpu.get('device', '?')} ({health.get('device', '?')}, {health.get('compute_type', '?')})")
    if "total_mb" in gpu:
        print(f"  {c(DIM, 'VRAM:'.ljust(9))} {gpu.get('free_mb')} MB free / {gpu.get('total_mb')} MB total  ({gpu.get('used_mb')} MB used)")
    if loaded:
        idle = int(health.get("idle_seconds") or 0)
        timeout = int(health.get("idle_timeout") or 0)
        remaining = max(0, timeout - idle)
        mm, ss = divmod(remaining, 60)
        print(f"  {c(DIM, 'State:'.ljust(9))} {c(GREEN, 'loaded')}  idle {idle}s, unloads in {c(BOLD, f'{mm}m {ss}s')}")
    else:
        print(f"  {c(DIM, 'State:'.ljust(9))} {c(YELLOW, 'unloaded')}  (will warm up on next /transcribe)")

    cfg = health.get("batch_size_config")
    now = health.get("batch_size_now")
    if cfg is not None and now is not None:
        label = "Batch:"
        if str(cfg) == "auto":
            print(f"  {c(DIM, label.ljust(9))} auto (would use {c(BOLD, str(now))} right now, based on free VRAM)")
        else:
            print(f"  {c(DIM, label.ljust(9))} fixed at {c(BOLD, str(cfg))}")

    par_cfg = health.get("max_parallel_config")
    par_now = health.get("max_parallel_now")
    if par_cfg is not None and par_now is not None:
        label = "Parallel:"
        if str(par_cfg) == "auto":
            if int(par_now) > 1:
                print(f"  {c(DIM, label.ljust(9))} auto ({c(BOLD, str(par_now))} concurrent transcribes allowed)")
            else:
                print(f"  {c(DIM, label.ljust(9))} auto ({c(BOLD, '1')} at a time, VRAM tight)")
        else:
            print(f"  {c(DIM, label.ljust(9))} fixed at {c(BOLD, str(par_cfg))}")
    print(c(CYAN, "=" * width))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files via a remote whisper-server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("files", nargs="*", type=Path, help="audio/video files (omit with --status)")
    parser.add_argument("--status", action="store_true", help="print server status (model loaded, idle, remaining unload time) and exit")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="whisper-server URL")
    parser.add_argument("--formats", "-f", type=_parse_formats, default=list(ALL_FORMATS),
                        help="comma list of txt,srt,vtt,json (or 'all')")
    parser.add_argument("--language", "-l", default=None, help="force language code (e.g. en)")
    parser.add_argument("--word-timestamps", "-w", action="store_true", help="include per-word timestamps (in .json)")
    parser.add_argument("--diarize", "-D", action="store_true",
                        help="label speakers via pyannote Community-1 (requires HF_TOKEN on the server)")
    parser.add_argument("--diarize-model", default=None,
                        help="HF model id for diarization (e.g. Revai/reverb-diarization-v2). "
                             "Default: server's WHISPER_DIARIZE_MODEL.")
    parser.add_argument("--output-dir", "-o", type=Path, default=None,
                        help="write outputs to this folder (default: next to each source)")
    parser.add_argument("--no-pause", action="store_true", help="exit immediately instead of waiting for a keypress")
    parser.add_argument("--no-recursive", action="store_true",
                        help="when a folder is given, only scan its top level (default: walk sub-folders)")
    args = parser.parse_args()

    if args.status:
        try:
            health = http_get_json(args.server, "/health", timeout=5.0)
        except Exception as e:
            print(c(RED, f"cannot reach whisper-server at {args.server}"))
            print(c(DIM, f"  {e}"))
            return 2
        _print_status(args.server, health)
        return 0

    if not args.files:
        parser.error("no files given (use --status to just query server state)")

    sources: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            return
        seen.add(resolved)
        sources.append(path)

    def _scan_dir(folder: Path) -> int:
        added = 0
        iterator = folder.rglob("*") if not args.no_recursive else folder.iterdir()
        for entry in sorted(iterator):
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in MEDIA_EXTS:
                continue
            before = len(sources)
            _add(entry)
            if len(sources) > before:
                added += 1
        return added

    for p in args.files:
        if not p.exists():
            print(c(YELLOW, f"skip: {p} (not found)"), file=sys.stderr)
            continue
        if p.is_dir():
            scope = "recursively" if not args.no_recursive else "(top level only)"
            n = _scan_dir(p)
            if n == 0:
                print(c(YELLOW, f"skip: {p} (no media files found {scope})"), file=sys.stderr)
            else:
                print(c(DIM, f"  + {p}: {n} media file(s) found {scope}"))
            continue
        _add(p)

    if not sources:
        print(c(RED, "no valid files to transcribe"), file=sys.stderr)
        if not args.no_pause:
            press_any_key()
        return 1

    # Health check
    try:
        health = http_get_json(args.server, "/health", timeout=5.0)
    except Exception as e:
        print(c(RED, f"cannot reach whisper-server at {args.server}"))
        print(c(DIM, f"  {e}"))
        print()
        print(c(DIM, "hints:"))
        print(c(DIM, "  * is the server running? (systemctl --user status whisper-server)"))
        print(c(DIM, "  * is WHISPER_HOST set to 0.0.0.0 so LAN clients can reach it?"))
        print(c(DIM, "  * is the port 9000 allowed by the firewall on the server?"))
        if not args.no_pause:
            press_any_key()
        return 2

    print_header(args.server, health)

    if not health.get("model_loaded"):
        print(c(DIM, "note: the first file will also load the model (~5s warm, a few minutes if not yet downloaded)"))
        print()

    if args.diarize:
        if not health.get("diarize_available"):
            print(c(RED, "error: --diarize requested but the server has no HF_TOKEN set."))
            print(c(DIM, "  set HF_TOKEN in the service env and restart whisper-server."))
            if not args.no_pause:
                press_any_key()
            return 2
        diar_model_label = args.diarize_model or health.get('diarize_model')
        print(c(YELLOW, f"diarization: on (model: {diar_model_label})"))
        print()

    print(c(BOLD, f"Transcribing {len(sources)} file(s):"))
    print(c(DIM, "  (Ctrl+C: stop queue after current file · twice: hard abort)"))
    ok = 0
    failed: list[tuple[Path, str]] = []
    skipped: list[Path] = []
    total_audio = 0.0
    total_wall = 0.0
    out_dirs: set[Path] = set()

    # First Ctrl+C: ask the server to cancel the in-flight job, stop queue
    # after current file. Second Ctrl+C: hard abort.
    cancel_state: dict[str, Any] = {"count": 0, "active_job_id": None}

    def _async_cancel(job_id: str) -> None:
        try:
            http_post_form(args.server, "/cancel", {"job_id": job_id}, timeout=3.0)
        except Exception:
            pass  # best effort

    def _on_sigint(_signum, _frame):
        cancel_state["count"] += 1
        if cancel_state["count"] == 1:
            print()
            print(c(YELLOW, "  cancel requested — asking server to abort current file, then stopping queue."))
            print(c(DIM, "  (press Ctrl+C again to hard abort)"))
            jid = cancel_state.get("active_job_id")
            if jid:
                threading.Thread(target=_async_cancel, args=(jid,), daemon=True).start()
        else:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt()

    prev_handler = signal.signal(signal.SIGINT, _on_sigint)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(sources, start=1):
        if cancel_state["count"] >= 1:
            skipped = list(sources[i - 1:])
            break
        label = f"  [{i}/{len(sources)}] {c(BOLD, src.name)}  {c(DIM, _fmt_bytes(src.stat().st_size))}"
        print(label)

        job_id = uuid.uuid4().hex
        fields = {
            "formats": ",".join(args.formats),
            "word_timestamps": "true" if args.word_timestamps else "false",
            "vad_filter": "true",
            "diarize": "true" if args.diarize else "false",
            "job_id": job_id,
        }
        if args.language:
            fields["language"] = args.language
        if args.diarize and args.diarize_model:
            fields["diarize_model"] = args.diarize_model

        # Convert video / uncompressed audio to a tiny MP3 before upload.
        upload_path, tmp_dir, note = _prep_upload(src)
        if note:
            print(f"    {c(DIM, note)}")

        prefix = f"    {c(DIM, 'working')}"
        t0 = time.monotonic()
        data: dict | None = None
        err: str | None = None
        cancel_state["active_job_id"] = job_id
        try:
            with Spinner(prefix):
                data = http_post_multipart_file(args.server, "/transcribe", fields, upload_path)
        except Exception as e:
            err = str(e)
        finally:
            cancel_state["active_job_id"] = None
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        wall = time.monotonic() - t0
        # Server returns 499 with {"cancelled": true, ...} when /cancel hits
        # before the worker finishes. Don't treat that as a failure.
        if data and data.get("cancelled"):
            skipped.append(src)
            print(f"    {c(YELLOW, 'cancelled')} after {wall:.1f}s")
            continue
        if err and "HTTP 499" in err:
            skipped.append(src)
            print(f"    {c(YELLOW, 'cancelled')} after {wall:.1f}s")
            continue
        if err or data is None:
            failed.append((src, err or "unknown error"))
            print(f"    {c(RED, 'FAILED')} in {wall:.1f}s  {c(DIM, err or '')}")
            continue

        dur = float(data.get("duration") or 0.0)
        total_audio += dur
        total_wall += wall
        ok += 1

        # Write outputs
        base = args.output_dir if args.output_dir else src.parent
        base.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for fmt in args.formats:
            body = (data.get("formats") or {}).get(fmt)
            if body is None:
                continue
            dst = base / f"{src.stem}.{fmt}"
            dst.write_text(body, encoding="utf-8")
            written.append(dst)
        # Non-English source: server returns an English translation too -> *.translated.ext
        translated_fmts = data.get("formats_translated") or {}
        for fmt in args.formats:
            body = translated_fmts.get(fmt)
            if body is None:
                continue
            dst = base / f"{src.stem}.translated.{fmt}"
            dst.write_text(body, encoding="utf-8")
            written.append(dst)
        out_dirs.add(base)

        rtf = dur / wall if wall > 0 else 0.0
        exts = ",".join(p.suffix.lstrip(".") for p in written)
        lang = data.get("language") or "?"
        tnote = " +translated" if data.get("translated") else ""
        extra = f"lang={lang}{tnote} · wrote {exts}"
        if data.get("diarized"):
            spk = data.get("num_speakers") or 0
            d_s = data.get("diarize_seconds") or 0
            extra = f"{spk} speakers (+{d_s}s) · " + extra
        print(
            f"    {c(GREEN, 'done')}  {dur:.1f}s audio in {wall:.1f}s  "
            f"{c(DIM, f'({rtf:.1f}x · {extra})')}"
        )

    signal.signal(signal.SIGINT, prev_handler)

    if skipped:
        print()
        print(c(YELLOW, f"  cancelled — {len(skipped)} file(s) skipped:"))
        for p in skipped:
            print(c(DIM, f"    - {p.name}"))

    print_summary(ok, len(sources), failed, total_audio, total_wall, out_dirs)

    exit_code = 0 if not failed and not skipped else (130 if skipped else 1)
    if not args.no_pause:
        press_any_key()
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
