# API mode: the same server without a local model

**Short answer: yes.** You can run this server with no GPU and no local Whisper model by having it call a hosted speech API, and keep feature parity for every client. The trick is to put the API **behind the existing HTTP contract**, not to point clients at a provider directly.

Status: design notes. Not yet implemented in `server.py`. Provider capabilities below were checked in October 2026; recheck the linked docs before building, because these APIs change often.

## Why keep the server in the middle

Every consumer already speaks this server's contract (see [SOP.md section 11](SOP.md#11-how-the-server-behaves)):

* `client/transcribe-remote.py` and `client/transcribe-remote-diarize.py`
* the right click launchers
* [`video-ultracut/scripts/transcribe.py`](https://github.com/crxssrazr93/video-ultracut/blob/main/scripts/transcribe.py), which posts `formats=json,srt`, `word_timestamps=true`, optional `diarize`, and reads `formats.json.segments[].words[]`, `formats.srt`, `language`, `diarized`, `is_english`, `translated` and `formats_translated`
* any other transcript script that uses `WHISPER_SERVER`

If the server swaps its engine and keeps the response shapes, **none of them change**. You only change the `WHISPER_SERVER` URL if the API mode server runs somewhere else. Pointing each tool at a provider instead would mean rewriting every tool, and you would lose the uniform "raw plus English translation" and speaker label behavior, because no single provider does all of it the same way.

## What parity requires

| Feature the clients use | Local server does it with | An API backend needs |
|---|---|---|
| Auto language detection | Whisper pass 1 | Detected language in the response |
| Raw transcript in the spoken language | `task=transcribe` | A transcription endpoint |
| Segment timestamps (srt, vtt) | Whisper segments | Segment level times |
| Word timestamps (json, captions, transcript cuts) | `word_timestamps=True` | Word level times |
| English translation for non English audio, timed | `task=translate` | An audio to English **translation** endpoint, or a text translation step mapped back onto the raw segments |
| Speaker labels, automatic speaker count | pyannote Community-1 or Reverb v2 | Diarization (hosted or local) |
| Cancel a running job (HTTP 499) | Job flag | Abort the upstream request, or stop between chunks |
| Anti repetition on noisy audio | faster-whisper decode options | Mostly not controllable; see gaps |

## Provider fit

| Provider | Language detect | Word times | Audio to English translate | Diarization | Notes |
|---|---|---|---|---|---|
| **Groq**, `whisper-large-v3` | yes | yes (`verbose_json`, `timestamp_granularities=word,segment`) | yes, translations endpoint (English only, which is all we need) | no | Same Whisper large-v3 weights as the local box, so output is closest to the local server. Upload limit 25 MB free tier, 100 MB dev tier. |
| **OpenAI**, `whisper-1` | yes | yes (`verbose_json`) | yes, `/audio/translations` | no | 25 MB upload limit. |
| **OpenAI**, `gpt-4o-transcribe-diarize` | yes | **no** (segment level speakers only, no `timestamp_granularities`) | no | yes | Fails the word timestamp requirement, so it cannot replace the server alone. |
| **ElevenLabs** Scribe v2 | yes | yes | no | yes, up to 32 speakers | Covers transcribe, words and speakers in one call. Pair it with Groq or OpenAI for the translation pass. |
| **Deepgram** Nova 3 | yes | yes | no | yes (speaker per word) | Same pairing note as ElevenLabs. |
| **AssemblyAI** | yes | yes | text translation option | yes (`speaker_labels`) | Translation is text based, so map it back onto segments. |
| **pyannoteAI** (hosted pyannote) | n/a | n/a | n/a | yes | The diarization half only. `pyannoteai-sdk` is already in the lock file because pyannote.audio 4 depends on it. |

## Recommended combinations

**1. Closest parity: Groq Whisper plus pyannote.** Groq `whisper-large-v3` for both passes (transcribe, then translate when not English), and diarization from either
* local pyannote Community-1 on the **CPU** (no GPU needed; slower, but it is the exact same diarization model as today, so speaker behavior stays identical), or
* hosted pyannoteAI, for a fully GPU free and fast setup.

This is the recommended API mode. The words come from the same model weights the box runs, and the diarization is the same model family.

**2. One vendor for transcription and speakers: ElevenLabs Scribe plus Groq translate.** Scribe returns words with speaker ids; call Groq's translations endpoint only for non English audio, then copy speakers onto the translated segments by time overlap, the same way `_assign_speakers` does now.

**3. OpenAI only:** `whisper-1` transcribe and translate, plus local CPU pyannote for speakers. Use it if you already have an OpenAI key and nothing else.

## The adapter seam

`server.py` is already split so that only the engine needs replacing:

* `_get_model()` loads faster-whisper. In API mode it loads nothing.
* `_do_transcribe()` runs pass 1 and the optional pass 2 and returns plain dicts: segments `{start, end, text, words: [{start, end, word}]}`, the language code, and translated segments. **This is the only function an API backend must reimplement.**
* `_get_diar_pipeline()` and `_assign_speakers()` work on those dicts and on a waveform, independent of the engine.
* `render_formats()` turns the dicts into txt, srt, vtt and json. Unchanged.
* `/health`, `/transcribe`, `/cancel`, idle exit and socket activation. Unchanged.

Proposed configuration:

```
WHISPER_BACKEND=local          # local | groq | openai | elevenlabs
WHISPER_API_KEY=...            # key for the chosen backend
WHISPER_TRANSLATE_BACKEND=groq # where pass 2 goes when WHISPER_BACKEND cannot translate
WHISPER_DIARIZE_BACKEND=local  # local (pyannote on CPU or GPU) | pyannoteai
WHISPER_DEVICE=cpu             # for local pyannote on a machine with no GPU
```

Sketch of a Groq engine (illustrative, not tested):

```python
import requests

def _groq(path, audio_path, translate=False):
    with open(audio_path, "rb") as f:
        r = requests.post(
            f"https://api.groq.com/openai/v1/audio/{'translations' if translate else 'transcriptions'}",
            headers={"Authorization": f"Bearer {API_KEY}"},
            files={"file": f},
            data={"model": "whisper-large-v3",
                  "response_format": "verbose_json",
                  "timestamp_granularities[]": ["word", "segment"]},
            timeout=1800)
    r.raise_for_status()
    return r.json()

def _to_segments(js):
    words = js.get("words") or []
    segs = []
    for s in js.get("segments") or []:
        segs.append({"start": s["start"], "end": s["end"], "text": s["text"].strip(),
                     "words": [{"start": w["start"], "end": w["end"], "word": w["word"]}
                               for w in words if s["start"] <= w["start"] < s["end"]]})
    return segs

# pass 1: raw = _groq(..., translate=False) -> language = raw["language"]
# pass 2 (only if language is not English): eng = _groq(..., translate=True)
# then reuse the existing diarization and render_formats code unchanged
```

Note that the hosted APIs return the language as a name ("spanish") in some cases and as a code ("es") in others. Normalise it to the ISO code, because clients compare it with `en`.

## Gaps to handle so parity is real

* **File size limits.** 25 MB at OpenAI and Groq free tier (100 MB at Groq dev tier). The clients already upload 64 kbps mono MP3, about 0.48 MB per minute, so 25 MB is roughly 50 minutes. For longer files the adapter must split the audio (ffmpeg, cut at silences, about 20 minute chunks), send chunks, add each chunk's start offset to every timestamp, and merge. Take the language from the first chunk and pass it explicitly to the rest. Run diarization once over the whole file, not per chunk, or speaker ids will not match across chunks.
* **Word times on the translation pass.** Confirm the provider returns word level times for translations. If it only returns segments, fill words by spreading each segment's duration across its words in proportion to their length, and mark them approximate. Captions built from `.en.words.json` in video-ultracut depend on this.
* **Anti repetition.** Hosted Whisper usually exposes only `temperature` and `prompt`, not `repetition_penalty`, `no_repeat_ngram_size` or `compression_ratio_threshold`. Noisy phone voice notes may loop again. Mitigate in the adapter: trim silence first (ffmpeg `silenceremove`, which also covers `vad_filter=true`), then drop segments whose text compresses too well (gzip ratio above 2.4, the same threshold the local server uses) or repeats the previous segment.
* **Cancel.** Run the upstream request in a worker thread and close it when `/cancel` arrives; with chunking, also stop before the next chunk. Still return HTTP 499 `{"cancelled": true}`.
* **`/health`.** Return `"gpu": {"device": "api:groq"}` (no VRAM numbers; clients handle that) and `"model": "groq:whisper-large-v3"`, so a glance at the client header shows which mode answered.
* **`num_speakers`.** Keep counting from the diarization label set, whatever the diarization source.

## Things that change no matter what

* **Privacy.** Audio leaves the LAN. Fine for public videos, not for private calls or client recordings unless the provider's data terms allow it.
* **Cost.** The local box costs only electricity. APIs bill per audio minute, and diarization is often billed separately. Check current pricing on each provider's page before choosing.
* **Rate limits** on free tiers can fail long batch runs. Add retry with backoff on HTTP 429.
* **Security.** If the API mode server runs on a VPS instead of the LAN, it must not be exposed without authentication, because it would spend your API credit for anyone. Add a bearer token check, or keep it on the LAN or behind a VPN.

## Hybrid fallback (best of both)

Run the API mode server on the **laptop** (localhost) with `WHISPER_BACKEND=auto`: try the GPU box first and fall back to the hosted API when the box is off or busy. Clients then point at `http://localhost:9000` and always get an answer, and every client gets the fallback for free. video-ultracut already has its own local fallback (in process faster-whisper), but a server level fallback also covers the CLI clients and launchers.

## Sources

* [Groq speech to text docs](https://console.groq.com/docs/speech-to-text)
* [OpenAI speech to text guide](https://developers.openai.com/api/docs/guides/speech-to-text.md)
* [OpenAI community: GPT-4o Transcribe Diarize announcement](https://community.openai.com/t/introducing-gpt-4o-transcribe-diarize-now-available-in-the-audio-api/1362933)
* [ElevenLabs speech to text skill reference](https://tessl.io/registry/skills/github/elevenlabs/skills/speech-to-text)
* [Deepgram Nova 3 on Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/models/nova-3/)
* [AssemblyAI speaker labels FAQ](https://assemblyai.com/docs/faq/how-are-individual-speakers-identified-and-how-does-the-speaker-label-feature-work)
