# SST (Speech-to-Text) Endpoint

Local speech-to-text API powered by the Cohere Transcribe model (`CohereLabs/cohere-transcribe-03-2026`) — a 2B-parameter conformer-based encoder-decoder, Apache 2.0 licensed.

## Supported Languages

14 languages are supported. Pass the language code in the `language` field of the request.

| Language | Code |
|---|---|
| English | `en` |
| French | `fr` |
| German | `de` |
| Spanish | `es` |
| Italian | `it` |
| Portuguese | `pt` |
| Dutch | `nl` |
| Polish | `pl` |
| Greek | `el` |
| Chinese (Mandarin) | `zh` |
| Japanese | `ja` |
| Korean | `ko` |
| Vietnamese | `vi` |
| Arabic | `ar` |

**Note**: There is no automatic language detection — you must specify the language explicitly.

## Server

- **URL**: `http://localhost:7700` (default port, configurable)
- **Launch**: `./launch_sst_server.sh [PORT]`

## Endpoint

### `POST /transcribe`

Transcribes a base64-encoded audio file to text.

#### Request

**Content-Type**: `application/json`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `audio_base64` | string | yes | — | The audio file encoded as a base64 string. Supports any format that `soundfile` can read (WAV, FLAC, OGG, etc.). |
| `language` | string | no | `"en"` | One of the 14 supported language codes (see table above). |

#### Response

**Content-Type**: `application/json`

| Field | Type | Description |
|---|---|---|
| `text` | string | The transcribed text. |

#### Error Responses

| Status | Reason |
|---|---|
| 400 | Invalid base64 data or unsupported/corrupted audio format |
| 500 | Transcription model failed internally |

## Example

### Python

```python
import base64
import requests

# Read and encode the audio file
with open("audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:7700/transcribe",
    json={"audio_base64": audio_b64, "language": "en"},
)
print(response.json())
# {"text": "Hello, how are you?"}
```

### curl

```bash
AUDIO_B64=$(base64 -w0 audio.wav)

curl -X POST http://localhost:7700/transcribe \
  -H "Content-Type: application/json" \
  -d "{\"audio_base64\": \"$AUDIO_B64\", \"language\": \"en\"}"
```

## Notes

- Audio is automatically converted to mono and resampled to 16 kHz if needed.
- The model is loaded per request and unloaded after, so the first call has a cold-start delay.
- The server must have access to the gated Hugging Face model. Authenticate first with `hf auth login`.
