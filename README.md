# Cortex Audio Transcriber

Transcribe and summarize Spanish audio using Whisper + Transformers in Docker.

## Requirements
- Docker
- Docker Compose

## Usage
Place your `.mp3` file in this folder, then run:

```bash
docker compose run --rm -e AUDIO_FILE=CG-20260203.mp3 transcriber
```

Outputs are written to this same folder by default:
- `CG-20260203_transcription.txt`
- `CG-20260203_summary.txt`

## Notes
- `OUTPUT_DIR` is already set to `/app/output` in `docker-compose.yml`, which is mounted to this project folder.
- If you omit `AUDIO_FILE`, the script default is `CG-20260203.mp3`.
