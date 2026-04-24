# VideoRAG-CS510

Project scaffold for the CS510 VideoRAG assignment.

## Structure

- `src/videorag_cs510/config.py`: shared assignment constants
- `src/videorag_cs510/preprocessing.py`: Part 1 video preprocessing
- `requirements.txt`: Python dependencies

## Implemented

- Part 1: video duration, video splitting, frame extraction
- Part 2: audio transcription, frame captioning, segment processing
- Part 3: JSON metadata storage, vector store, text chunk building

## Notes

- `ffmpeg` and `ffprobe` must be available on your system `PATH`
- Update the test video path in your notebook or scripts before running checks
- Default Part 2 path is `Local Whisper` for transcription and `Gemini` for simple visual captioning
