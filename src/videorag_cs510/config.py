import os
from pathlib import Path


TEST_VIDEO_PATH = "./test_video.mp4"
WORKING_DIR = "./videorag_output"
VIDEO_SEG_LENGTH = 30
FRAMES_PER_SEG = 4
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K_CHUNKS = 3

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

LLM_PROVIDER = os.environ.get("VIDEORAG_LLM_PROVIDER", "gemini")
TRANSCRIPTION_METHOD = os.environ.get("VIDEORAG_TRANSCRIPTION_METHOD", "whisper_local")
VISUAL_CAPTION_METHOD = os.environ.get("VIDEORAG_VISUAL_CAPTION_METHOD", "gemini")


def ensure_working_dir(path: str = WORKING_DIR) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
