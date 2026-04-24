import base64
import os
import subprocess
import tempfile
from typing import Dict, List

import cv2
import numpy as np

from .config import (
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    TRANSCRIPTION_METHOD,
    VISUAL_CAPTION_METHOD,
)
from .preprocessing import extract_frames_from_segment


def _extract_audio_to_mp3(segment_path: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        segment_path,
        "-vn",
        "-q:a",
        "0",
        "-map",
        "a",
        tmp_path,
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except subprocess.CalledProcessError:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return ""

    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return ""

    return tmp_path


def _transcribe_with_local_whisper(audio_path: str) -> str:
    import whisper

    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return (result.get("text") or "").strip()


def _caption_with_gemini(image_b64_list: List[str], transcript: str = "") -> str:
    if not GEMINI_API_KEY or not image_b64_list:
        return ""

    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = (
        "You are a video analysis assistant. Given sampled frames from one short video "
        "segment and an optional transcript, write a concise 1-3 sentence caption of "
        "what is visually shown. Mention key objects, people, actions, and visible "
        "on-screen text when present. Be specific and factual.\n\n"
        f"Transcript context:\n{transcript or '(no transcript available)'}"
    )

    contents = [prompt]
    for image_b64 in image_b64_list:
        contents.append({"mime_type": "image/jpeg", "data": image_b64})

    response = model.generate_content(contents)
    text = getattr(response, "text", "") or ""
    return text.strip()


def _caption_with_openai(image_b64_list: List[str], transcript: str = "") -> str:
    if not OPENAI_API_KEY or not image_b64_list:
        return ""

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    user_content = [
        {
            "type": "text",
            "text": (
                "Describe these sampled frames from one short video segment in 1-3 "
                "factual sentences. Mention visible objects, actions, and on-screen "
                f"text when present.\n\nTranscript context:\n{transcript or '(no transcript available)'}"
            ),
        }
    ]
    user_content.extend(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        }
        for image_b64 in image_b64_list
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a concise and factual video captioning assistant.",
            },
            {"role": "user", "content": user_content},
        ],
        max_tokens=200,
    )
    return (response.choices[0].message.content or "").strip()


def transcribe_segment(segment_path: str) -> str:
    """
    Extract audio from a video segment and return a plain-text transcript.

    Currently implemented transcription path:
    - whisper_local

    Returns an empty string when no audio track is available.
    """
    audio_path = _extract_audio_to_mp3(segment_path)
    if not audio_path:
        return ""

    try:
        if TRANSCRIPTION_METHOD == "whisper_local":
            return _transcribe_with_local_whisper(audio_path)
        raise ValueError(f"Unsupported transcription method: {TRANSCRIPTION_METHOD}")
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)


def caption_frames(frames: List[np.ndarray], transcript: str = "") -> str:
    """
    Generate a short visual caption for sampled video frames.

    Supported caption methods:
    - gemini
    - openai
    - skip

    Returns an empty string when captioning is skipped or unavailable.
    """
    if not frames or VISUAL_CAPTION_METHOD == "skip":
        return ""

    image_b64_list: List[str] = []
    for frame in frames:
        ok, encoded = cv2.imencode(".jpg", frame)
        if ok:
            image_b64_list.append(base64.b64encode(encoded.tobytes()).decode("utf-8"))

    if not image_b64_list:
        return ""

    if VISUAL_CAPTION_METHOD == "gemini":
        return _caption_with_gemini(image_b64_list, transcript=transcript)
    if VISUAL_CAPTION_METHOD == "openai":
        return _caption_with_openai(image_b64_list, transcript=transcript)
    if VISUAL_CAPTION_METHOD == "skip":
        return ""

    raise ValueError(f"Unsupported visual caption method: {VISUAL_CAPTION_METHOD}")


def process_segment(segment: Dict) -> Dict:
    """
    Transcribe audio and caption sampled frames for a single segment.
    """
    enriched = dict(segment)
    transcript = transcribe_segment(segment["path"])
    frames = extract_frames_from_segment(segment["path"])
    caption = caption_frames(frames, transcript=transcript)
    enriched["transcript"] = transcript
    enriched["caption"] = caption
    return enriched
