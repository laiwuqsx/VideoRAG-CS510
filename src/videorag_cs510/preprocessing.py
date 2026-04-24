import json
import math
import os
import subprocess
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .config import FRAMES_PER_SEG, VIDEO_SEG_LENGTH


def get_video_duration(video_path: str) -> float:
    """
    Return the total duration of a video file in seconds.

    Prefer the video stream's duration. Fall back to the container format
    duration when the stream duration is missing.
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        video_path,
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    metadata = json.loads(out)

    for stream in metadata.get("streams", []):
        if stream.get("codec_type") == "video" and stream.get("duration"):
            return float(stream["duration"])

    format_duration = metadata.get("format", {}).get("duration")
    if format_duration is not None:
        return float(format_duration)

    raise ValueError(f"Could not determine duration for video: {video_path}")


def split_video_into_segments(
    video_path: str,
    output_dir: str,
    segment_length: int = VIDEO_SEG_LENGTH,
) -> List[Dict]:
    """
    Split a video into fixed-length segments and return segment metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    duration = get_video_duration(video_path)
    stem = Path(video_path).stem
    n_segs = math.ceil(duration / segment_length)
    segments: List[Dict] = []

    for i in range(n_segs):
        start = float(i * segment_length)
        end = float(min(duration, start + segment_length))
        seg_name = f"{stem}_seg{i}"
        out_path = str(Path(output_dir) / f"{seg_name}.mp4")

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            video_path,
            "-t",
            str(end - start),
            "-c",
            "copy",
            out_path,
        ]
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        segments.append(
            {
                "index": i,
                "name": seg_name,
                "path": out_path,
                "start_time": start,
                "end_time": end,
            }
        )

    return segments


def extract_frames_from_segment(
    segment_path: str,
    num_frames: int = FRAMES_PER_SEG,
) -> List[np.ndarray]:
    """
    Extract evenly spaced BGR frames from a video segment.
    """
    cap = cv2.VideoCapture(segment_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video segment: {segment_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    num_frames = max(1, min(num_frames, total))
    interval = total / num_frames
    frame_indices = [min(total - 1, int((i + 0.5) * interval)) for i in range(num_frames)]

    frames: List[np.ndarray] = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)

    cap.release()
    return frames
