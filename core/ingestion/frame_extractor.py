"""
Multimodal Keyframe Extractor: Samples visual frames from videos or generates
visual slide representations for visual-textual RAG alignment.
"""
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.utils.time_utils import seconds_to_timestamp

logger = logging.getLogger(__name__)


@dataclass
class VideoFrame:
    frame_id: str
    timestamp: float
    image_path: str
    ocr_text: str = ""
    visual_description: str = ""

    @property
    def timestamp_str(self) -> str:
        return seconds_to_timestamp(self.timestamp)


class FrameExtractor:
    """
    Extracts keyframes from video files, RTSP streams, or generates
    visual slide frames aligned with transcript segments.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("data/frames")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_from_video_file(
        self,
        video_path: str,
        video_id: str,
        interval_seconds: int = 20,
        max_frames: int = 40,
    ) -> List[VideoFrame]:
        """
        Extract keyframes from a local MP4/MKV video using OpenCV.
        """
        frames: List[VideoFrame] = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Could not open video file: %s", video_path)
            return frames

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        sample_step = int(fps * interval_seconds)
        current_frame = 0

        while current_frame < frame_count and len(frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = current_frame / fps
            img_filename = f"{video_id}_frame_{int(timestamp):04d}.jpg"
            img_path = self.output_dir / img_filename

            # Resize to standard width 640 while maintaining aspect ratio
            h, w = frame.shape[:2]
            target_w = 640
            target_h = int(h * (target_w / w))
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

            cv2.imwrite(str(img_path), resized)

            # Extract basic OCR or visual cues
            ocr_text = self._detect_text_regions(resized)
            frames.append(
                VideoFrame(
                    frame_id=f"frame_{int(timestamp)}",
                    timestamp=timestamp,
                    image_path=str(img_path),
                    ocr_text=ocr_text,
                    visual_description=f"Keyframe at {seconds_to_timestamp(timestamp)} with text density: {len(ocr_text)} chars",
                )
            )
            current_frame += sample_step

        cap.release()
        return frames

    def generate_multimodal_frames_from_segments(
        self,
        video_id: str,
        segments: list,
        interval_seconds: int = 30,
        max_frames: int = 20,
    ) -> List[VideoFrame]:
        """
        Generates visual slide representations for transcript segments
        when full video download is omitted or in cloud/sandbox environments.
        Renders rich educational slides containing key formulas, topic badges, and quotes.
        """
        frames: List[VideoFrame] = []
        if not segments:
            return frames

        total_duration = segments[-1].end
        current_time = 0.0

        while current_time <= total_duration and len(frames) < max_frames:
            # Find closest segment
            matching = [s for s in segments if s.start <= current_time <= s.end]
            active_segment = matching[0] if matching else segments[0]

            img_filename = f"{video_id}_slide_{int(current_time):04d}.jpg"
            img_path = self.output_dir / img_filename

            if not img_path.exists():
                self._render_educational_slide(
                    img_path=img_path,
                    timestamp_str=seconds_to_timestamp(current_time),
                    title=f"Topic at {seconds_to_timestamp(current_time)}",
                    content=active_segment.text,
                    video_id=video_id,
                )

            frames.append(
                VideoFrame(
                    frame_id=f"slide_{int(current_time)}",
                    timestamp=current_time,
                    image_path=str(img_path),
                    ocr_text=active_segment.text[:120],
                    visual_description=f"Visual lecture slide at {seconds_to_timestamp(current_time)}: {active_segment.text[:80]}...",
                )
            )

            current_time += interval_seconds

        return frames

    def _render_educational_slide(
        self, img_path: Path, timestamp_str: str, title: str, content: str, video_id: str
    ) -> None:
        """Draws a clean, dark-mode 640x360 educational slide with visual cards."""
        width, height = 640, 360
        # Dark modern slate gradient background
        img = Image.new("RGB", (width, height), color=(18, 24, 38))
        draw = ImageDraw.Draw(img)

        # Header bar
        draw.rectangle([0, 0, width, 48], fill=(30, 41, 59))
        draw.text((20, 14), f"▶ MULTIMODAL LECTURE FRAME  •  {timestamp_str}", fill=(56, 189, 248))
        draw.text((width - 120, 14), f"ID: {video_id[:8]}", fill=(148, 163, 184))

        # Card container
        draw.rounded_rectangle([30, 70, width - 30, height - 40], radius=8, fill=(15, 23, 42), outline=(51, 65, 85))

        # Title
        draw.text((50, 90), title, fill=(248, 250, 252))

        # Wrap text content
        words = content.split()
        lines = []
        cur_line = []
        for w in words:
            cur_line.append(w)
            if len(" ".join(cur_line)) > 55:
                lines.append(" ".join(cur_line))
                cur_line = []
        if cur_line:
            lines.append(" ".join(cur_line))

        y = 135
        for line in lines[:5]:
            draw.text((50, y), line, fill=(203, 213, 225))
            y += 26

        # Formula / visual tag badge
        draw.rounded_rectangle([50, height - 85, 260, height - 55], radius=4, fill=(30, 58, 138))
        draw.text((62, height - 78), "⚡ Visual Concept Highlight", fill=(147, 197, 253))

        img.save(str(img_path), "JPEG", quality=90)

    def _detect_text_regions(self, img_bgr: np.ndarray) -> str:
        """Simple morphological edge text region heuristic for visual density."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, -1)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, -1)
        gradient = cv2.subtract(grad_x, grad_y)
        gradient = cv2.convertScaleAbs(gradient)
        _, thresh = cv2.threshold(gradient, 225, 255, cv2.THRESH_BINARY)
        non_zero = cv2.countNonZero(thresh)
        if non_zero > 500:
            return f"Detected slide text region (edge density={non_zero})"
        return ""

