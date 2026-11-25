import os
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INITIAL_POLL_DELAY_SECONDS = 15
POLL_INTERVAL_SECONDS = 3
GENERATION_TIMEOUT_SECONDS = 900
INITIAL_CLIP_DURATIONS = (4, 6, 8)
MAX_SCENE_NARRATION_WORDS = 22
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 24
MAX_CONCURRENT_SCENE_GENERATIONS = 5


def _google_video_client() -> Any:
    load_dotenv(PROJECT_ROOT / ".env")
    service_account_file = os.getenv("SA_FILE")
    project_id = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION", "us-central1")

    if not service_account_file:
        raise RuntimeError("SA_FILE is missing from the project .env file.")
    if not project_id:
        raise RuntimeError("PROJECT_ID is missing from the project .env file.")

    credential_path = Path(service_account_file).expanduser()
    if not credential_path.is_absolute():
        credential_path = PROJECT_ROOT / credential_path
    if not credential_path.is_file():
        raise RuntimeError(
            f"Google service-account file was not found: {credential_path}"
        )
    if credential_path.stat().st_size == 0:
        raise RuntimeError(
            f"The Google service-account file at '{credential_path.name}' is empty (0 bytes). "
            "Please paste your Google Cloud Service Account JSON key into this file to enable Veo video generation."
        )

    try:
        from google import genai
        from google.oauth2 import service_account
    except ImportError as exc:
        raise RuntimeError(
            "Google Veo dependencies are missing. "
            "Install requirements.txt."
        ) from exc

    credentials = service_account.Credentials.from_service_account_file(
        str(credential_path),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        credentials=credentials,
    )


def _wait_for_video_operation(
    video_client: Any,
    operation: Any,
    output_path: Path,
    status_callback: Callable[[str], None] | None,
    waiting_message: str,
) -> None:
    deadline = time.monotonic() + GENERATION_TIMEOUT_SECONDS
    # Veo video generation takes at least 15-20 seconds; initial sleep avoids redundant polling
    if not operation.done:
        if status_callback:
            status_callback(waiting_message)
        time.sleep(INITIAL_POLL_DELAY_SECONDS)
        operation = video_client.operations.get(operation)

    while not operation.done:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Google Veo did not finish generating video within "
                f"{GENERATION_TIMEOUT_SECONDS // 60} minutes."
            )
        if status_callback:
            status_callback(waiting_message)
        time.sleep(POLL_INTERVAL_SECONDS)
        operation = video_client.operations.get(operation)

    if operation.error:
        raise RuntimeError(f"Google Veo video generation failed: {operation.error}")

    generated_videos = (
        operation.response.generated_videos
        if operation.response is not None
        else None
    )
    if not generated_videos:
        raise RuntimeError("Google Veo completed without returning a video.")

    generated_videos[0].video.save(str(output_path))
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("Google Veo returned an empty video file.")


def _generate_veo_clip(
    video_client: Any,
    model: str,
    prompt: str,
    duration_seconds: int,
    output_path: Path,
    status_callback: Callable[[str], None] | None,
) -> None:
    from google.genai import types

    operation = video_client.models.generate_videos(
        model=model,
        prompt=prompt,
        config=types.GenerateVideosConfig(
            aspect_ratio="16:9",
            duration_seconds=duration_seconds,
            number_of_videos=1,
            resolution="720p",
            generate_audio=True,
        ),
    )
    _wait_for_video_operation(
        video_client,
        operation,
        output_path,
        status_callback,
        "Waiting for Google Veo to generate this scene...",
    )


def _clip_duration_for_narration(narration: str) -> int:
    word_count = len(narration.split())
    if word_count <= 3:
        return 4
    # Explainer scenes with full-sentence voiceover require 8 seconds in Veo
    # to allow natural lead-in audio, articulate pacing, and clean trailing room tone.
    return 8


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts" / "arial.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    current_line = ""
    for word in text.split():
        candidate = f"{current_line} {word}".strip()
        if current_line and draw.textlength(candidate, font=font) > max_width:
            lines.append(current_line)
            current_line = word
        else:
            current_line = candidate
    if current_line:
        lines.append(current_line)
    return lines


def _draw_caption(
    narration: str,
    output_path: Path,
) -> None:
    image = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(28)
    lines = _wrap_text(draw, narration, font, int(VIDEO_WIDTH * 0.84))
    line_height = 36
    box_height = max(48, len(lines) * line_height + 18)
    top = VIDEO_HEIGHT - box_height - 17
    draw.rounded_rectangle(
        (35, top, VIDEO_WIDTH - 35, VIDEO_HEIGHT - 12),
        radius=9,
        fill=(0, 0, 0, 195),
    )
    draw.multiline_text(
        (VIDEO_WIDTH / 2, top + 7),
        "\n".join(lines),
        font=font,
        fill=(255, 255, 255, 255),
        align="center",
        anchor="ma",
        spacing=5,
    )
    image.save(output_path)


def _caption_segments(
    narration: str,
    clip_duration: float,
) -> list[tuple[str, float, float]]:
    words = narration.split()
    max_words_per_caption = 8
    groups = [
        words[index:index + max_words_per_caption]
        for index in range(0, len(words), max_words_per_caption)
    ]
    total_words = len(words)
    elapsed = 0.0
    segments: list[tuple[str, float, float]] = []
    for group in groups:
        start = elapsed
        elapsed += clip_duration * len(group) / total_words
        segments.append((" ".join(group), start, elapsed))
    return segments


def _run_ffmpeg(arguments: list[str]) -> None:
    import imageio_ffmpeg

    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", *arguments]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(
            f"Video composition failed: {details[-3000:]}"
        ) from exc


def _render_scene(
    source_path: Path,
    caption_paths: list[tuple[Path, float, float]],
    output_path: Path,
    duration: float,
) -> None:
    input_args = ["-y", "-i", str(source_path)]
    filters = [
        f"[0:v]fps={VIDEO_FPS},scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:"
        "force_original_aspect_ratio=decrease,"
        f"pad={VIDEO_WIDTH}:{VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1[v0]"
    ]
    overlay_input_indices: list[int] = []
    for path, _, _ in caption_paths:
        overlay_input_indices.append(len(overlay_input_indices) + 1)
        input_args.extend(
            ["-loop", "1", "-framerate", str(VIDEO_FPS), "-i", str(path)]
        )
    last_video = "v0"
    for caption_index, (_, start, end) in enumerate(caption_paths):
        output_label = f"v{caption_index + 1}"
        filters.append(
            f"[{last_video}][{overlay_input_indices[caption_index]}:v]"
            "overlay=0:0:shortest=1:"
            f"enable='between(t,{start:.3f},{end:.3f})'[{output_label}]"
        )
        last_video = output_label

    filter_graph = ";".join(filters)
    _run_ffmpeg([
        *input_args,
        "-filter_complex",
        filter_graph,
        "-map",
        f"[{last_video}]",
        "-map",
        "0:a:0",
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(VIDEO_FPS),
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output_path),
    ])


def _concat_file_entry(path: Path) -> str:
    escaped_path = path.as_posix().replace("'", "'\\''")
    return f"file '{escaped_path}'\n"


def generate_storyboard_video(
    storyboard: dict[str, Any],
    status_callback: Callable[[str], None] | None = None,
) -> bytes:
    """Generate Veo scenes with native audio in parallel and assemble them."""
    scenes = storyboard.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("The video storyboard must contain scenes.")
    include_subtitles = storyboard.get("include_subtitles", False)
    if not isinstance(include_subtitles, bool):
        raise ValueError("Storyboard include_subtitles must be a boolean.")

    model = os.getenv("MODEL", "veo-3.1-lite-generate-001")
    video_client = _google_video_client()
    temporary_directory = tempfile.TemporaryDirectory()
    temporary_path = Path(temporary_directory.name)

    total_scenes = len(scenes)
    completed_scenes = 0

    def _process_single_scene(index: int, scene: dict[str, Any]) -> tuple[int, Path]:
        title = scene["title"]
        narration = scene["narration"]
        duration = _clip_duration_for_narration(narration)
        veo_prompt = (
            "Cinematic 3D isometric architectural motion graphics on a seamless matte dark charcoal background (#0E1117). "
            "High-end technical presentation featuring structured architectural diagrams, modular workflow blocks, "
            "prominent directional connector arrows, comparative visual bar meters, and system topology layouts. "
            "Volumetric studio rim lighting with subtle floor reflections. "
            "CRITICAL: Do NOT render any on-screen text, words, letters, numbers, badges, or code inside the video pixels; "
            "depict all workflows using distinct geometric diagram blocks, directional connector arrows, and color status states instead. "
            "Show no human faces, presenter, or talking heads. "
            "Generate native high-fidelity audio with off-screen voiceover speaking clearly and naturally. "
            f'The voiceover must say exactly: "{narration}"\n\n'
            f"Scene: {title}\n"
            f"Visual direction: {scene['visual_prompt']}"
        )
        raw_video_path = temporary_path / f"veo_scene_{index:04d}.mp4"
        _generate_veo_clip(
            video_client,
            model,
            veo_prompt,
            duration,
            raw_video_path,
            None,
        )

        if include_subtitles:
            caption_paths = []
            for caption_index, (caption, start, end) in enumerate(
                _caption_segments(narration, float(duration)),
                start=1,
            ):
                caption_path = temporary_path / (
                    f"caption_{index:04d}_{caption_index:04d}.png"
                )
                _draw_caption(caption, caption_path)
                caption_paths.append((caption_path, start, end))

            rendered_path = temporary_path / f"scene_{index:04d}.mp4"
            _render_scene(
                raw_video_path,
                caption_paths,
                rendered_path,
                float(duration),
            )
            return index, rendered_path
        else:
            return index, raw_video_path

    try:
        max_workers = min(total_scenes, MAX_CONCURRENT_SCENE_GENERATIONS)
        if status_callback:
            status_callback(
                f"Starting parallel generation of {total_scenes} scene(s) with Veo (concurrency: {max_workers})..."
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(_process_single_scene, index, scene): index
                for index, scene in enumerate(scenes, start=1)
            }
            results = []
            for future in as_completed(future_to_index):
                result = future.result()
                results.append(result)
                completed_scenes += 1
                if status_callback:
                    status_callback(
                        f"Generated {completed_scenes}/{total_scenes} scenes in parallel with Veo..."
                    )

        results.sort(key=lambda item: item[0])
        scene_paths = [path for _, path in results]

        if status_callback:
            status_callback("Assembling the complete video...")
        concat_path = temporary_path / "scenes.txt"
        concat_path.write_text(
            "".join(_concat_file_entry(path) for path in scene_paths),
            encoding="utf-8",
        )
        output_path = temporary_path / "storyboard.mp4"
        _run_ffmpeg([
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ])
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("The assembled video file is empty.")
        video_bytes = output_path.read_bytes()
    finally:
        temporary_directory.cleanup()

    if status_callback:
        status_callback("Video generation completed.")
    return video_bytes
