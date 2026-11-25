import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MOVIE_PATH = PROJECT_ROOT / "json2video_portal_gateway_flow.json"
API_URL = "https://api.json2video.com/v2/movies"
POLL_INTERVAL_SECONDS = 3
RENDER_TIMEOUT_SECONDS = 900


def request_json(
    url: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    headers = {"x-api-key": api_key}
    body = None
    method = "GET"

    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
        method = "POST"

    request = Request(url, data=body, headers=headers, method=method)

    try:
        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"JSON2Video API returned HTTP {exc.code}: "
            f"{error_body or exc.reason}"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError("JSON2Video returned an invalid API response.")

    return data


def validate_movie(movie: dict[str, Any]) -> None:
    required_fields = {
        "resolution",
        "fps",
        "quality",
        "comment",
        "elements",
        "scenes",
    }
    missing_fields = required_fields.difference(movie)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise RuntimeError(
            f"Movie JSON is missing required fields: {missing}."
        )

    scenes = movie.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise RuntimeError("Movie JSON must contain at least one scene.")

    movie_elements = movie.get("elements", [])
    if not isinstance(movie_elements, list) or not any(
        isinstance(element, dict)
        and element.get("type") == "subtitles"
        for element in movie_elements
    ):
        raise RuntimeError(
            "Movie JSON must include a global subtitles element."
        )

    for index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            raise RuntimeError(
                f"Movie scene {index} must be a JSON object."
            )
        duration = scene.get("duration")
        if "duration" in scene and (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or (duration <= 0 and duration != -1)
        ):
            raise RuntimeError(
                f"Movie scene {index} duration must be positive or -1 "
                "for automatic timing."
            )
        elements = scene.get("elements")
        if not isinstance(elements, list):
            raise RuntimeError(
                f"Movie scene {index} must contain an elements list."
            )
        if not any(
            isinstance(element, dict)
            and element.get("type") == "voice"
            and isinstance(element.get("text"), str)
            and element["text"].strip()
            for element in elements
        ):
            raise RuntimeError(
                f"Movie scene {index} must contain voice narration."
            )
        has_image_or_video = any(
            isinstance(element, dict)
            and element.get("type") in {"image", "video"}
            and isinstance(element.get("src"), str)
            and element["src"].strip()
            for element in elements
        )
        has_text_visual = any(
            isinstance(element, dict)
            and element.get("type") == "text"
            and isinstance(element.get("text"), str)
            and element["text"].strip()
            for element in elements
        )
        has_html_visual = any(
            isinstance(element, dict)
            and element.get("type") == "html"
            and isinstance(element.get("html"), str)
            and element["html"].strip()
            for element in elements
        )
        if not has_image_or_video and not has_text_visual and not has_html_visual:
            raise RuntimeError(
                f"Movie scene {index} must contain at least one visible "
                "image, video, text, or HTML element in addition to narration."
            )


def render_movie_data(
    movie: dict[str, Any],
    status_callback: Callable[[str], None] | None = None,
) -> str:
    """Submit movie JSON directly to JSON2Video and return its video URL."""

    validate_movie(movie)
    render_movie = {
        **movie,
        "scenes": [
            {**scene, "duration": -1}
            for scene in movie["scenes"]
        ],
    }
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("JSON_TO_VIDEO_API_KEY")
    if not api_key:
        raise RuntimeError(
            "JSON_TO_VIDEO_API_KEY is missing from the project .env file."
        )

    submission = request_json(API_URL, api_key, payload=render_movie)
    project_id = submission.get("project")
    if not isinstance(project_id, str) or not project_id:
        raise RuntimeError(
            f"JSON2Video did not return a project ID: "
            f"{json.dumps(submission)}"
        )

    if status_callback:
        status_callback(
            f"JSON2Video project {project_id} submitted. Rendering video..."
        )
    deadline = time.monotonic() + RENDER_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        query = urlencode({"project": project_id})
        result = request_json(f"{API_URL}?{query}", api_key)
        movie_result = result.get("movie")
        if not isinstance(movie_result, dict):
            raise RuntimeError(
                f"JSON2Video returned an invalid movie status: "
                f"{json.dumps(result)}"
            )

        status = movie_result.get("status")
        if status == "done":
            video_url = movie_result.get("url")
            if not isinstance(video_url, str) or not video_url:
                raise RuntimeError(
                    "JSON2Video completed rendering but returned no video URL."
                )
            if status_callback:
                status_callback("Video rendering completed.")
            return video_url

        if status in {"error", "timeout"}:
            message = movie_result.get("message") or status
            raise RuntimeError(f"JSON2Video render failed: {message}")

        if status_callback:
            status_callback(
                f"JSON2Video render status: {status or 'waiting'}"
            )
        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"JSON2Video did not finish within "
        f"{RENDER_TIMEOUT_SECONDS // 60} minutes. Project: {project_id}"
    )


def render_movie(movie_path: Path) -> str:
    load_dotenv(PROJECT_ROOT / ".env")

    try:
        movie = json.loads(movie_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Movie JSON file not found: {movie_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Movie file is not valid JSON: {movie_path} ({exc})"
        ) from exc

    if not isinstance(movie, dict):
        raise RuntimeError("Movie JSON must contain a top-level object.")

    return render_movie_data(
        movie,
        status_callback=lambda message: print(message, flush=True),
    )


def download_video(video_url: str, output_path: Path) -> None:
    parsed_url = urlparse(video_url)
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise RuntimeError("JSON2Video returned an invalid video URL.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    download_completed = False
    try:
        with urlopen(video_url, timeout=60) as response:
            with output_path.open("wb") as video_file:
                while chunk := response.read(1024 * 1024):
                    video_file.write(chunk)
        download_completed = True
    except HTTPError as exc:
        raise RuntimeError(
            f"Video download failed (HTTP {exc.code}): {exc.reason}"
        ) from exc
    finally:
        if not download_completed:
            output_path.unlink(missing_ok=True)

    if not output_path.exists() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("The downloaded video file is empty.")


def download_video_bytes(video_url: str) -> bytes:
    parsed_url = urlparse(video_url)
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise RuntimeError("JSON2Video returned an invalid video URL.")

    try:
        with urlopen(video_url, timeout=60) as response:
            video_data = response.read()
    except HTTPError as exc:
        raise RuntimeError(
            f"Video download failed (HTTP {exc.code}): {exc.reason}"
        ) from exc

    if not video_data:
        raise RuntimeError("The downloaded video file is empty.")

    return video_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Submit a JSON2Video movie JSON file, wait for its MP4, "
            "and download it."
        )
    )
    parser.add_argument(
        "movie_json",
        nargs="?",
        type=Path,
        default=DEFAULT_MOVIE_PATH,
        help=(
            "Path to a JSON2Video movie file "
            f"(default: {DEFAULT_MOVIE_PATH.name})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path for the downloaded MP4 (default: next to the movie JSON).",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Do not wait for Enter before closing an interactive console.",
    )
    args = parser.parse_args()
    movie_path = args.movie_json.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else movie_path.with_suffix(".mp4")
    )
    if output_path == movie_path:
        parser.error("The MP4 output path must not overwrite the movie JSON.")

    try:
        video_url = render_movie(movie_path)
        download_video(video_url, output_path)
    except (RuntimeError, TimeoutError, URLError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print(f"Video downloaded: {output_path}", flush=True)
    if sys.stdin.isatty() and not args.no_pause:
        try:
            input("Press Enter to close this window...")
        except EOFError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
