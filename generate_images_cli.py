#!/usr/bin/env python3
import argparse
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

DEFAULT_BASE_URL = "https://api-inference.modelscope.cn/"
DEFAULT_MODEL = "Tongyi-MAI/Z-Image-Turbo"
DEFAULT_API_KEY = "你的token"

KNOWN_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
CONTENT_TYPE_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/gif": ".gif",
}

RATIO_TO_DEFAULT_SIZE = {
    "1:1": (1024, 1024),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
    "3:2": (1248, 832),
    "2:3": (832, 1248),
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "21:9": (1344, 576),
    "9:21": (576, 1344),
}

MODELS_USING_SIZE_FIELD = {
    "tongyi-mai/z-image",
    "tongyi-mai/z-image-turbo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ModelScope image generation tasks from CLI and download outputs."
    )
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="ModelScope token. Defaults to the built-in token used by test.py.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID.")

    parser.add_argument(
        "--ratio",
        help=(
            "Aspect ratio, e.g. 1:1, 16:9. "
            "Will be injected into prompt as --ar and also mapped to resolution when --size/--width/--height are not set."
        ),
    )
    parser.add_argument("--num-images", type=int, default=1, help="Target number of images.")
    parser.add_argument(
        "--mode",
        choices=["auto", "api", "loop"],
        default="auto",
        help=(
            "How to satisfy --num-images: "
            "api=send n in one task; loop=submit N tasks; "
            "auto=try api then fallback to loop on request error."
        ),
    )

    parser.add_argument("--negative-prompt", help="Optional negative prompt.")
    parser.add_argument(
        "--size",
        help=(
            "Optional size value (example: 1024*1024 or 1024x1024). "
            "Request payload uses WxH format, e.g. 1024x1024."
        ),
    )
    parser.add_argument("--width", type=int, help="Optional width if model supports it.")
    parser.add_argument("--height", type=int, help="Optional height if model supports it.")
    parser.add_argument("--seed", type=int, help="Optional seed if model supports it.")
    parser.add_argument("--steps", type=int, help="Optional steps if model supports it.")
    parser.add_argument("--guidance-scale", type=float, help="Optional guidance scale if model supports it.")
    parser.add_argument(
        "--resolution-mode",
        choices=["auto", "size", "width-height", "both"],
        default="auto",
        help=(
            "How to pass resolution fields in payload: "
            "auto=use size for Tongyi-MAI/Z-Image* and width/height for others; "
            "size=only size (WxH); width-height=only width+height; both=send all three."
        ),
    )
    parser.add_argument(
        "--lora",
        action="append",
        default=[],
        help=(
            "LoRA config. Single LoRA: --lora repo-id; "
            "Multiple LoRA with weights: --lora repo1=0.6 --lora repo2=0.4"
        ),
    )
    parser.add_argument(
        "--extra-json",
        help="Extra payload fields as a JSON object, e.g. '{\"foo\":1,\"bar\":\"x\"}'.",
    )

    parser.add_argument("--poll-interval", type=float, default=5.0, help="Polling interval in seconds.")
    parser.add_argument("--timeout", type=int, default=600, help="Per-task timeout in seconds.")
    parser.add_argument("--output-dir", default="result_pic", help="Directory to save images.")
    parser.add_argument("--filename-prefix", default="result_image", help="Saved filename prefix.")
    parser.add_argument("--print-payload", action="store_true", help="Print final task payload before submit.")
    parser.add_argument("--check-size", action="store_true", help="Print saved image dimensions.")

    return parser.parse_args()


def normalize_ratio(ratio: str | None) -> str | None:
    if ratio is None:
        return None

    match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", ratio)
    if not match:
        raise ValueError("--ratio must be in W:H format, e.g. 1:1 or 16:9")

    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError("--ratio values must be positive integers")

    gcd = math.gcd(width, height)
    return f"{width // gcd}:{height // gcd}"


def parse_size(size: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[xX*]\s*(\d+)\s*", size)
    if not match:
        raise ValueError("--size must be WIDTHxHEIGHT or WIDTH*HEIGHT")

    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError("--size width/height must be positive")

    return width, height


def dimensions_from_ratio(ratio: str) -> tuple[int, int]:
    mapped = RATIO_TO_DEFAULT_SIZE.get(ratio)
    if mapped is not None:
        return mapped

    width_text, height_text = ratio.split(":", 1)
    ratio_width = int(width_text)
    ratio_height = int(height_text)

    base_area = 1024 * 1024
    width = math.sqrt(base_area * ratio_width / ratio_height)
    height = math.sqrt(base_area * ratio_height / ratio_width)

    rounded_width = max(64, int(round(width / 8.0) * 8))
    rounded_height = max(64, int(round(height / 8.0) * 8))
    return rounded_width, rounded_height


def format_size(width: int, height: int) -> str:
    return f"{width}x{height}"


def prefers_size_field(model_id: str) -> bool:
    model_key = model_id.strip().lower()
    return model_key in MODELS_USING_SIZE_FIELD or model_key.startswith("tongyi-mai/z-image")


def resolve_resolution_mode(args: argparse.Namespace) -> tuple[bool, bool]:
    mode = args.resolution_mode
    if mode == "size":
        return True, False
    if mode == "width-height":
        return False, True
    if mode == "both":
        return True, True

    if prefers_size_field(args.model):
        return True, False
    return False, True


def with_aspect_ratio(prompt: str, ratio: str | None) -> str:
    if not ratio:
        return prompt
    token = f"--ar {ratio}"
    if re.search(r"--ar\s+\S+", prompt):
        return re.sub(r"--ar\s+\S+", token, prompt, count=1)
    return f"{prompt.rstrip()} {token}".strip()


def parse_loras(items: list[str]) -> str | dict[str, float] | None:
    if not items:
        return None
    if len(items) > 6:
        raise ValueError("At most 6 LoRAs are allowed.")

    has_weight = any("=" in item for item in items)
    if has_weight:
        weighted: dict[str, float] = {}
        for item in items:
            if "=" not in item:
                raise ValueError("When using weighted LoRAs, every --lora must be repo=weight format.")
            repo, weight_text = item.split("=", 1)
            repo = repo.strip()
            if not repo:
                raise ValueError(f"Invalid lora value: {item}")
            try:
                weight = float(weight_text)
            except ValueError as exc:
                raise ValueError(f"Invalid LoRA weight: {item}") from exc
            weighted[repo] = weight

        total = sum(weighted.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"LoRA weights must sum to 1.0, got {total}.")
        return weighted

    if len(items) > 1:
        raise ValueError("Without weights, only one --lora is allowed.")
    return items[0].strip()


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    normalized_ratio = normalize_ratio(args.ratio)

    payload: dict[str, Any] = {
        "model": args.model,
        "prompt": with_aspect_ratio(args.prompt, normalized_ratio),
    }

    if (args.width is None) != (args.height is None):
        raise ValueError("--width and --height must be provided together")

    final_width: int | None = None
    final_height: int | None = None

    if args.width is not None and args.height is not None:
        if args.width <= 0 or args.height <= 0:
            raise ValueError("--width and --height must be positive")
        final_width = args.width
        final_height = args.height

        if args.size is not None:
            size_width, size_height = parse_size(args.size)
            if (size_width, size_height) != (final_width, final_height):
                raise ValueError("--size conflicts with --width/--height")
    elif args.size is not None:
        final_width, final_height = parse_size(args.size)
    elif normalized_ratio is not None:
        final_width, final_height = dimensions_from_ratio(normalized_ratio)

    optional_fields = {
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
    }
    for key, value in optional_fields.items():
        if value is not None:
            payload[key] = value

    include_size, include_width_height = resolve_resolution_mode(args)
    if final_width is not None and final_height is not None:
        if include_width_height:
            payload["width"] = final_width
            payload["height"] = final_height
        if include_size:
            payload["size"] = format_size(final_width, final_height)

    loras = parse_loras(args.lora)
    if loras is not None:
        payload["loras"] = loras

    if args.extra_json:
        try:
            extra = json.loads(args.extra_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--extra-json is not valid JSON: {exc}") from exc
        if not isinstance(extra, dict):
            raise ValueError("--extra-json must be a JSON object.")
        payload.update(extra)

    return payload


def submit_task(
    session: requests.Session,
    base_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> str:
    url = f"{base_url.rstrip('/')}/v1/images/generations"
    resp = session.post(
        url,
        headers={**headers, "X-ModelScope-Async-Mode": "true"},
        json=payload,
        timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(f"Create task failed ({resp.status_code}): {resp.text[:800]}")

    data = resp.json()
    task_id = data.get("task_id")
    if not task_id:
        raise RuntimeError(f"Create task succeeded but no task_id found: {data}")
    return task_id


def poll_task(
    session: requests.Session,
    base_url: str,
    headers: dict[str, str],
    task_id: str,
    poll_interval: float,
    timeout_seconds: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/tasks/{task_id}"
    deadline = time.time() + timeout_seconds

    while True:
        if time.time() > deadline:
            raise TimeoutError(f"Task {task_id} timed out after {timeout_seconds}s")

        resp = session.get(
            url,
            headers={**headers, "X-ModelScope-Task-Type": "image_generation"},
            timeout=60,
        )
        if not resp.ok:
            raise RuntimeError(f"Poll task failed ({resp.status_code}): {resp.text[:800]}")

        data = resp.json()
        status = data.get("task_status")
        if status == "SUCCEED":
            return data
        if status == "FAILED":
            message = data.get("message") or data.get("error") or json.dumps(data, ensure_ascii=False)
            raise RuntimeError(f"Task {task_id} failed: {message}")

        time.sleep(poll_interval)


def infer_extension(image_url: str, content_type: str | None) -> str:
    path = urlparse(image_url).path
    ext = Path(path).suffix.lower()
    if ext in KNOWN_EXTENSIONS:
        return ext

    ctype = (content_type or "").split(";")[0].strip().lower()
    return CONTENT_TYPE_TO_EXT.get(ctype, ".jpg")


def download_image(
    session: requests.Session,
    image_url: str,
    target_stem: Path,
) -> Path:
    resp = session.get(image_url, timeout=120)
    resp.raise_for_status()

    ext = infer_extension(image_url, resp.headers.get("content-type"))
    output_path = target_stem.with_suffix(ext)
    output_path.write_bytes(resp.content)
    return output_path


def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for --check-size. Install with: pip install pillow") from exc

    with Image.open(image_path) as image:
        return image.width, image.height


def short_task_id(task_id: str) -> str:
    text = str(task_id)
    suffix = text[-4:]
    if len(suffix) < 4:
        return suffix.rjust(4, "0")
    return suffix


def build_run_dir_name(task_ids: list[str]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    if not task_ids:
        return f"{timestamp}_0000-0000"

    start_suffix = short_task_id(task_ids[0])
    end_suffix = short_task_id(task_ids[-1])
    return f"{timestamp}_{start_suffix}-{end_suffix}"


def run_api_mode(
    session: requests.Session,
    args: argparse.Namespace,
    headers: dict[str, str],
    base_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    payload = dict(base_payload)
    if args.num_images > 1:
        payload["n"] = args.num_images

    task_id = submit_task(session, args.base_url, headers, payload)
    print(f"Submitted task: {task_id}")
    result = poll_task(session, args.base_url, headers, task_id, args.poll_interval, args.timeout)
    return [result], [task_id]


def run_loop_mode(
    session: requests.Session,
    args: argparse.Namespace,
    headers: dict[str, str],
    base_payload: dict[str, Any],
    num_tasks: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    total = num_tasks if num_tasks is not None else args.num_images

    results: list[dict[str, Any]] = []
    task_ids: list[str] = []
    for i in range(total):
        task_id = submit_task(session, args.base_url, headers, base_payload)
        task_ids.append(task_id)
        print(f"Submitted task {i + 1}/{total}: {task_id}")
        result = poll_task(session, args.base_url, headers, task_id, args.poll_interval, args.timeout)
        results.append(result)
    return results, task_ids


def count_output_images(task_results: list[dict[str, Any]]) -> int:
    total = 0
    for result in task_results:
        total += len(result.get("output_images") or [])
    return total


def main() -> int:
    args = parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Use --api-key or edit DEFAULT_API_KEY in script.")
    if args.num_images < 1:
        raise ValueError("--num-images must be >= 1")

    payload = build_payload(args)
    if args.print_payload:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }

    with requests.Session() as session:
        if args.mode == "loop":
            task_results, task_ids = run_loop_mode(session, args, headers, payload)
        elif args.mode == "api":
            task_results, task_ids = run_api_mode(session, args, headers, payload)
            if args.num_images > 1:
                api_count = count_output_images(task_results)
                if api_count < args.num_images:
                    print(
                        f"API batch mode returned {api_count}/{args.num_images} image(s). "
                        "Try --mode loop if you need exact count.",
                        file=sys.stderr,
                    )
        else:
            try:
                task_results, task_ids = run_api_mode(session, args, headers, payload)
                if args.num_images > 1:
                    api_count = count_output_images(task_results)
                    if api_count < args.num_images:
                        remaining = args.num_images - api_count
                        print(
                            f"API batch mode returned {api_count}/{args.num_images} image(s), "
                            f"fallback to loop mode for remaining {remaining}.",
                            file=sys.stderr,
                        )
                        more_results, more_task_ids = run_loop_mode(
                            session,
                            args,
                            headers,
                            payload,
                            num_tasks=remaining,
                        )
                        task_results.extend(more_results)
                        task_ids.extend(more_task_ids)
            except Exception as exc:
                if args.num_images == 1:
                    raise
                print(f"API batch mode failed, fallback to loop mode: {exc}", file=sys.stderr)
                task_results, task_ids = run_loop_mode(session, args, headers, payload)

        image_urls: list[str] = []
        for result in task_results:
            urls = result.get("output_images") or []
            image_urls.extend(urls)

        if not image_urls:
            raise RuntimeError("Task succeeded but output_images is empty.")

        output_root = Path(args.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        run_dir = output_root / build_run_dir_name(task_ids)
        run_dir.mkdir(parents=True, exist_ok=True)

        saved: list[Path] = []
        for idx, image_url in enumerate(image_urls, start=1):
            stem = run_dir / f"{args.filename_prefix}_{idx:03d}"
            saved_path = download_image(session, image_url, stem)
            saved.append(saved_path)

            if args.check_size:
                width, height = get_image_dimensions(saved_path)
                print(f"Saved: {saved_path} ({width}x{height})")
            else:
                print(f"Saved: {saved_path}")

    print(f"Done. {len(saved)} image(s) saved to {run_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
