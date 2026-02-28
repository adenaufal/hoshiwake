from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from config import CATEGORIES, LABEL_TO_CATEGORY, SUPPORTED_EXTENSIONS

SFW_LABELS = ("Anime Picture", "Normal")
NSFW_LABELS = ("Hentai", "Pornography")


def discover_images(input_dir: Path) -> list[Path]:
    """List top-level image files in the input directory."""
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_image(path: Path) -> Image.Image | None:
    """Load image as RGB. For GIF, use first frame."""
    try:
        with Image.open(path) as image:
            if path.suffix.lower() == ".gif":
                try:
                    image.seek(0)
                except EOFError:
                    pass
            return image.convert("RGB")
    except (
        FileNotFoundError,
        PermissionError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        print(f"[WARN] Skipping '{path}': {exc}")
        return None


def ensure_output_dirs(output_dir: Path) -> None:
    for category in CATEGORIES:
        (output_dir / category).mkdir(parents=True, exist_ok=True)


def _resolve_collision(destination: Path) -> Path:
    if not destination.exists():
        return destination

    stem, suffix = destination.stem, destination.suffix
    counter = 1
    while True:
        candidate = destination.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def sort_file(src: Path, output_dir: Path, category: str, mode: str) -> Path:
    destination_dir = output_dir / category
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = _resolve_collision(destination_dir / src.name)

    if mode == "copy":
        shutil.copy2(src, destination)
    elif mode == "move":
        shutil.move(str(src), str(destination))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return destination


def _score_group(all_scores: dict, labels: tuple[str, ...]) -> float:
    return float(sum(float(all_scores.get(label, 0.0)) for label in labels))


def determine_category(result: dict, threshold: float, margin: float) -> str:
    all_scores = result.get("all_scores")

    if isinstance(all_scores, dict) and all_scores:
        sfw_score = _score_group(all_scores, SFW_LABELS)
        nsfw_score = _score_group(all_scores, NSFW_LABELS)

        if nsfw_score >= threshold and (nsfw_score - sfw_score) >= margin:
            return "NSFW"
        if sfw_score >= threshold and (sfw_score - nsfw_score) >= margin:
            return "SFW"
        return "UNCERTAIN"

    if float(result.get("score", 0.0)) < threshold:
        return "UNCERTAIN"

    label = str(result.get("label", ""))
    return LABEL_TO_CATEGORY.get(label, "UNCERTAIN")
