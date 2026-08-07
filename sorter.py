from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from config import CATEGORIES, LABEL_TO_CATEGORY, SUPPORTED_EXTENSIONS

SFW_LABELS = ("Anime Picture", "Normal")
NSFW_LABELS = ("Hentai", "Pornography")

# NSFW keywords must be tested before SFW keywords: "sfw" is a substring of
# "nsfw", "safe" of "unsafe", and "not_safe..." spellings contain "safe", so
# the reverse order would bucket NSFW labels into the SFW group.
NSFW_KEYWORDS = (
    "nsfw",
    "unsafe",
    "not_safe",
    "not-safe",
    "not safe",
    "notsafe",
    "hentai",
    "porn",
    "adult",
    "explicit",
    "prohibit",
)
SFW_KEYWORDS = ("sfw", "safe", "allow", "normal", "anime", "neutral", "drawing", "general", "sensitive")

# Index-fallback names are matched exactly, not as substrings, so that
# "label_1" cannot match inside "label_12".
_SFW_EXACT = {label.lower() for label in SFW_LABELS} | {"label_0"}
_NSFW_EXACT = {label.lower() for label in NSFW_LABELS} | {"label_1"}


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
        Image.DecompressionBombError,
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
    if mode not in ("copy", "move"):
        raise ValueError(f"Unsupported mode: {mode}")

    destination_dir = output_dir / category
    destination_dir.mkdir(parents=True, exist_ok=True)

    candidate = destination_dir / src.name
    if candidate.exists() and src.exists() and candidate.samefile(src):
        return src  # already in the right place; avoid self-copy duplication

    destination = _resolve_collision(candidate)

    if mode == "copy":
        shutil.copy2(src, destination)
    else:
        shutil.move(str(src), str(destination))

    return destination


def score_groups(all_scores: dict) -> tuple[float, float]:
    """Aggregate per-label probabilities into (sfw_score, nsfw_score).

    Each label is bucketed exactly once: exact label names take priority,
    then keyword matching (NSFW keywords first). Labels matching neither
    group contribute to neither score.
    """
    sfw = 0.0
    nsfw = 0.0
    for name, score in all_scores.items():
        lowered = str(name).lower()
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if lowered in _NSFW_EXACT:
            nsfw += value
        elif lowered in _SFW_EXACT:
            sfw += value
        elif any(keyword in lowered for keyword in NSFW_KEYWORDS):
            nsfw += value
        elif any(keyword in lowered for keyword in SFW_KEYWORDS):
            sfw += value
    return sfw, nsfw


def determine_category(result: dict, threshold: float, margin: float) -> str:
    all_scores = result.get("all_scores")

    if isinstance(all_scores, dict) and all_scores:
        sfw_score, nsfw_score = score_groups(all_scores)

        if nsfw_score >= threshold and (nsfw_score - sfw_score) >= margin:
            return "NSFW"
        if sfw_score >= threshold and (sfw_score - nsfw_score) >= margin:
            return "SFW"
        return "UNCERTAIN"

    if float(result.get("score", 0.0)) < threshold:
        return "UNCERTAIN"

    label = str(result.get("label", ""))
    lowered_label = label.lower()
    if lowered_label in _NSFW_EXACT or any(token in lowered_label for token in NSFW_KEYWORDS):
        return "NSFW"
    if lowered_label in _SFW_EXACT or any(token in lowered_label for token in SFW_KEYWORDS):
        return "SFW"
    return LABEL_TO_CATEGORY.get(label, "UNCERTAIN")
