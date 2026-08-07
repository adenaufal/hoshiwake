"""Configuration constants for the anime image sorter."""

from pathlib import Path

# Prefer a local checkpoint when one is present; otherwise fall back to the
# Hugging Face hub id so a fresh clone works without any manual download step.
_LOCAL_MODEL_DIR = "models/siglip2-explicit"
_HF_MODEL_ID = "prithivMLmods/siglip2-x256-explicit-content"

MODEL_ID = _LOCAL_MODEL_DIR if Path(_LOCAL_MODEL_DIR).is_dir() else _HF_MODEL_ID

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

LABEL_TO_CATEGORY = {
    "Anime Picture": "SFW",
    "Normal": "SFW",
    "Hentai": "NSFW",
    "Pornography": "NSFW",
    "Enticing or Sensual": "UNCERTAIN",
}

DEFAULT_THRESHOLD = 0.65
DEFAULT_MARGIN = 0.10
DEFAULT_BATCH_SIZE = 8
DEFAULT_DEVICE = "cpu"
DEFAULT_MODE = "copy"

CATEGORIES = ["SFW", "NSFW", "UNCERTAIN"]
