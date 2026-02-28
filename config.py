"""Configuration constants for the anime image sorter."""

MODEL_ID = "models/siglip2-explicit"

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
