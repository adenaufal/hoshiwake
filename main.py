from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from classifier import classify_batch, load_model
from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_MARGIN,
    DEFAULT_MODE,
    DEFAULT_THRESHOLD,
)
from reporter import print_summary, write_csv
from sorter import (
    determine_category,
    discover_images,
    ensure_output_dirs,
    load_image,
    sort_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify and sort anime images into SFW/NSFW/UNCERTAIN folders."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input image directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--mode",
        choices=["copy", "move"],
        default=DEFAULT_MODE,
        help="Whether to copy or move files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum aggregated category score required for hard SFW/NSFW decisions",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help="Minimum SFW-vs-NSFW score gap required for hard decisions",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for model inference",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default=DEFAULT_DEVICE,
        help="Inference device",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify only and generate report without moving/copying files",
    )
    return parser.parse_args()


def chunked(items: list[Path], size: int):
    if size < 1:
        raise ValueError("Batch size must be >= 1.")
    for index in range(0, len(items), size):
        yield items[index : index + size]


def resolve_device(requested_device: str) -> str:
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"

    if requested_device == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not has_mps:
            print("[WARN] MPS requested but unavailable. Falling back to CPU.")
            return "cpu"

    return requested_device


def run() -> int:
    args = parse_args()

    if not args.input.exists() or not args.input.is_dir():
        print(f"Error: input path '{args.input}' does not exist or is not a directory.")
        return 1
    if args.batch_size < 1:
        print("Error: --batch-size must be >= 1.")
        return 1
    if not (0.0 <= args.threshold <= 1.0):
        print("Error: --threshold must be between 0.0 and 1.0.")
        return 1
    if not (0.0 <= args.margin <= 1.0):
        print("Error: --margin must be between 0.0 and 1.0.")
        return 1

    image_paths = discover_images(args.input)
    if not image_paths:
        print("No supported images found in input directory.")
        return 0

    device = resolve_device(args.device)

    print("Loading model...")
    processor, model = load_model(device)

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        ensure_output_dirs(args.output)

    records: list[dict] = []

    try:
        with tqdm(total=len(image_paths), desc="Processing", unit="img") as progress:
            for batch_paths in chunked(image_paths, args.batch_size):
                loaded_images = []
                loaded_paths = []

                for path in batch_paths:
                    image = load_image(path)
                    if image is None:
                        records.append(
                            {
                                "filename": path.name,
                                "category": "UNCERTAIN",
                                "label": "",
                                "score": 0.0,
                                "all_scores": {},
                                "status": "skipped",
                            }
                        )
                        progress.update(1)
                        continue

                    loaded_images.append(image)
                    loaded_paths.append(path)

                if not loaded_images:
                    continue

                try:
                    batch_results = classify_batch(loaded_images, processor, model, device)
                    if len(batch_results) != len(loaded_paths):
                        raise RuntimeError("Classifier returned an unexpected number of results.")

                    for path, result in zip(loaded_paths, batch_results):
                        category = determine_category(result, args.threshold, args.margin)
                        status = "dry-run" if args.dry_run else "sorted"

                        if not args.dry_run:
                            sort_file(path, args.output, category, args.mode)

                        records.append(
                            {
                                "filename": path.name,
                                "category": category,
                                "label": result["label"],
                                "score": result["score"],
                                "all_scores": result["all_scores"],
                                "status": status,
                            }
                        )
                        progress.update(1)
                finally:
                    for image in loaded_images:
                        image.close()

    except KeyboardInterrupt:
        print("\nInterrupted by user. Writing partial report...")
        report_path = write_csv(records, args.output)
        print_summary(records, report_path)
        return 130

    report_path = write_csv(records, args.output)
    print_summary(records, report_path)
    return 0


def main() -> int:
    try:
        return run()
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
