from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from classifier import classify_batch, load_model
from config import (
    CATEGORIES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_MARGIN,
    MODEL_ID,
    DEFAULT_MODE,
    DEFAULT_THRESHOLD,
)
from reporter import print_summary, write_csv
from sorter import (
    determine_category,
    discover_images,
    ensure_output_dirs,
    load_image,
    score_groups,
    sort_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify and sort anime images into SFW/NSFW/UNCERTAIN folders."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input image directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_ID,
        help="Model path or Hugging Face model id to use for classification",
    )
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


def validate_args(args: argparse.Namespace) -> str | None:
    """Return an error message for invalid arguments, or None if they are usable."""
    if not args.input.exists() or not args.input.is_dir():
        return f"input path '{args.input}' does not exist or is not a directory."
    if args.output.exists() and not args.output.is_dir():
        return f"output path '{args.output}' exists but is not a directory."
    if args.batch_size < 1:
        return "--batch-size must be >= 1."
    if not (0.0 <= args.threshold <= 1.0):
        return "--threshold must be between 0.0 and 1.0."
    if not (0.0 <= args.margin <= 1.0):
        return "--margin must be between 0.0 and 1.0."
    if not args.model.strip():
        return "--model must not be empty."

    input_resolved = args.input.resolve()
    output_resolved = args.output.resolve()
    if input_resolved == output_resolved:
        return "--input and --output must be different directories."
    if (
        args.mode == "copy"
        and input_resolved.parent == output_resolved
        and input_resolved.name in CATEGORIES
    ):
        # Re-triaging a category folder back into the same output is fine in
        # move mode (sort_file no-ops on same-file destinations), but in copy
        # mode it would duplicate every file.
        return (
            f"--input is a category folder inside --output ('{args.input}'); "
            "copying it into itself would duplicate files. Use --mode move to re-triage."
        )
    return None


def build_record(
    path: Path,
    category: str,
    result: dict,
    status: str,
    destination: Path | None = None,
) -> dict:
    all_scores = result.get("all_scores") or {}
    sfw_score, nsfw_score = score_groups(all_scores) if all_scores else (0.0, 0.0)
    return {
        "filename": path.name,
        "category": category,
        "label": result.get("label", ""),
        "score": result.get("score", 0.0),
        "sfw_score": sfw_score,
        "nsfw_score": nsfw_score,
        "all_scores": all_scores,
        "destination": str(destination) if destination is not None else "",
        "status": status,
    }


def run() -> int:
    args = parse_args()

    error = validate_args(args)
    if error:
        print(f"Error: {error}")
        return 1
    if args.threshold == 1.0:
        print("[WARN] --threshold 1.0 is unreachable for softmax outputs; every image will be UNCERTAIN.")

    image_paths = discover_images(args.input)
    if not image_paths:
        print("No supported images found in input directory.")
        return 0

    device = resolve_device(args.device)

    print(f"Loading model '{args.model}'...")
    processor, model = load_model(device, args.model)

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        ensure_output_dirs(args.output)

    records: list[dict] = []
    exit_code = 0

    try:
        with tqdm(total=len(image_paths), desc="Processing", unit="img") as progress:
            for batch_paths in chunked(image_paths, args.batch_size):
                loaded_images = []
                loaded_paths = []

                for path in batch_paths:
                    image = load_image(path)
                    if image is None:
                        records.append(build_record(path, "UNCERTAIN", {}, "skipped"))
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
                        if result.get("error"):
                            records.append(build_record(path, "UNCERTAIN", result, "error"))
                            progress.update(1)
                            continue

                        category = determine_category(result, args.threshold, args.margin)
                        status = "dry-run" if args.dry_run else "sorted"
                        destination = None

                        if not args.dry_run:
                            try:
                                destination = sort_file(path, args.output, category, args.mode)
                            except OSError as exc:
                                print(f"[WARN] Failed to {args.mode} '{path.name}': {exc}")
                                status = "error"

                        records.append(build_record(path, category, result, status, destination))
                        progress.update(1)
                finally:
                    for image in loaded_images:
                        image.close()

    except KeyboardInterrupt:
        print("\nInterrupted by user. Writing partial report...")
        exit_code = 130
    except Exception as exc:
        print(f"\n[ERROR] Run aborted: {exc}")
        print("Writing partial report for the files processed so far...")
        exit_code = 1

    report_path = write_csv(records, args.output)
    print_summary(records, report_path)
    return exit_code


def main() -> int:
    try:
        return run()
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
