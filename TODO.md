# TODO

## Evaluation Plan

The published benchmark predates the decision-logic fixes, and the label bucketing
it relied on only ever behaved correctly for the default 5-class model. Alternate
models need fresh numbers.

- Re-run the default model to confirm the published distribution still holds:
  - `python main.py --input "L:\Backup Sementara\NovelAI\want_to_sort" --output "L:\Backup Sementara\NovelAI\sorted_default_recheck" --device cuda --batch-size 64 --threshold 0.80 --mode copy`
- Re-run `electrohead` with relaxed gates:
  - `python main.py --input "L:\Backup Sementara\NovelAI\want_to_sort" --output "L:\Backup Sementara\NovelAI\sorted_electrohead_tuned" --device cuda --batch-size 64 --threshold 0.62 --margin 0.05 --mode copy --model "models/electrohead-vit-fetish-nsfw-detector"`
  - Its binary `sfw`/`nsfw` labels could never reach the NSFW branch before the fix,
    so any earlier tuning notes for this model are void.
- Compare tuned runs using manual audit:
  - False NSFW in `SFW`
  - False SFW in `NSFW`
  - NSFW inside `UNCERTAIN`
- Tune from the report instead of re-running inference where possible: the
  `all_scores` column holds the full probability vector per image.

## Next Implementation Tasks

- Add asymmetric thresholds:
  - `--sfw-threshold`
  - `--nsfw-threshold`
- Add optional ensemble mode:
  - `electrohead` for stricter SFW gating
  - a second model for stronger NSFW catch
- Add a `--recursive` flag for subfolder scanning.

## Notes

- `CaveduckAI/nsfw-classifier` has been removed from Hugging Face. The `timm`
  ConvNeXt backend in `classifier.py` still loads a previously downloaded local
  copy, but the model can no longer be fetched, so it is out of the evaluation
  plan. Drop the backend (and the `timm`/`torchvision` dependencies) if no local
  copy is worth keeping.
