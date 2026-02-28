# TODO

## Tonight Evaluation Plan

- Re-run `electrohead` with relaxed gates:
  - `python main.py --input "L:\Backup Sementara\NovelAI\want_to_sort" --output "L:\Backup Sementara\NovelAI\sorted_electrohead_tuned" --device cuda --batch-size 64 --threshold 0.62 --margin 0.05 --mode copy --model "models/electrohead-vit-fetish-nsfw-detector"`
- Re-run `caveduck` with stricter gates:
  - `python main.py --input "L:\Backup Sementara\NovelAI\want_to_sort" --output "L:\Backup Sementara\NovelAI\sorted_caveduck_tuned" --device cuda --batch-size 64 --threshold 0.88 --margin 0.18 --mode copy --model "models/caveduck-nsfw-classifier"`
- Compare both tuned runs using manual audit:
  - False NSFW in `SFW`
  - False SFW in `NSFW`
  - NSFW inside `UNCERTAIN`

## Next Implementation Tasks

- Add asymmetric thresholds:
  - `--sfw-threshold`
  - `--nsfw-threshold`
- Add optional ensemble mode:
  - `electrohead` for stricter SFW gating
  - `caveduck` for stronger NSFW catch
- Document model-specific recommended defaults in `README.md`.
