# Contributing

Thanks for contributing to `hoshiwake`.

## Development Setup

1. Fork and clone the repository.
2. Install Python 3.10+.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Checks

Run these checks before opening a pull request:

```bash
python -m compileall -q main.py classifier.py sorter.py reporter.py config.py tests
python -c "import torch; import transformers; from PIL import Image; import tqdm"
python -m unittest discover -s tests
```

The unit tests cover the decision logic, file sorting, and reporting. They need
neither a model nor a GPU, so they run in seconds — the `torch`/`transformers`
import line is the only check that needs the heavy dependencies installed.

Optional dry-run against a folder of your own sample images:

```bash
python main.py --input /path/to/sample-images --output ./sorted --dry-run
```

## Testing Guidelines

- Add a test to `tests/` for every bug fix, asserting the old behavior would fail.
- Label-bucketing changes in `sorter.py` are the highest-risk area: keyword
  matching is substring-based, so any new keyword must be checked against the
  opposite bucket's names (`sfw` is a substring of `nsfw`, `safe` of `unsafe` and
  `not_safe_for_work`). Cover both spellings when you touch those lists.
- Keep tests free of model downloads and network access.

## Pull Request Guidelines

- Keep changes focused and small.
- Update `README.md` when behavior or CLI options change.
- Include reproduction steps for bug fixes.
- Include benchmark notes if you modify model decision logic.

## Issue Reports

When filing issues, include:

- Command used
- Platform and Python version
- GPU/CPU device details
- Relevant terminal output
