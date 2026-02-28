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
python -m compileall .
python -c "import torch; import transformers; from PIL import Image; import tqdm"
```

Optional dry-run against a sample folder:

```bash
python main.py --input ./images --output ./sorted --dry-run
```

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
