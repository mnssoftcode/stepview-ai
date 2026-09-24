# StepView

StepView is a computer-vision research and edge engineering project that identifies visually suitable terrain regions for human foot placement while walking or hiking.

## Project Structure

```
StepViewProject/
├── docs/             # Technical specifications, PRD, ML specs, audits, roadmaps
├── data/             # Dataset storage (raw, images, masks, metadata, splits)
├── stepview/         # Core Python package
│   ├── data/         # Dataset loaders, schemas, and verification utilities
│   ├── models/       # Segmentation backbones and heads (PyTorch)
│   ├── postprocess/  # Candidate region generation and spatial scoring
│   ├── inference/    # Offline and exported model inference pipelines
│   └── utils/        # Geometry, visualization, and validation helpers
├── tests/            # Automated test suite (software correctness)
├── experiments/      # Experiment configurations and run logs
├── models/           # Exported checkpoints and ONNX artifacts
├── pyproject.toml    # Python project configuration and dependencies
└── README.md
```

## Quickstart

Initialize environment and run tests with `uv`:

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest
```

See [docs/00_PROJECT_MASTER.md](file:///Users/mns/Mns_Data/code/AiMl/StepViewProject/docs/00_PROJECT_MASTER.md) and [docs/01_PRD.md](file:///Users/mns/Mns_Data/code/AiMl/StepViewProject/docs/01_PRD.md) for full project documentation.
